"""
GameCover - screen reader + auto-collector for a Graal-style game.

Setup:   pip install mss opencv-python numpy pydirectinput
Run:     python main.py

Hotkeys (click the preview window first, except F8):
    F8     start/stop auto-collect for ACTIVE_JOB (works while the game has focus)
    M      switch job (trash / mushroom) while stopped
    S      save current frame + YOLO labels to dataset/
    T      crop a new template from the current frame
    P      guided capture of the 4 player facing templates (do it away from map edges)
    + / -  raise / lower the match threshold
    G      toggle tile grid
    [ / ]  shrink / grow tile size (1 px per press)
    J/L    nudge the player anchor left / right
    I/K    nudge the player anchor up / down
    Q/Esc  quit

Auto-collect only runs while the GAME window is focused: click the game, then press F8.
"""
import atexit
import ctypes
import math
import sys
import threading
import time
from collections import namedtuple
from pathlib import Path

import cv2
import mss
import numpy as np
import pydirectinput

pydirectinput.PAUSE = 0

# =============================== SETTINGS ===============================
WINDOW_TITLE = "Graal"      # part of your game window's title (case-insensitive)
THRESHOLD = 0.80            # template match threshold (0..1)
TEMPLATE_SCALE = 1.0        # resize sprite PNGs to their on-screen size (2.0 = twice as big)
MAX_HITS = 300              # keep only the best N raw hits per template (speed guard)
                            # (matching is done in grayscale: ~8x faster than color)
TILE_SIZE = 32              # on-screen tile size in pixels (calibrate with [ and ])
SHOW_GRID = True            # draw the tile grid centered on the player

# --- what to collect ---
JOBS = {                    # template filenames, prefix match (tire_2.png counts as "tire")
    "trash": ["tire", "paper", "newspaper", "bottle"],
    "mushroom": ["mushroom"],   # every color pays the same: mushroom_red.png etc. all match
}
ACTIVE_JOB = "trash"        # switch while stopped with M
EQUIP_ON_START = True       # press the slot key once when starting, to equip the tool
SLOT_KEYS = {"trash": "1", "mushroom": "1"}   # inventory slot of the trash pick / mushroom knife
PICKUP_KEY = "space"
PICKUP_TAP_S = 0.05
PICKUP_WAIT_S = 0.6         # wait this long after pressing pickup before checking
VERIFY_WINDOW_S = 0.6       # the target must stay gone this long to count as picked up
PICKUP_RETRIES = 3          # presses per target before giving up on it
VERIFY_RADIUS_PX = 24

# --- walking ---
KEYS = {"left": "a", "right": "d", "up": "w", "down": "s"}
SPEED_PX_S = 220            # walking speed from calibration.json
DIAGONAL_SPEED_FACTOR = 1.0 # per-axis speed when moving diagonally (try 0.71 if it overshoots)
STANDOFF_PX = 32            # stop this far from the target's center (32 = one tile in front)
DEADBAND_PX = 10            # an axis counts as lined up within this many pixels
REACH_TOL_PX = 20           # how far from the ideal spot in front of the target still counts as in reach
FACE_TARGET = True          # nudge toward the target if the character is not already facing it
FACE_TAP_S = 0.03
MATCH_RADIUS_PX = 96        # how far a detection may jump and still be the same target
LOST_SECONDS = 1.5          # give up if the target is not seen for this long
STUCK_SECONDS = 1.0         # skip a target if no progress for this long (blocked)
NO_TARGET_SECONDS = 1.5     # stop when nothing is left in view for this long
IGNORE_SECONDS = 30         # how long to skip a target we could not reach / pick up
IGNORE_RADIUS_PX = 48
REQUIRE_FOCUS = True        # stop auto-collect if the game window loses focus
VK_F8 = 0x77

# --- player tracking (the camera stops following near map edges) ---
PLAYER_PREFIX = "player"    # player_down.png, player_up.png, player_left.png, player_right.png (+ frames)
PLAYER_THRESHOLD = 0.70     # player crops include background, so use a lower match threshold
PLAYER_HOLD_S = 0.5         # keep the last known position this long if a frame misses
PLAYER_JUMP_PX = 200        # ignore "player" hits this far from the last position (other players)
PLAYER_SEARCH_PX = 200      # once found, only search this far around the last position
PLAYER_DIRS = ["down", "up", "left", "right"]
CAPTURE_SECONDS = 2.0       # how long P captures frames for each direction

TEMPLATE_DIR = Path("templates")
DATASET_DIR = Path("dataset")
# =========================================================================

# ------------------------------ Windows helpers ------------------------------
IS_WIN = sys.platform == "win32"
user32 = None
if IS_WIN:
    import ctypes.wintypes as wt
    try:                                    # report real pixel sizes on scaled (125%/150%) displays
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except Exception:
        ctypes.windll.user32.SetProcessDPIAware()
    user32 = ctypes.windll.user32


def list_windows():
    """[(hwnd, title)] for all visible windows with a title."""
    found = []

    @ctypes.WINFUNCTYPE(wt.BOOL, wt.HWND, wt.LPARAM)
    def callback(hwnd, _):
        if user32.IsWindowVisible(hwnd):
            n = user32.GetWindowTextLengthW(hwnd)
            if n:
                buf = ctypes.create_unicode_buffer(n + 1)
                user32.GetWindowTextW(hwnd, buf, n + 1)
                found.append((hwnd, buf.value))
        return True

    user32.EnumWindows(callback, 0)
    return found


def find_window(part):
    for hwnd, title in list_windows():
        if part.lower() in title.lower():
            return hwnd, title
    return None, None


def client_region(hwnd):
    """Screen rectangle of the window's drawable area (no title bar/borders)."""
    rect = wt.RECT()
    user32.GetClientRect(hwnd, ctypes.byref(rect))
    pt = wt.POINT(0, 0)
    user32.ClientToScreen(hwnd, ctypes.byref(pt))
    return {"left": pt.x, "top": pt.y, "width": rect.right, "height": rect.bottom}


def grab(sct, hwnd):
    return np.ascontiguousarray(np.array(sct.grab(client_region(hwnd)))[:, :, :3])


# ------------------------------ templates & detection ------------------------------
Template = namedtuple("Template", "img prep")      # img: gray uint8; prep: data for masked matching, or None


def masked_prep(gray, opaque):
    """Pre-compute what the fast masked match needs: mask, pixel count, zero-mean masked template."""
    M = opaque.astype(np.float32)
    N = float(M.sum())
    if N < 8:
        return None
    t = gray.astype(np.float32)
    Tp = (t - (t * M).sum() / N) * M
    denT = float(np.sqrt((Tp * Tp).sum()))
    return None if denT < 1e-3 else (M, N, Tp, denT)


def load_templates():
    """name -> Template. Transparent PNGs are matched ignoring their see-through pixels."""
    TEMPLATE_DIR.mkdir(exist_ok=True)
    templates = {}
    for p in sorted(TEMPLATE_DIR.glob("*.png")):
        img = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
        if img is None:
            continue
        if TEMPLATE_SCALE != 1.0:           # nearest-neighbor keeps pixel art crisp
            img = cv2.resize(img, None, fx=TEMPLATE_SCALE, fy=TEMPLATE_SCALE,
                             interpolation=cv2.INTER_NEAREST)
        alpha = None
        if img.ndim == 3 and img.shape[2] == 4:
            alpha, img = img[:, :, 3], img[:, :, :3]
        gray = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        prep = None
        if alpha is not None and (alpha == 0).any():
            prep = masked_prep(gray, alpha > 0)
            if prep is None:
                print(f"skipping {p.name}: too few visible pixels or no detail")
                continue
        templates[p.stem] = Template(np.ascontiguousarray(gray), prep)
    return templates


def match_one(tpl, plane, thr, offset):
    """Match one template in a search plane -> [(score, x, y, w, h)] in full-frame coordinates.
    plane = [gray, float32 or None, float32 squared or None] (floats are filled in on demand)."""
    gray = plane[0]
    h, w = tpl.img.shape
    if gray.shape[0] < h or gray.shape[1] < w:
        return []
    if tpl.prep is None:
        res = cv2.matchTemplate(gray, tpl.img, cv2.TM_CCOEFF_NORMED)
    else:
        # masked match (normalized cross-correlation) from 3 fast plain correlations;
        # identical to OpenCV's masked mode but ~20x faster
        if plane[1] is None:
            plane[1] = gray.astype(np.float32)
            plane[2] = plane[1] * plane[1]
        M, N, Tp, denT = tpl.prep
        num = cv2.matchTemplate(plane[1], Tp, cv2.TM_CCORR)
        s1 = cv2.matchTemplate(plane[1], M, cv2.TM_CCORR)
        s2 = cv2.matchTemplate(plane[2], M, cv2.TM_CCORR)
        var = s2 - s1 * s1 / N
        res = np.where(var > 4.0 * N, num / (denT * np.sqrt(np.maximum(var, 1e-6))), 0.0)
    res = np.nan_to_num(res, nan=0.0, posinf=0.0, neginf=0.0)
    ys, xs = np.where(res >= thr)
    if len(xs) == 0:
        return []
    scores = res[ys, xs]
    if len(xs) > MAX_HITS:                  # pathological case (very low threshold): keep the best
        top = np.argpartition(scores, -MAX_HITS)[-MAX_HITS:]
        xs, ys, scores = xs[top], ys[top], scores[top]
    boxes = [[int(x), int(y), w, h] for x, y in zip(xs, ys)]
    keep = cv2.dnn.NMSBoxes(boxes, scores.astype(float).tolist(), thr, 0.3)   # merge duplicate hits
    ox, oy = offset
    return [(float(scores[i]), boxes[i][0] + ox, boxes[i][1] + oy, w, h)
            for i in np.array(keep).flatten()]


def detect(img, templates, threshold, names=None, roi_for=None):
    """img (gray) -> [(label, score, x, y, w, h)].
    names: run only these templates (None = all).
    roi_for(name) -> (x0, y0, x1, y1) window to search for that template, or None for the whole frame."""
    results, planes = [], {}
    for name, tpl in templates.items():
        if names is not None and name not in names:
            continue
        roi = roi_for(name) if roi_for else None
        if roi not in planes:
            planes[roi] = [img if roi is None else img[roi[1]:roi[3], roi[0]:roi[2]], None, None]
        thr = PLAYER_THRESHOLD if name.startswith(PLAYER_PREFIX) else threshold
        off = (roi[0], roi[1]) if roi else (0, 0)
        results += [(name,) + h for h in match_one(tpl, planes[roi], thr, off)]
    return results


def plan_search(templates, walker, tracker, px, py, fw, fh):
    """-> (names, roi_for). Idle: everything (the player in a small window). Collecting: only the
    player plus either all job targets (looking for the next one) or just the current target,
    searched in a small window around where we expect it. px, py = player position (last frame)."""
    p_roi = tracker.roi(fw, fh)
    if not walker.active:
        return None, lambda n: p_roi if n.startswith(PLAYER_PREFIX) else None
    focus = walker.focus(px, py, fw, fh)
    prefixes = (PLAYER_PREFIX,) + ((focus[0],) if focus else tuple(JOBS[ACTIVE_JOB]))
    names = {n for n in templates if n.startswith(prefixes)}
    return names, lambda n: p_roi if n.startswith(PLAYER_PREFIX) else (focus[1] if focus else None)


# ------------------------------ preview, saving, capturing ------------------------------
def draw(frame, detections, px, py, tile, grid):
    """Boxes: green = current job target, orange = other, blue = player. px,py = player anchor."""
    out = frame.copy()
    H, W = out.shape[:2]
    if grid:
        x0, y0 = px - tile // 2, py - tile // 2          # top-left of the player's tile
        for x in range(x0 % tile, W, tile):
            cv2.line(out, (x, 0), (x, H), (200, 200, 200), 1)
        for y in range(y0 % tile, H, tile):
            cv2.line(out, (0, y), (W, y), (200, 200, 200), 1)
        cv2.rectangle(out, (x0, y0), (x0 + tile, y0 + tile), (0, 0, 255), 2)
    job = tuple(JOBS[ACTIVE_JOB])
    for name, score, x, y, w, h in detections:
        if name.startswith(PLAYER_PREFIX):
            color, text = (255, 200, 0), name
        else:
            color = (0, 255, 0) if name.startswith(job) else (0, 200, 255)
            dx, dy = round((x + w / 2 - px) / tile), round((y + h / 2 - py) / tile)
            text = f"{name} ({dx:+d},{dy:+d})"           # tile offset from the player
        cv2.rectangle(out, (x, y), (x + w, y + h), color, 2)
        cv2.putText(out, text, (x, max(12, y - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
    return out


def save_frame(frame, detections, class_names):
    """Save image + YOLO-format labels (class cx cy w h, normalized) to dataset/."""
    (DATASET_DIR / "images").mkdir(parents=True, exist_ok=True)
    (DATASET_DIR / "labels").mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    H, W = frame.shape[:2]
    cv2.imwrite(str(DATASET_DIR / "images" / f"{stamp}.png"), frame)
    with open(DATASET_DIR / "labels" / f"{stamp}.txt", "w") as f:
        for name, _, x, y, w, h in detections:
            f.write(f"{class_names.index(name)} {(x + w / 2) / W:.6f} {(y + h / 2) / H:.6f} "
                    f"{w / W:.6f} {h / H:.6f}\n")
    (DATASET_DIR / "classes.txt").write_text("\n".join(class_names))
    print(f"saved {stamp}  ({len(detections)} boxes)")


def crop_template(frame):
    """Draw a box around one sprite in a frozen frame, then name it."""
    title = "Select sprite (Enter=ok, C=cancel)"
    x, y, w, h = map(int, cv2.selectROI(title, frame, showCrosshair=False))
    cv2.destroyWindow(title)
    if w == 0 or h == 0:
        return
    name = input("Template name (e.g. tire): ").strip()
    if name:
        TEMPLATE_DIR.mkdir(exist_ok=True)
        cv2.imwrite(str(TEMPLATE_DIR / f"{name}.png"), frame[y:y + h, x:x + w])
        print(f"saved templates/{name}.png")


def capture_player_templates(sct, hwnd):
    """Guided capture of the 4 facing templates (player_down/up/left/right).
    The character must be at the screen center (camera following), so do this away from
    the map edges. Existing player_<direction>.png files are overwritten."""
    frame = grab(sct, hwnd)
    title = "Draw a tight box around your character (Enter = ok, C = cancel)"
    x, y, w, h = map(int, cv2.selectROI(title, frame, showCrosshair=False))
    cv2.destroyWindow(title)
    if w == 0 or h == 0:
        print("Cancelled.")
        return
    H, W = frame.shape[:2]
    ox, oy = x - W // 2, y - H // 2            # box position relative to the screen center
    TEMPLATE_DIR.mkdir(exist_ok=True)
    for d in PLAYER_DIRS:
        print(f"\nClick the GAME window and make your character face {d.upper()}.")
        for n in (3, 2, 1):
            print(f"  capturing in {n}...  (walk {d} a little during the capture, or just stand)")
            time.sleep(1)
        saved, t_end = [], time.time() + CAPTURE_SECONDS
        while time.time() < t_end:
            fr = grab(sct, hwnd)
            fh, fw = fr.shape[:2]
            x0, y0 = fw // 2 + ox, fh // 2 + oy
            if x0 >= 0 and y0 >= 0:
                crop = fr[y0:y0 + h, x0:x0 + w]
                # keep only frames that differ from the ones already saved (walk animation)
                if crop.shape[:2] == (h, w) and len(saved) < 4 and all(
                        np.abs(crop.astype(int) - s.astype(int)).mean() > 4 for s in saved):
                    saved.append(crop.copy())
            time.sleep(0.2)
        for i, crop in enumerate(saved):
            name = f"{PLAYER_PREFIX}_{d}" + ("" if i == 0 else f"_{i + 1}")
            cv2.imwrite(str(TEMPLATE_DIR / f"{name}.png"), crop)
        print(f"  saved {len(saved)} frame(s) for {d}")
    print("\nDone. Turn your character around and watch 'facing:' in the preview.")


# ------------------------------ player tracking ------------------------------
class PlayerTracker:
    """Finds the character on screen and which way it faces. Normally it sits at the screen
    center, but near the map edges the camera stops following and the character walks
    off-center. The facing comes from which template matched (player_down...)."""

    def __init__(self):
        self.pos = None
        self.seen_at = 0.0
        self.facing = None

    def roi(self, fw, fh):
        """Search window around the last known position (None = search the whole frame)."""
        if self.pos is None or time.perf_counter() - self.seen_at > PLAYER_HOLD_S:
            return None
        x, y, r = self.pos[0], self.pos[1], PLAYER_SEARCH_PX
        return max(0, int(x - r)), max(0, int(y - r)), min(fw, int(x + r)), min(fh, int(y + r))

    def update(self, detections, fw, fh, nudge_x, nudge_y):
        """-> (px, py, found). Falls back to the screen center if no player template matches."""
        now = time.perf_counter()
        fresh = self.pos is not None and now - self.seen_at <= PLAYER_HOLD_S
        cands = [(d[0], d[1], d[2] + d[4] / 2, d[3] + d[5] / 2) for d in detections
                 if d[0].startswith(PLAYER_PREFIX)]
        if cands:
            ref = self.pos if fresh else (fw / 2, fh / 2)
            _, _, x, y = min(cands, key=lambda c: math.hypot(c[2] - ref[0], c[3] - ref[1]))
            if not fresh or math.hypot(x - self.pos[0], y - self.pos[1]) < PLAYER_JUMP_PX:
                # several facing templates can match the same spot: the best score wins
                same = [c for c in cands if math.hypot(c[2] - x, c[3] - y) < 24]
                name, _, bx, by = max(same, key=lambda c: c[1])
                parts = name.split("_")
                self.facing = parts[1] if len(parts) > 1 and parts[1] in PLAYER_DIRS else None
                self.pos, self.seen_at, fresh = (bx, by), now, True
        if not fresh:
            self.facing = None
        x, y = self.pos if fresh else (fw / 2, fh / 2)
        return int(round(x)) + nudge_x, int(round(y)) + nudge_y, fresh


# ------------------------------ auto-collect ------------------------------
class AutoWalker:
    """Collects every target of the active job, one after another:
    find the nearest -> walk there (continuous, diagonal) -> stop one tile in front and
    face it -> press the pickup key -> check it disappeared (retry if not) -> next.

    - Holds W/A/S/D instead of tapping; a fast background thread releases keys at
      scheduled times, so stopping is precise even if screen detection is slow.
    - Between detections the target position is dead-reckoned from our own walking speed,
      so a missed frame does not stutter the movement.
    - It only presses pickup when the target is REALLY detected in front of the character,
      and checks the result at the target's real position.
    - Targets it cannot reach or cannot pick up are remembered and skipped.
    """

    def __init__(self):
        self.active = False
        self.lock = threading.Lock()
        self.held = {}              # key -> time (perf_counter) when it should be released
        self.moved = [0.0, 0.0]     # pixels walked since the last update()
        self.state = "SEEK"         # SEEK -> WALK -> PICKUP -> VERIFY -> SEEK ...
        self.target = None          # estimated target offset from the player (px)
        self.label = None           # category of the current target, e.g. "tire"
        self.axis = None            # 'x' or 'y': the side we approach from
        self.facing = None          # set every frame from the player templates
        self.last_seen = 0.0
        self.best = None
        self.t_improve = 0.0
        self.attempts = 0
        self.reach_fails = 0
        self.t_pick = 0.0
        self.pick_at = None         # where the target REALLY was when we pressed pickup
        self.idle_since = None
        self.phantom = [0.0, 0.0]   # movement we commanded but that made no progress (blocked)
        self.ignore = []            # [ox, oy, expires_at, last_seen_at]: targets we gave up on
        self.collected = {}         # label -> how many picked up
        threading.Thread(target=self._timer, daemon=True).start()

    # ---- key handling (thread-safe) ----
    def _vel_locked(self):
        vx = (KEYS["right"] in self.held) - (KEYS["left"] in self.held)
        vy = (KEYS["down"] in self.held) - (KEYS["up"] in self.held)
        speed = SPEED_PX_S * (DIAGONAL_SPEED_FACTOR if vx and vy else 1.0)
        return vx * speed, vy * speed

    def _timer(self):
        """Releases keys at their scheduled time and measures how far we walked."""
        last = time.perf_counter()
        while True:
            time.sleep(0.004)
            now = time.perf_counter()
            dt, last = now - last, now
            with self.lock:
                vx, vy = self._vel_locked()
                self.moved[0] += vx * dt
                self.moved[1] += vy * dt
                for key, deadline in list(self.held.items()):
                    if now >= deadline:
                        pydirectinput.keyUp(key)
                        del self.held[key]

    def release(self):
        with self.lock:
            for key in self.held:
                pydirectinput.keyUp(key)
            self.held.clear()

    def _press(self, key, release_at):
        with self.lock:
            if key not in self.held:
                pydirectinput.keyDown(key)
            self.held[key] = release_at

    def _drop(self, key):
        with self.lock:
            if key in self.held:
                pydirectinput.keyUp(key)
                del self.held[key]

    @staticmethod
    def _tap(key, seconds):
        pydirectinput.keyDown(key)
        time.sleep(seconds)
        pydirectinput.keyUp(key)

    # ---- bookkeeping ----
    def _reset_target(self):
        self.target = self.label = self.axis = self.best = None
        self.attempts = self.reach_fails = 0
        self.phantom = [0.0, 0.0]

    def _ignored(self, ox, oy):
        return any(math.hypot(ox - e[0], oy - e[1]) < IGNORE_RADIUS_PX for e in self.ignore)

    def _give_up(self, msg, undo=None):
        """Skip the current target for a while and look for another.
        undo = movement that never really happened (we were blocked): cancel it in the
        remembered positions that we could not see, so they do not drift."""
        if undo:
            for e in self.ignore:
                if e[3] < self.t_improve:
                    e[0] += undo[0]
                    e[1] += undo[1]
        if self.target is not None:
            now = time.perf_counter()
            self.ignore.append([self.target[0], self.target[1], now + IGNORE_SECONDS, now])
        print(msg)
        self.release()
        self._reset_target()
        self.state = "SEEK"

    def _summary(self):
        done = ", ".join(f"{k} x{v}" for k, v in self.collected.items()) or "nothing"
        return f"Collected: {done}"

    def toggle(self):
        self.active = not self.active
        self.release()
        self._reset_target()
        self.ignore, self.collected, self.idle_since = [], {}, None
        self.state = "SEEK"
        with self.lock:
            self.moved = [0.0, 0.0]
        if self.active:
            if EQUIP_ON_START:
                self._tap(SLOT_KEYS[ACTIVE_JOB], 0.05)       # equip the tool (trash pick / knife)
                time.sleep(0.3)
            print(f"AUTO-COLLECT ON  job={ACTIVE_JOB}  targets={', '.join(JOBS[ACTIVE_JOB])}")
        else:
            print("AUTO-COLLECT OFF. " + self._summary())

    def stop(self, msg):
        self.release()
        self.active = False
        print(msg + " " + self._summary())

    def _in_reach(self, ox, oy):
        """Is a target at offset (ox, oy) about one standoff away in front of us?"""
        along, across = (ox, oy) if self.axis == "x" else (oy, ox)
        return abs(abs(along) - STANDOFF_PX) <= REACH_TOL_PX and abs(across) <= REACH_TOL_PX

    def _face_target(self):
        """Arrived: make sure the character faces the target, then go to PICKUP."""
        tx, ty = self.target
        if self.axis == "x":
            key = KEYS["right"] if tx > 0 else KEYS["left"]
        else:
            key = KEYS["down"] if ty > 0 else KEYS["up"]
        want = {v: k for k, v in KEYS.items()}[key]          # 'left' / 'right' / 'up' / 'down'
        if FACE_TARGET and self.facing != want:
            self._tap(key, FACE_TAP_S)                       # tiny nudge toward the target
        self.state = "PICKUP"

    def focus(self, px, py, fw, fh):
        """While walking to / picking up a target we only need to look for that one target,
        in a window around where we expect it. -> (label, (x0, y0, x1, y1)) or None (= look for
        any job target on the whole screen)."""
        if not self.active or self.state == "SEEK" or self.target is None:
            return None
        if self.state == "VERIFY":
            (cx, cy), r = self.pick_at, VERIFY_RADIUS_PX
        else:
            (cx, cy), r = self.target, MATCH_RADIUS_PX + min(300, SPEED_PX_S * (time.perf_counter() - self.last_seen))
        r += 48                                         # margin: sprite size + movement since last frame
        sx, sy = px + cx, py + cy
        return self.label, (max(0, int(sx - r)), max(0, int(sy - r)), min(fw, int(sx + r)), min(fh, int(sy + r)))

    # ---- main entry point ----
    def update(self, detections, px, py, t_grab):
        """Call once per frame. t_grab = perf_counter() when the frame was captured."""
        if not self.active:
            return
        now = time.perf_counter()
        with self.lock:
            mx, my = self.moved
            self.moved = [0.0, 0.0]
            was_moving = bool(self.held)
            vx, vy = self._vel_locked()

        # everything remembered is relative to the player: shift it by how far we walked
        if self.target is not None:
            self.target = (self.target[0] - mx, self.target[1] - my)
        for e in self.ignore:
            e[0] -= mx
            e[1] -= my
        self.ignore = [e for e in self.ignore if e[2] > now]
        if self.state == "WALK":
            self.phantom[0] += mx
            self.phantom[1] += my

        # visible targets of the active job: (category, dx, dy) from the player,
        # shifted forward to "now" to cancel the processing delay
        labels = tuple(JOBS[ACTIVE_JOB])
        lag = now - t_grab
        cands = []
        for d in detections:
            for p in labels:
                if d[0].startswith(p):
                    cands.append((p, d[2] + d[4] / 2 - px - vx * lag, d[3] + d[5] / 2 - py - vy * lag))
                    break

        for e in self.ignore:                       # re-anchor skipped items on the real thing
            for c in cands:
                if math.hypot(c[1] - e[0], c[2] - e[1]) < IGNORE_RADIUS_PX:
                    e[0], e[1], e[3] = c[1], c[2], now
                    break

        # ---------------- SEEK: nearest target we have not given up on ----------------
        if self.state == "SEEK":
            free = [c for c in cands if not self._ignored(c[1], c[2])]
            if not free:
                if self.idle_since is None:
                    self.idle_since = now
                elif now - self.idle_since > NO_TARGET_SECONDS:
                    self.stop("No more targets in view.")
                return
            self.idle_since = None
            self.label, ox, oy = min(free, key=lambda c: abs(c[1]) + abs(c[2]))
            self.target = (ox, oy)
            self.axis = "x" if abs(ox) >= abs(oy) else "y"
            self.last_seen, self.best, self.t_improve, self.attempts = now, None, now, 0
            self.state = "WALK"
            return
        if self.target is None:
            self.state = "SEEK"
            return

        # is our target still visible near where we expect it? (search wider the longer it was unseen)
        if self.state == "VERIFY":
            radius, (tx, ty) = VERIFY_RADIUS_PX, self.pick_at
        else:
            radius = MATCH_RADIUS_PX + min(300, SPEED_PX_S * (now - self.last_seen))
            tx, ty = self.target
        near = None
        same = [c for c in cands if c[0] == self.label]
        if same:
            c = min(same, key=lambda c: math.hypot(c[1] - tx, c[2] - ty))
            if math.hypot(c[1] - tx, c[2] - ty) <= radius:
                near = c

        # ---------------- PICKUP: only when it is REALLY in front of us ----------------
        if self.state == "PICKUP":
            if near is None or not self._in_reach(near[1], near[2]):
                self.reach_fails += 1
                if self.reach_fails >= 3:
                    self._give_up(f"Could not get in front of the {self.label} - skipping it.")
                else:
                    if near is not None:
                        self.target = (near[1], near[2])
                    self.state, self.last_seen = "WALK", now
                return
            self.pick_at = (near[1], near[2])               # where it REALLY is, measured now
            self.release()
            self._tap(PICKUP_KEY, PICKUP_TAP_S)
            self.attempts += 1
            self.t_pick = time.perf_counter()
            self.state = "VERIFY"
            return

        # ---------------- VERIFY: did it disappear? ----------------
        if self.state == "VERIFY":
            waited = now - self.t_pick
            if waited < PICKUP_WAIT_S:
                return
            if near is not None:                            # still there
                if self.attempts < PICKUP_RETRIES:
                    self.state = "PICKUP"
                else:
                    self._give_up(f"Could not pick up {self.label} after {self.attempts} tries - skipping it.")
                return
            if waited >= PICKUP_WAIT_S + VERIFY_WINDOW_S:   # gone for the whole window
                self.collected[self.label] = self.collected.get(self.label, 0) + 1
                print(f"Picked up {self.label}  ({sum(self.collected.values())} total)")
                self._reset_target()
                self.state = "SEEK"
            return

        # ---------------- WALK: continuous, diagonal, stop one tile in front ----------------
        seen = near is not None
        if seen:
            self.target, self.last_seen = (near[1], near[2]), now
        if now - self.last_seen > LOST_SECONDS:
            print(f"Lost sight of the {self.label}.")
            self.release()
            self._reset_target()
            self.state = "SEEK"
            return

        tx, ty = self.target
        if self.axis == "x":
            gx, gy = tx - math.copysign(STANDOFF_PX, tx), ty
        else:
            gx, gy = tx, ty - math.copysign(STANDOFF_PX, ty)

        # blocked? only trust real detections, not dead-reckoning
        if not was_moving:
            self.t_improve = now
            self.phantom = [0.0, 0.0]
        if seen:
            dist = abs(gx) + abs(gy)
            if self.best is None or dist < self.best - 2:
                self.best, self.t_improve = dist, now
                self.phantom = [0.0, 0.0]
            elif was_moving and now - self.t_improve > STUCK_SECONDS:
                self._give_up(f"Blocked on the way to the {self.label} - skipping it "
                              "(pathfinding will fix this later).", undo=self.phantom)
                return

        # per-axis control: hold a key until that axis is lined up (diagonal = both held)
        n_active = (abs(gx) > DEADBAND_PX) + (abs(gy) > DEADBAND_PX)
        speed = SPEED_PX_S * (DIAGONAL_SPEED_FACTOR if n_active == 2 else 1.0)
        for g, plus, minus in ((gx, KEYS["right"], KEYS["left"]),
                               (gy, KEYS["down"], KEYS["up"])):
            want, other = (plus, minus) if g > 0 else (minus, plus)
            self._drop(other)
            if abs(g) > DEADBAND_PX:
                self._press(want, now + abs(g) / speed)     # auto-release when we should arrive
            else:
                self._drop(want)

        if n_active == 0 and not was_moving:
            self._face_target()


# ------------------------------ main loop ------------------------------
def has_player_templates(templates):
    return any(n.startswith(PLAYER_PREFIX) for n in templates)


def try_start(walker, templates):
    """F8 pressed while stopped: check the templates exist, then start."""
    missing = [p for p in JOBS[ACTIVE_JOB] if not any(n.startswith(p) for n in templates)]
    if missing:
        print(f"No template yet for: {', '.join(missing)} (save them in templates/)")
    if len(missing) < len(JOBS[ACTIVE_JOB]):
        walker.toggle()
    else:
        print(f"Cannot start: no templates for job '{ACTIVE_JOB}'.")


def main():
    global THRESHOLD, ACTIVE_JOB
    hwnd, title = find_window(WINDOW_TITLE)
    if not hwnd:
        print(f'No window with "{WINDOW_TITLE}" in its title. Visible windows:')
        for _, t in list_windows():
            print("  -", t)
        print("Set WINDOW_TITLE at the top of main.py to match one of these.")
        return
    print(f"Capturing: {title}")

    templates = load_templates()
    if not templates:
        print("No templates yet. Press T on the preview to crop your first sprite.")

    preview, placed = "GameCover preview", False
    tile, show_grid = TILE_SIZE, SHOW_GRID
    nudge_x = nudge_y = 0                   # shifts the player anchor, in pixels
    walker, tracker = AutoWalker(), PlayerTracker()
    atexit.register(walker.release)         # never leave a key stuck down
    prev_f8, last_t = False, time.perf_counter()
    px = py = 0                             # player position from the previous frame

    try:
        with mss.mss() as sct:
            while True:
                region = client_region(hwnd)
                if region["width"] < 10 or region["height"] < 10:     # minimized
                    time.sleep(0.2)
                    continue

                frame = np.ascontiguousarray(np.array(sct.grab(region))[:, :, :3])
                t_grab = time.perf_counter()
                dt, last_t = t_grab - last_t, t_grab
                fh, fw = frame.shape[:2]

                # detect: everything when idle; while collecting only the player + the current target
                img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                names, roi_for = plan_search(templates, walker, tracker, px, py, fw, fh)
                detections = detect(img, templates, THRESHOLD, names, roi_for)
                px, py, player_found = tracker.update(detections, fw, fh, nudge_x, nudge_y)
                walker.facing = tracker.facing if player_found else None

                # F8 works even when the game has focus
                f8 = bool(user32.GetAsyncKeyState(VK_F8) & 0x8000)
                if f8 and not prev_f8:
                    if walker.active:
                        walker.toggle()
                    else:
                        try_start(walker, templates)
                prev_f8 = f8

                # safety stops
                if walker.active and has_player_templates(templates) and not player_found:
                    walker.stop("Lost sight of the player - stopped for safety.")
                if walker.active and REQUIRE_FOCUS and user32.GetForegroundWindow() != hwnd:
                    walker.stop("Game window is not focused - stopped. Click the game, then press F8.")

                walker.update(detections, px, py, t_grab)

                # preview
                shown = draw(frame, detections, px, py, tile, show_grid)
                status = f"thr {THRESHOLD:.2f}  tile {tile}px  hits {len(detections)}  {dt * 1000:.0f}ms"
                status += (f"  AUTO {ACTIVE_JOB}:{walker.state} ({sum(walker.collected.values())})"
                           if walker.active else f"  job:{ACTIVE_JOB}")
                if tracker.facing:
                    status += f"  facing:{tracker.facing}"
                if not player_found:
                    status += "  player=center (no match)"
                cv2.putText(shown, status, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                cv2.imshow(preview, shown)
                if not placed:      # put the preview beside the game so it is not captured
                    cv2.moveWindow(preview, region["left"] + region["width"] + 20, region["top"])
                    placed = True

                # keys
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    break
                elif key == ord("s"):
                    save_frame(frame, detections, sorted(templates))
                elif key == ord("m") and not walker.active:
                    names_ = list(JOBS)
                    ACTIVE_JOB = names_[(names_.index(ACTIVE_JOB) + 1) % len(names_)]
                    print(f"Job: {ACTIVE_JOB}  ({', '.join(JOBS[ACTIVE_JOB])})")
                elif key == ord("p"):
                    capture_player_templates(sct, hwnd)
                    templates = load_templates()
                elif key == ord("t"):
                    crop_template(frame)
                    templates = load_templates()
                elif key == ord("g"):
                    show_grid = not show_grid
                elif key == ord("]"):
                    tile += 1
                elif key == ord("["):
                    tile = max(8, tile - 1)
                elif key == ord("j"):
                    nudge_x -= 1
                elif key == ord("l"):
                    nudge_x += 1
                elif key == ord("i"):
                    nudge_y -= 1
                elif key == ord("k"):
                    nudge_y += 1
                elif key in (ord("+"), ord("=")):
                    THRESHOLD = min(0.99, THRESHOLD + 0.02)
                elif key in (ord("-"), ord("_")):
                    THRESHOLD = max(0.30, THRESHOLD - 0.02)
    finally:
        walker.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
