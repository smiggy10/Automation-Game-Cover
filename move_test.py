"""
Movement calibration.
Presses W/A/S/D for different durations, compares screenshots before/after,
and reports how far the character moved (in pixels and tiles).

Setup:  pip install pydirectinput
Run:    python move_test.py
Stand in an open area with ~6 free tiles in every direction before starting.
Press Ctrl+C in the console to abort at any time.
"""
import ctypes
import json
import time

import cv2
import mss
import numpy as np
import pydirectinput

from main import WINDOW_TITLE, TILE_SIZE, client_region, find_window

pydirectinput.PAUSE = 0

DURATIONS = [0.05, 0.10, 0.20, 0.50, 1.00]          # seconds to hold each key
PAIRS = [("d", "a"), ("s", "w")]                   # each key is followed by its reverse
HORIZONTAL = ("a", "d")                            # keys that move left/right
SETTLE = 0.6                                       # wait for walk animation to finish


def grab_gray(sct, hwnd):
    frame = np.array(sct.grab(client_region(hwnd)))[:, :, :3]
    return cv2.cvtColor(np.ascontiguousarray(frame), cv2.COLOR_BGR2GRAY)


def measure_shift(a, b):
    """How far the player moved (dx, dy in px) between gray frames a and b.
    Tracks a patch of scenery above the player, since the player and the HUD
    do not scroll with the world."""
    H, W = a.shape
    x0, y0 = int(W * 0.35), int(H * 0.12)
    patch = a[y0:int(H * 0.32), x0:int(W * 0.65)]
    res = cv2.matchTemplate(b, patch, cv2.TM_CCOEFF_NORMED)
    _, score, _, loc = cv2.minMaxLoc(res)
    world_dx, world_dy = loc[0] - x0, loc[1] - y0
    return -world_dx, -world_dy, score          # player moves opposite to the world


def tap(key, seconds):
    pydirectinput.keyDown(key)
    time.sleep(seconds)
    pydirectinput.keyUp(key)


def main():
    hwnd, title = find_window(WINDOW_TITLE)
    if not hwnd:
        print(f'Window "{WINDOW_TITLE}" not found. Set WINDOW_TITLE in main.py.')
        return
    ctypes.windll.user32.SetForegroundWindow(hwnd)
    for n in (3, 2, 1):
        print(f"Starting in {n}... (make sure the game window is focused)")
        time.sleep(1)

    results = []
    with mss.mss() as sct:
        for fwd, back in PAIRS:
            for dur in DURATIONS:
                for key in (fwd, back):
                    before = grab_gray(sct, hwnd)
                    tap(key, dur)
                    time.sleep(SETTLE)
                    after = grab_gray(sct, hwnd)
                    dx, dy, score = measure_shift(before, after)
                    px = dx if key in HORIZONTAL else dy
                    results.append({"key": key, "hold_s": dur, "px": int(px), "score": round(score, 2)})
                    flag = "" if score > 0.6 else "   <-- unreliable (low match)"
                    print(f"{key:>5}  hold {dur:4.2f}s  moved {px:+4d}px = {px / TILE_SIZE:+5.2f} tiles"
                          f"  (match {score:.2f}){flag}")

    # speed estimate from the longest holds
    longest = [r for r in results if r["hold_s"] == DURATIONS[-1] and r["score"] > 0.6]
    if longest:
        speed = np.mean([abs(r["px"]) / r["hold_s"] for r in longest])
        print(f"\nApprox. walking speed: {speed:.0f} px/s = {speed / TILE_SIZE:.2f} tiles/s")
    else:
        speed = None
    with open("calibration.json", "w") as f:
        json.dump({"tile_size": TILE_SIZE, "speed_px_per_s": speed, "samples": results}, f, indent=2)
    print("Saved calibration.json")


if __name__ == "__main__":
    main()
