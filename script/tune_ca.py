"""Interactive chromatic aberration (purple fringe) tuning tool.

Usage:
    python tune_ca.py <image_path>

Controls:
    S — save current parameters to ca_params.json
    Q — quit
"""

import json
import sys

import argparse

import cv2
import numpy as np

PARAMS_PATH = "ca_params_1.json"

DEFAULT = dict(
    hue_lo=125,
    hue_hi=160,
    sat_thresh=30,
    val_thresh=150,  # only correct pixels darker than this (0-255)
    strength=90,
    darken=30,       # also reduce V by this % when correcting (0=no darkening)
    r_scale=1000,
    b_scale=1000,
)


# ---------------------------------------------------------------------------
# Correction logic
# ---------------------------------------------------------------------------

def process(img, p):
    b, g, r = cv2.split(img.astype(np.float32))
    h, w = g.shape
    cx, cy = w / 2.0, h / 2.0

    xs = np.tile(np.arange(w, dtype=np.float32), (h, 1))
    ys = np.tile(np.arange(h, dtype=np.float32).reshape(-1, 1), (1, w))

    def remap(ch, scale):
        mx = (xs - cx) / scale + cx
        my = (ys - cy) / scale + cy
        return cv2.remap(ch, mx, my, cv2.INTER_LINEAR)

    r_fixed = remap(r, p["r_scale"] / 1000.0)
    b_fixed = remap(b, p["b_scale"] / 1000.0)
    img_ca = cv2.merge([b_fixed, g, r_fixed]).astype(np.uint8)

    img_hsv = cv2.cvtColor(img_ca, cv2.COLOR_BGR2HSV).astype(np.float32)
    hh, s, v = cv2.split(img_hsv)
    mask = ((hh >= p["hue_lo"]) & (hh <= p["hue_hi"]) &
            (s > p["sat_thresh"]) & (v < p["val_thresh"])).astype(np.float32)
    mask = cv2.GaussianBlur(mask, (3, 3), 0)
    alpha = mask * (p["strength"] / 100.0)
    s -= s * alpha
    s  = np.clip(s, 0, 255)
    v -= v * mask * (p["darken"] / 100.0)
    v  = np.clip(v, 0, 255)
    return cv2.cvtColor(cv2.merge([hh, s, v]).astype(np.uint8), cv2.COLOR_HSV2BGR)


# ---------------------------------------------------------------------------
# GUI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    args = parser.parse_args()
    img_path = args.image
    img_orig = cv2.imread(img_path)
    if img_orig is None:
        print(f"Cannot read image: {img_path}")
        sys.exit(1)

    # Downscale for display if very large
    display_h = 600
    scale_disp = display_h / img_orig.shape[0]
    disp_w = int(img_orig.shape[1] * scale_disp)
    disp_h = display_h
    img_small = cv2.resize(img_orig, (disp_w, disp_h))

    params = dict(DEFAULT)
    win = "CA Tuner  |  S=save  Q=quit"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, disp_w * 2, disp_h)

    def nothing(_): pass

    cv2.createTrackbar("Hue Low",    win, params["hue_lo"],      179, nothing)
    cv2.createTrackbar("Hue High",   win, params["hue_hi"],      179, nothing)
    cv2.createTrackbar("Sat Thresh", win, params["sat_thresh"],  255, nothing)
    cv2.createTrackbar("Val Thresh", win, params["val_thresh"],  255, nothing)
    cv2.createTrackbar("Strength",   win, params["strength"],    100, nothing)
    cv2.createTrackbar("Darken",     win, params["darken"],      100, nothing)
    cv2.createTrackbar("R Scale",    win, params["r_scale"],    1010, nothing)
    cv2.createTrackbar("B Scale",    win, params["b_scale"],    1010, nothing)

    cv2.setTrackbarMin("R Scale", win, 990)
    cv2.setTrackbarMin("B Scale", win, 990)

    while True:
        params["hue_lo"]     = cv2.getTrackbarPos("Hue Low",    win)
        params["hue_hi"]     = cv2.getTrackbarPos("Hue High",   win)
        params["sat_thresh"] = cv2.getTrackbarPos("Sat Thresh", win)
        params["val_thresh"] = cv2.getTrackbarPos("Val Thresh", win)
        params["strength"]   = cv2.getTrackbarPos("Strength",   win)
        params["darken"]     = cv2.getTrackbarPos("Darken",     win)
        params["r_scale"]    = cv2.getTrackbarPos("R Scale",    win)
        params["b_scale"]    = cv2.getTrackbarPos("B Scale",    win)

        corrected = process(img_small.copy(), params)
        display = np.concatenate([img_small, corrected], axis=1)

        # Draw divider line
        cv2.line(display, (disp_w, 0), (disp_w, disp_h), (0, 255, 0), 1)
        cv2.putText(display, "Original",  (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(display, "Corrected", (disp_w + 10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow(win, display)
        key = cv2.waitKey(30) & 0xFF

        if key == ord('q') or key == ord('Q'):
            break
        elif key == ord('s') or key == ord('S'):
            with open(PARAMS_PATH, "w") as f:
                json.dump(params, f, indent=2)
            print(f"Parameters saved to {PARAMS_PATH}: {params}")

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
