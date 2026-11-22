#!/usr/bin/env python3
"""
One-off generator for the thin orange silhouette outline that traces a cutout
photo in index.html (the "hero-outline" / "about-outline" <svg><path>). Used
for both portrait cutouts on the page: assets/hero-1-clean.png (hero) and
assets/hero2-clean.png (about).

Re-run this whenever one of those source photos changes:

    /tmp/outline-venv/bin/python3 scripts/generate-hero-outline.py assets/hero-1-clean.png
    /tmp/outline-venv/bin/python3 scripts/generate-hero-outline.py assets/hero2-clean.png

(or any Python env with opencv-python-headless + pillow + numpy installed).
It prints a <path d="..."> you paste into the matching inline <svg> in
index.html, replacing the existing `d` attribute. The viewBox must stay
"0 0 <img width> <img height>" (printed below) so the overlay lines up with
the <img> at any render size.
"""
import sys
import cv2
import numpy as np
from PIL import Image

SRC = sys.argv[1] if len(sys.argv) > 1 else "assets/hero-1-clean.png"
DILATE_PX = 8  # how far the line floats off the subject's edge
SIMPLIFY_EPS = 1.2  # approxPolyDP tolerance in source-image pixels

im = Image.open(SRC).convert("RGBA")
w, h = im.size
alpha = np.array(im)[:, :, 3]

# Binary mask of the subject, then dilate so the traced line floats
# slightly outside the real edge instead of hugging it.
_, mask = cv2.threshold(alpha, 16, 255, cv2.THRESH_BINARY)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (DILATE_PX * 2 + 1,) * 2)
dilated = cv2.dilate(mask, kernel)

contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
largest = max(contours, key=cv2.contourArea)

simplified = cv2.approxPolyDP(largest, SIMPLIFY_EPS, closed=True)
pts = simplified.reshape(-1, 2)

d = "M " + " L ".join(f"{x},{y}" for x, y in pts) + " Z"

print(f"viewBox: 0 0 {w} {h}")
print(f"points: {len(pts)}")
print()
print(f'<path d="{d}" />')
