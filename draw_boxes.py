"""Send an image to the local layout service and save a copy with boxes drawn on it.

Usage: python draw_boxes.py IMAGE [--confidence 0.3] [--url http://127.0.0.1:8000/inference]
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import requests

LABELS = [
    "Caption", "Footnote", "Formula", "Text", "Page-footer", "Page-header",
    "Picture", "Title", "Table", "Code Block", "Stamp", "Chemical Structure",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("image", type=Path)
    parser.add_argument("--confidence", type=float, default=0.3)
    parser.add_argument("--url", default="http://127.0.0.1:8000/inference")
    args = parser.parse_args()

    raw = args.image.read_bytes()
    response = requests.post(
        args.url,
        files=[("images", (args.image.name, raw, "application/octet-stream"))],
        data={"confidence": str(args.confidence)},
        timeout=300,
    )
    response.raise_for_status()
    data = response.json()["data"]
    labels, boxes, scores = data["labels"][0], data["boxes"][0], data["scores"][0]

    img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    print(f"Found {len(labels)} regions:")
    for label, box, score in zip(labels, boxes, scores):
        name = LABELS[label]
        x1, y1, x2, y2 = (int(v) for v in box)
        print(f"  {name:<12} score={score:.2f}  box=({x1},{y1})-({x2},{y2})")
        color = tuple(int(c) for c in np.random.default_rng(label).integers(0, 200, 3))
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img, f"{name} {score:.2f}", (x1, max(y1 - 6, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    out = args.image.with_name(args.image.stem + "_layout.jpg")
    cv2.imwrite(str(out), img)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
