"""End-to-end OXR text extraction on CPU: layout service -> crop regions -> OXR-1.0.

Usage: python extract_text.py IMAGE [--confidence 0.5] [--layout-url http://127.0.0.1:8000/inference]

Writes IMAGE_ocr.md (text in reading order) and IMAGE_ocr.json (boxes, text, timings).
"""

import argparse
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np
from PIL import Image
import requests
import torch

import oxr_cpu

LABELS = [
    "Caption", "Footnote", "Formula", "Text", "Page-footer", "Page-header",
    "Picture", "Title", "Table", "Code Block", "Stamp", "Chemical Structure",
]
# Same category -> prompt mapping as oxr/pipeline/recognition.py.
PROMPTS = {
    "Table": "Recognize the table of the image into html format.",
    "Formula": "Recognize the formula of the image into KaTex format.",
    "Code Block": "Recognize the code of the image into markdown code format.",
    "Chemical Structure": "Recognize the chemical structure of the image into SMILES.",
}
TEXT_PROMPT = "Recognize the content of the image."


def detect_layout(raw: bytes, name: str, url: str, confidence: float) -> list[dict]:
    response = requests.post(
        url,
        files=[("images", (name, raw, "application/octet-stream"))],
        data={"confidence": str(confidence)},
        timeout=300,
    )
    response.raise_for_status()
    data = response.json()["data"]
    # The layout model can return one box under two classes; keep the higher score.
    best: dict[tuple, dict] = {}
    for label, box, score in zip(data["labels"][0], data["boxes"][0], data["scores"][0]):
        key = tuple(round(v) for v in box)
        if key not in best or score > best[key]["score"]:
            best[key] = {"type": LABELS[label], "bbox": [round(v) for v in box], "score": round(score, 3)}
    # Simple reading order: top-to-bottom, then left-to-right.
    return sorted(best.values(), key=lambda r: (r["bbox"][1] // 15, r["bbox"][0]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("image", type=Path)
    parser.add_argument("--confidence", type=float, default=0.5)
    parser.add_argument("--layout-url", default="http://127.0.0.1:8000/inference")
    args = parser.parse_args()
    # Windows consoles default to cp1252, which can't print Hindi or other scripts.
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    raw = args.image.read_bytes()
    page = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)

    started = time.perf_counter()
    regions = detect_layout(raw, args.image.name, args.layout_url, args.confidence)
    layout_s = time.perf_counter() - started
    print(f"Layout: {len(regions)} regions in {layout_s:.1f}s")

    started = time.perf_counter()
    model = oxr_cpu.load_model()
    print(f"OXR model loaded in {time.perf_counter() - started:.1f}s\n")

    total_started = time.perf_counter()
    for i, region in enumerate(regions, 1):
        if region["type"] == "Picture":
            region["text"], region["seconds"] = "", 0.0
            print(f"[{i}/{len(regions)}] Picture  (skipped, no OCR for pictures)")
            continue
        x1, y1, x2, y2 = region["bbox"]
        h, w = page.shape[:2]
        crop = page[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
        image = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        prompt = PROMPTS.get(region["type"], TEXT_PROMPT)
        started = time.perf_counter()
        text, tokens = oxr_cpu.recognize(model, image, prompt)
        region["text"], region["seconds"] = text, round(time.perf_counter() - started, 1)
        print(f"[{i}/{len(regions)}] {region['type']:<11} {region['seconds']:>5.1f}s  {text}")
    ocr_s = time.perf_counter() - total_started

    md = "\n\n".join(r["text"] for r in regions if r["text"])
    out_md = args.image.with_name(args.image.stem + "_ocr.md")
    out_json = args.image.with_name(args.image.stem + "_ocr.json")
    out_md.write_text(md + "\n", encoding="utf-8")
    out_json.write_text(json.dumps({
        "image": str(args.image), "layout_seconds": round(layout_s, 1),
        "ocr_seconds": round(ocr_s, 1), "threads": torch.get_num_threads(), "regions": regions,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nRecognition took {ocr_s:.0f}s for {len(regions)} regions "
          f"({ocr_s / max(1, len(regions)):.1f}s per region)")
    print(f"Saved: {out_md}\nSaved: {out_json}")


if __name__ == "__main__":
    main()
