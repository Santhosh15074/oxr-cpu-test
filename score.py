"""Score OXR output (*_ocr.json) against ground truth and validate MRZ check digits.

Usage: python score.py passport   -> writes passport/results.json
"""

import json
from pathlib import Path
import re
import sys


def norm(text: str) -> str:
    # Case- and whitespace-insensitive: "SCHOOL, JALARAM" == "SCHOOL,JALARAM".
    return re.sub(r"\s+", "", text).upper()


def edit_distance(a: str, b: str, substring: bool = False) -> int:
    """Levenshtein distance; with substring=True, a may match anywhere inside b."""
    prev = [0] * (len(b) + 1) if substring else list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, cb in enumerate(b, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb))
        prev = cur
    return min(prev) if substring else prev[-1]


def best_match(truth: str, texts: list[str], substring: bool | None = None) -> tuple[float, str]:
    """Return (character accuracy 0..1, best OCR text) for a ground-truth string."""
    t = norm(truth)
    best = (0.0, "")
    # A region can hold several lines (e.g. "Type\nP\nCode\nIND", or both MRZ lines),
    # so also try each line on its own; shortest candidate wins ties.
    lines = [line for text in texts for line in text.splitlines()]
    for text in sorted(set(texts + lines), key=len):
        n = norm(text)
        if not n:
            continue
        # Short values ("M", "IND") must match a whole region; longer ones may be
        # part of a region that also contains its label.
        dist = edit_distance(t, n, substring=len(t) >= 6 if substring is None else substring)
        acc = max(0.0, 1 - dist / len(t))
        if acc > best[0]:
            best = (acc, text)
    return best


def mrz_check(data: str) -> str:
    weights, total = (7, 3, 1), 0
    for i, ch in enumerate(data):
        value = int(ch) if ch.isdigit() else (ord(ch) - 55 if ch.isalpha() else 0)
        total += value * weights[i % 3]
    return str(total % 10)


def validate_mrz_line2(line: str) -> list[dict]:
    line = norm(line)
    if len(line) != 44:
        return [{"check": "Line length 44", "ok": False, "detail": f"got {len(line)} characters"}]
    checks = [
        ("Passport number", line[0:9], line[9]),
        ("Date of birth", line[13:19], line[19]),
        ("Date of expiry", line[21:27], line[27]),
        ("Personal number", line[28:42], line[42]),
        ("Composite", line[0:10] + line[13:20] + line[21:43], line[43]),
    ]
    return [
        {"check": name, "ok": mrz_check(data) == digit,
         "detail": f"expected {mrz_check(data)}, read {digit}"}
        for name, data, digit in checks
    ]


def main() -> None:
    folder = Path(sys.argv[1] if len(sys.argv) > 1 else "passport")
    truth = json.loads((folder / "ground_truth.json").read_text(encoding="utf-8"))
    results = {"pages": {}}
    for page, spec in truth.items():
        ocr = json.loads((folder / f"{page}_ocr.json").read_text(encoding="utf-8"))
        texts = [r["text"] for r in ocr["regions"] if r.get("text")]
        fields = []
        for f in spec["fields"]:
            acc, got = best_match(f["text"], texts)
            fields.append({**f, "ocr": got, "accuracy": round(acc, 3), "exact": acc == 1.0})
        labels = []
        for label in spec["labels"]:
            # Labels sit inside bilingual captions ("Hindi / English"), so always
            # search for them as part of a region.
            acc, got = best_match(label, texts, substring=True)
            labels.append({"label": label, "ocr": got, "accuracy": round(acc, 3)})
        page_result = {
            "fields": fields, "labels": labels,
            "layout_seconds": ocr["layout_seconds"], "ocr_seconds": ocr["ocr_seconds"],
            "regions": [{k: r[k] for k in ("type", "score", "seconds", "text")} for r in ocr["regions"]],
        }
        mrz2 = next((f for f in fields if f["field"] == "MRZ line 2"), None)
        if mrz2:
            page_result["mrz_checks"] = validate_mrz_line2(mrz2["ocr"])
        results["pages"][page] = page_result
        print(f"== {page}: {ocr['ocr_seconds']}s OCR, {len(ocr['regions'])} regions")
        for f in fields:
            print(f"  {f['accuracy']*100:5.1f}%  {f['field']:<22} -> {f['ocr'][:60]!r}")
        for c in page_result.get("mrz_checks", []):
            print(f"  MRZ {c['check']:<16} {'OK' if c['ok'] else 'FAIL'} ({c['detail']})")
    (folder / "results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved: {folder / 'results.json'}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
