"""Build a self-contained HTML benchmark report from passport/results.json.

Usage: python build_report.py passport  -> writes passport/oxr_passport_benchmark.html
"""

import base64
from datetime import date
import html
import json
from pathlib import Path
import sys

folder = Path(sys.argv[1] if len(sys.argv) > 1 else "passport")
results = json.loads((folder / "results.json").read_text(encoding="utf-8"))
env = json.loads((folder / "environment.json").read_text(encoding="utf-8"))
pages = results["pages"]
e = html.escape


def pct(v: float) -> str:
    return f"{v * 100:.0f}%"


def avg(values) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def tone(v: float) -> str:
    return "good" if v >= 0.95 else "warn" if v >= 0.7 else "bad"


all_fields = [f for p in pages.values() for f in p["fields"]]
all_labels = [l for p in pages.values() for l in p["labels"]]
by_cat = {}
for f in all_fields:
    by_cat.setdefault(f["category"], []).append(f)
mrz_checks = pages["front"].get("mrz_checks", [])
mrz_ok = sum(c["ok"] for c in mrz_checks)
pictures = {name: sum(r["type"] == "Picture" for r in p["regions"]) for name, p in pages.items()}
located = sum(f["accuracy"] >= 0.5 for f in all_fields)
exact = sum(f["exact"] for f in all_fields)
page_secs = {name: p["layout_seconds"] + p["ocr_seconds"] for name, p in pages.items()}
load_s = env["model_load_seconds"]


def cat_row(cat):
    fs = by_cat.get(cat, [])
    return avg(f["accuracy"] for f in fs), sum(f["exact"] for f in fs), len(fs)


names_acc, names_ex, names_n = cat_row("Names & places")
num_acc, num_ex, num_n = cat_row("Numbers & dates")
addr_acc, addr_ex, addr_n = cat_row("Address")
mrz_acc, mrz_ex, mrz_n = cat_row("MRZ")
label_acc = avg(l["accuracy"] for l in all_labels)
region_secs = [r["seconds"] for p in pages.values() for r in p["regions"] if r["seconds"]]

laptop = "Laptop CPU (i5-1334U, 16 GB RAM)"
features = [
    # feature, benchmark text, measured, achievable, time, notes, infra
    ("Find every text block on the page (layout detection)",
     f"D-FINE layout model located {located}/{len(all_fields)} expected fields",
     located / len(all_fields), "95–100%",
     f"{avg(p['layout_seconds'] for p in pages.values()):.1f}s per page",
     "Small 75 MB model; fast even on CPU. Lower confidence (0.3) needed for faint print.",
     "CPU only; ~0.5 GB RAM"),
    ("Read names & places (Latin capitals)",
     f"{names_ex}/{names_n} fields exact", names_acc, "98–100%",
     f"{avg(region_secs):.0f}s per field (avg)",
     "Large printed capitals are OXR's strong point.", laptop),
    ("Read numbers & dates (passport no., DOB, expiry, file no.)",
     f"{num_ex}/{num_n} fields exact", num_acc, "98–100%",
     f"{avg(region_secs):.0f}s per field (avg)",
     "Single-character fields (Type 'P', Sex 'M') depend on layout finding a tiny box.", laptop),
    ("Read address lines",
     f"{addr_ex}/{addr_n} lines exact", addr_acc, "98–100%",
     f"{avg(region_secs):.0f}s per line (avg)",
     "Scored ignoring spaces; OXR sometimes adds a space after commas.", laptop),
    ("Read MRZ (2 machine-readable lines, 44 chars each)",
     f"{mrz_ex}/{mrz_n} lines exact; check digits {mrz_ok}/{len(mrz_checks)} pass",
     mrz_acc, "99–100% with check-digit validation",
     "Part of front page time",
     "MRZ check digits let you auto-detect OCR errors in passport no., DOB and expiry.", laptop),
    ("Read small printed field labels (English part)",
     f"Average character accuracy across {len(all_labels)} labels", label_acc, "60–85% (photo quality bound)",
     "Included above",
     "Approximate score (fuzzy match can pair a label with a similar caption). Tiny, blurred bilingual captions; usually not needed, since field position or MRZ identifies the field.", laptop),
    ("Read Hindi (Devanagari) labels",
     "Not scored (no verified Hindi ground truth)", None, "Low on this photo quality",
     "—", "Output for Hindi captions was garbled; needs a sharper scan to evaluate fairly.", laptop),
    ("Detect photo / signature / barcode regions",
     f"Picture regions found: front {pictures.get('front', 0)}, back {pictures.get('back', 0)}",
     None, "90–100%", "Included in layout time",
     "Pictures are cropped, not read. Barcode content is not decoded (needs a barcode library).",
     "CPU only"),
    ("Return structured fields (JSON: surname, DOB, …)",
     "Not a native OXR feature: OXR returns text blocks, not key-value pairs", 0.0,
     "~100% for MRZ fields via rule-based parsing",
     "< 1 ms (rules)", "Parse the MRZ with fixed positions; map other fields by layout order.",
     "Plain Python, no AI needed"),
]

infra = [
    ("This laptop (measured)", "Intel Core i5-1334U, 10 threads, Intel UHD (unused)",
     "16 GB", f"~{env['peak_ram_gb']:.1f} GB peak", "PyTorch CPU FP32 + Transformers",
     f"{page_secs['front'] / 60:.1f} min (front) · {page_secs['back'] / 60:.1f} min (back)",
     "Works. Fine for testing and low volume; too slow for live use."),
    ("Better CPU desktop / server (estimate)", "8+ performance cores, fast DDR5",
     "16–32 GB", "~8 GB", "Same as above",
     "~1–2 min per page (estimate)", "Moderate gain; still CPU-bound."),
    ("Official setup (vendor guidance, not measured)", "NVIDIA GPU, ≥24 GB VRAM",
     "32 GB+", "GPU", "vLLM + OXR plugin, Docker", "Seconds per page (not measured here)",
     "Recommended for production or batch processing."),
]

overall = avg(f["accuracy"] for f in all_fields)
total_min = sum(page_secs.values()) / 60


def img(path: Path) -> str:
    if not path.exists():
        return ""
    return "data:image/jpeg;base64," + base64.b64encode(path.read_bytes()).decode()


def bar(v):
    if v is None:
        return '<span class="na">n/a</span>'
    return (f'<div class="meter {tone(v)}"><span style="width:{v * 100:.0f}%"></span></div>'
            f'<b class="{tone(v)}">{pct(v)}</b>')


feature_rows = "\n".join(
    f"<tr><td><b>{e(f)}</b></td><td>{e(b)}</td><td class='q'>{bar(m)}</td>"
    f"<td>{e(a)}</td><td>{e(t)}</td><td class='muted'>{e(n)}</td><td>{e(i)}</td></tr>"
    for f, b, m, a, t, n, i in features
)

field_rows = []
for name, p in pages.items():
    for f in p["fields"]:
        status = "✅ exact" if f["exact"] else ("⚠️ close" if f["accuracy"] >= 0.8 else "❌ wrong")
        field_rows.append(
            f"<tr><td>{e(name.title())}</td><td>{e(f['field'])}</td><td><code>{e(f['text'])}</code></td>"
            f"<td><code>{e(f['ocr']) or '<i>not found</i>'}</code></td><td>{bar(f['accuracy'])}</td><td>{status}</td></tr>")
    for l in p["labels"]:
        field_rows.append(
            f"<tr class='lbl'><td>{e(name.title())}</td><td>Label</td><td><code>{e(l['label'])}</code></td>"
            f"<td><code>{e(l['ocr'])}</code></td><td>{bar(l['accuracy'])}</td><td class='muted'>label</td></tr>")

mrz_rows = "".join(
    f"<tr><td>{e(c['check'])}</td><td>{'✅ pass' if c['ok'] else '❌ fail'}</td><td class='muted'>{e(c['detail'])}</td></tr>"
    for c in mrz_checks)

timing_rows = []
for name, p in pages.items():
    mx = max([r["seconds"] for r in p["regions"]] + [1])
    for r in p["regions"]:
        w = r["seconds"] / 120 * 100
        timing_rows.append(
            f"<tr><td>{e(name.title())}</td><td>{e(r['type'])}</td><td>{r['score']:.2f}</td>"
            f"<td><div class='tbar'><span style='width:{min(w, 100):.1f}%'></span></div></td>"
            f"<td>{r['seconds']:.1f}s</td><td class='muted'>{e((r['text'] or '—')[:70])}</td></tr>")

infra_rows = "\n".join(
    f"<tr><td><b>{e(a)}</b></td><td>{e(b)}</td><td>{e(c)}</td><td>{e(d)}</td><td>{e(f)}</td><td>{e(g)}</td><td class='muted'>{e(h)}</td></tr>"
    for a, b, c, d, f, g, h in infra)

page_html = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>OXR-1.0 Passport OCR Benchmark</title>
<style>
:root {{ --bg:#f7f8fa; --card:#fff; --ink:#1d2433; --muted:#667085; --line:#e4e7ec;
  --good:#12805c; --warn:#b54708; --bad:#c01048; --accent:#3e5bd8; }}
@media (prefers-color-scheme: dark) {{ :root {{ --bg:#0f1218; --card:#171b24; --ink:#e6e9ef; --muted:#98a2b3;
  --line:#2a303c; --good:#3ccb8f; --warn:#f5a524; --bad:#f2557f; --accent:#7c93ff; }} }}
* {{ box-sizing:border-box }}
body {{ margin:0; font:14px/1.5 system-ui,-apple-system,Segoe UI,sans-serif; background:var(--bg); color:var(--ink) }}
main {{ max-width:1280px; margin:auto; padding:28px 20px 60px }}
h1 {{ margin:0 0 4px; font-size:26px }} h2 {{ margin:34px 0 10px; font-size:19px }}
.sub {{ color:var(--muted) }}
.kpis {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(180px,1fr)); gap:12px; margin:20px 0 }}
.kpi {{ background:var(--card); border:1px solid var(--line); border-radius:12px; padding:14px 16px }}
.kpi .v {{ font-size:28px; font-weight:700 }} .kpi .l {{ color:var(--muted); font-size:12.5px }}
.card {{ background:var(--card); border:1px solid var(--line); border-radius:12px; overflow:auto }}
table {{ border-collapse:collapse; width:100% }}
th,td {{ text-align:left; padding:9px 12px; border-bottom:1px solid var(--line); vertical-align:top }}
th {{ font-size:12px; text-transform:uppercase; letter-spacing:.03em; color:var(--muted); background:var(--card); position:sticky; top:0 }}
tr:last-child td {{ border-bottom:none }}
td.q {{ min-width:150px }} .muted {{ color:var(--muted) }} .na {{ color:var(--muted) }}
.meter {{ display:inline-block; width:80px; height:8px; border-radius:4px; background:var(--line); vertical-align:middle; margin-right:8px; overflow:hidden }}
.meter span {{ display:block; height:100% }}
.meter.good span {{ background:var(--good) }} .meter.warn span {{ background:var(--warn) }} .meter.bad span {{ background:var(--bad) }}
b.good {{ color:var(--good) }} b.warn {{ color:var(--warn) }} b.bad {{ color:var(--bad) }}
.tbar {{ width:160px; height:8px; border-radius:4px; background:var(--line); overflow:hidden }}
.tbar span {{ display:block; height:100%; background:var(--accent) }}
code {{ font:12.5px ui-monospace,Consolas,monospace; word-break:break-all }}
tr.lbl td {{ opacity:.8 }}
.imgs {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(380px,1fr)); gap:14px }}
.imgs figure {{ margin:0; background:var(--card); border:1px solid var(--line); border-radius:12px; padding:10px }}
.imgs img {{ width:100%; border-radius:8px }} figcaption {{ color:var(--muted); font-size:12.5px; margin-top:6px }}
.note {{ background:var(--card); border:1px solid var(--line); border-left:4px solid var(--accent); border-radius:8px; padding:12px 16px; margin-top:12px }}
ul {{ margin:6px 0; padding-left:20px }}
</style></head><body><main>
<h1>OXR-1.0 benchmark — Indian passport (front + back)</h1>
<div class="sub">Real photos · local CPU inference without GPU · generated {date.today():%d %b %Y} ·
OXR-1.0 (SigLIP2 + Qwen3-0.6B, FP32) + D-FINE layout model</div>

<div class="kpis">
 <div class="kpi"><div class="v {tone(overall)}">{overall * 100:.1f}%</div><div class="l">Average character accuracy, {len(all_fields)} data fields</div></div>
 <div class="kpi"><div class="v">{exact}/{len(all_fields)}</div><div class="l">Fields read exactly right</div></div>
 <div class="kpi"><div class="v">{mrz_ok}/{len(mrz_checks)}</div><div class="l">MRZ check digits valid</div></div>
 <div class="kpi"><div class="v">{total_min:.1f} min</div><div class="l">Total for both pages (+{load_s:.0f}s model load)</div></div>
 <div class="kpi"><div class="v">~{env['peak_ram_gb']:.1f} GB</div><div class="l">Peak RAM used (of 16 GB)</div></div>
</div>

<h2>1 · Feature benchmark for passport data extraction</h2>
<div class="card"><table>
<tr><th>Specific feature required from AI</th><th>Model benchmark (measured)</th><th>Quality achieved here</th>
<th>Quality achievable</th><th>Time</th><th>Notes</th><th>Infrastructure for local run</th></tr>
{feature_rows}
</table></div>
<div class="note"><b>How to read this.</b> “Quality achieved” is measured on these two photos. “Quality achievable” is an
<i>estimate</i> of what this model can reach on this task with a good scan and simple post-processing (e.g. MRZ
check-digit validation); it is not measured.</div>

<h2>2 · Field-by-field results</h2>
<div class="card"><table>
<tr><th>Page</th><th>Field</th><th>Expected</th><th>OXR output (best matching region)</th><th>Char. accuracy</th><th>Result</th></tr>
{''.join(field_rows)}
</table></div>

<h2>3 · MRZ check-digit validation (front page, line 2)</h2>
<div class="card"><table><tr><th>Check</th><th>Result</th><th>Detail</th></tr>{mrz_rows}</table></div>
<div class="note">Each MRZ section has a check digit (ICAO 9303). When all checks pass, the passport number,
date of birth and expiry read by OCR are almost certainly correct, so this is the most reliable way to validate output automatically.</div>

<h2>4 · Time per region (CPU)</h2>
<div class="card"><table>
<tr><th>Page</th><th>Region type</th><th>Layout conf.</th><th>Time (bar = 120 s)</th><th>Seconds</th><th>Text read</th></tr>
{''.join(timing_rows)}
</table></div>
<div class="note"><b>Speed-up available:</b> the front page produced {len(pages["front"]["regions"])} regions, several overlapping (e.g. a date box and a date+sex box), so some text was read twice. Removing overlapping boxes before OCR would cut time noticeably.<br>Per region: ~10 s to encode the image crop + ~0.2–0.3 s per generated token. Model load: {load_s:.0f} s (once per run).
Layout detection: {pages['front']['layout_seconds']:.1f} s (front), {pages['back']['layout_seconds']:.1f} s (back).</div>

<h2>5 · Infrastructure required for local run</h2>
<div class="card"><table>
<tr><th>Setup</th><th>Processor</th><th>System RAM</th><th>Model memory</th><th>Software</th><th>Time per page</th><th>Verdict</th></tr>
{infra_rows}
</table></div>
<div class="note"><b>Software used:</b> Python {e(env['python'])}, PyTorch {e(env['torch'])}, Transformers {e(env['transformers'])},
Windows 11. Disk: ~3.1 GB for model files. No internet needed after download.</div>

<h2>6 · Layout detection on the real photos</h2>
<div class="imgs">
 <figure><img src="{img(folder / 'front_layout.jpg')}" alt="front page layout"><figcaption>Front page — boxes from the layout model (confidence ≥ 0.3)</figcaption></figure>
 <figure><img src="{img(folder / 'back_layout.jpg')}" alt="back page layout"><figcaption>Back page — boxes from the layout model (confidence ≥ 0.3)</figcaption></figure>
</div>

<h2>7 · Method &amp; limits</h2>
<div class="note"><ul>
<li>Pipeline: D-FINE layout service → crop each region → OXR-1.0 with the official per-type prompt; pictures are not read.</li>
<li>Ground truth was transcribed from the photos and confirmed by the passport's own MRZ check digits.</li>
<li>Character accuracy = 1 − edit distance ÷ length, ignoring case and spaces; each field is matched to the closest OXR region.</li>
<li>Only 2 images: results show direction, not statistical certainty. Test 20–50 documents before deciding.</li>
<li>⚠️ This file contains personal passport data. Keep it local; do not share or upload it.</li>
</ul></div>
</main></body></html>"""

out = folder / "oxr_passport_benchmark.html"
out.write_text(page_html, encoding="utf-8")
print(f"Saved: {out}")
