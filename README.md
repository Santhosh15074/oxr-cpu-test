# OXR-1.0 CPU test scripts

Scripts to run the OXR-1.0 document OCR pipeline on a CPU-only machine (no NVIDIA GPU, no vLLM).

| Script | Purpose |
|---|---|
| `extract_text.py` | Full pipeline: layout service -> crop regions -> OXR-1.0. Writes `IMAGE_ocr.md` and `IMAGE_ocr.json` |
| `oxr_cpu.py` | Loads OXR-1.0 on CPU (with Transformers 5 workarounds) and recognises one image |
| `draw_boxes.py` | Calls the layout service and saves `IMAGE_layout.jpg` with labelled boxes |
| `score.py` | Scores `*_ocr.json` against `ground_truth.json` and validates MRZ check digits |
| `build_report.py` | Builds a self-contained HTML benchmark report |
| `make_page.py` | Generates a synthetic test page (`page.png`) |

## Run (Windows cmd)

1. Start the layout service (keep this window open):
   ```cmd
   cd /d D:\HMIS_Ihl\OXR\services\layout_analyze
   set MODEL_PATH=D:\HMIS_Ihl\OXR\models\OXR-1.0\layout_analysis\last.pth
   set DFINE_DEVICE=cpu
   D:\HMIS_Ihl\OXR\.venv\Scripts\python.exe -m dfine_la
   ```
2. Extract text (second window):
   ```cmd
   cd /d D:\HMIS_Ihl\oxr_test
   set PYTHONIOENCODING=utf-8
   D:\HMIS_Ihl\OXR\.venv\Scripts\python.exe extract_text.py path\to\image.jpg
   ```

Expect about 15-18 s per region and ~8 GB peak RAM on a 4-core laptop CPU.

Test documents and outputs are git-ignored; do not commit personal data.
