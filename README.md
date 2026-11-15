# 🩸 HematoVision – Blood Cell Classification

A CNN (TensorFlow/Keras) that classifies white blood cell images into
**eosinophil, lymphocyte, monocyte, neutrophil**, served through a Flask web app.

- CNN classifier with augmentation, batch-norm, dropout, early stopping
- Data preprocessing pipeline, evaluation (classification report + confusion matrix)
- Web UI: drag & drop upload → prediction + per-class probabilities
- Input validation + automated tests for edge cases (corrupt/blank/tiny/huge files, wrong types, NaN outputs, low confidence, missing model)

## Project layout
```
HematoVision/
├── app.py                    # Flask app (factory + routes)
├── hematovision/
│   ├── config.py             # all settings
│   ├── data.py               # dataset loading / preprocessing
│   ├── model.py              # CNN architecture
│   ├── train.py              # training script
│   ├── evaluate.py           # metrics + confusion matrix
│   ├── predict.py            # inference + confidence handling
│   ├── preprocessing.py      # image -> model input
│   └── validation.py         # upload validation
├── templates/ static/        # web UI
├── tests/                    # pytest suite
├── scripts/generate_sample_data.py
├── data/  models/            # dataset in, trained model out
└── .vscode/                  # run/debug configs
```

## Setup (VS Code)
1. Unzip, then **File → Open Folder…** and choose `HematoVision`.
2. Open a terminal (`Ctrl+\``) and create a virtual environment (Python 3.10–3.12 recommended for TensorFlow):
   ```bash
   python -m venv .venv
   # Windows:  .venv\Scripts\activate
   # macOS/Linux: source .venv/bin/activate
   pip install -r requirements.txt
   ```
3. In VS Code press `Ctrl+Shift+P` → **Python: Select Interpreter** → pick `.venv`.

## Get data
**Real data (recommended):** download the Kaggle
[Blood Cell Images](https://www.kaggle.com/datasets/paultimothymooney/blood-cells) dataset
(folder `dataset2-master/images/` has `TRAIN` and `TEST`) and arrange it as:
```
data/train/EOSINOPHIL/*.jpeg      data/test/EOSINOPHIL/*.jpeg
data/train/LYMPHOCYTE/*.jpeg      data/test/LYMPHOCYTE/*.jpeg
data/train/MONOCYTE/*.jpeg        data/test/MONOCYTE/*.jpeg
data/train/NEUTROPHIL/*.jpeg      data/test/NEUTROPHIL/*.jpeg
```
A single `data/<class>/*.jpg` layout also works (auto-split 70/15/15).

**Quick smoke test (synthetic, not real cells):**
```bash
python scripts/generate_sample_data.py --per-class 60
```

## Train & evaluate
```bash
python -m hematovision.train --data-dir data --epochs 25
python -m hematovision.evaluate
```
Outputs: `models/hematovision.keras`, `models/class_names.json`,
`reports/training_history.png`, `reports/confusion_matrix.png`, `reports/classification_report.json`.

(Or use the **Run and Debug** panel → *Train model* / *Evaluate model*.)

## Run the web app
```bash
python app.py
```
Open <http://127.0.0.1:5000>. If no model exists yet, the UI shows a friendly
"train first" message (HTTP 503).

API: `POST /api/predict` with multipart field `file` →
```json
{"label": "neutrophil", "confidence": 0.93, "uncertain": false,
 "probabilities": {"eosinophil": 0.02, "lymphocyte": 0.02, "monocyte": 0.03, "neutrophil": 0.93}}
```

## Tests
```bash
pytest
```
Tests use a fake predictor, so they run quickly and don't need a trained model.

## Edge-case handling
| Case | Behaviour |
|---|---|
| Wrong extension / not really an image / truncated file | 400 with clear message |
| Empty, >5 MB, <32 px, >8000 px, decompression bomb | 400 / 413 |
| Blank or uniform image | 400 |
| Grayscale / RGBA / palette images | converted to RGB |
| Low confidence (< 0.60, see `config.py`) | returned with `uncertain: true` and a UI warning |
| NaN/inf or wrong-sized model output | caught; generic 500, details only in server log |
| Model file missing | 503 |

## Notes
Educational project – **not a medical device**; do not use for diagnosis.
