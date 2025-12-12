# sdf-adversarial-gmm

Clean, reproducible code structure for **Soft-penalty Neural SDF** vs **Adversarial GMM SDF**.

## Folder structure
- `src/` : reusable Python modules (models, training, preprocessing)
- `notebooks/` : your main run notebook (optional)
- `data/` : (ignored) put your parquet here locally
- `results/` : (ignored) outputs, logs, figures

## Quick start (local)
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Put your parquet at:
- `data/OpenAP_Macro.parquet.gzip`

Then run:
```bash
python scripts/run_empirical.py
```

## Notes
- This repo intentionally avoids absolute paths like `/Users/...`.
- Large data files are excluded by `.gitignore`.
