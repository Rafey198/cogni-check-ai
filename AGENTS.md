# cogni-check-ai

This repository hosts **NeuroSense AI**, a single Streamlit application located in
the `neurosense-ai/` subdirectory.

## Cursor Cloud specific instructions

### Project layout
- The application lives in `neurosense-ai/`, not the repo root. All commands below
  assume you are inside `neurosense-ai/`.
- It is a single Python/Streamlit service (no database, no external services, no
  secrets/env vars required). The bundled synthetic dataset
  (`data/sample_cognitive_data.csv`) and a pre-trained model
  (`models/trained_model.pkl`) ship in the repo, so every page works immediately.

### Python environment
- A virtualenv is used at `neurosense-ai/.venv` (Python 3.12). The update script
  creates it and installs `requirements.txt`. The system package `python3-venv`
  (e.g. `python3.12-venv`) is required to create the venv and is part of the VM
  snapshot; it is intentionally NOT in the update script.
- Run tools via the venv interpreter, e.g. `neurosense-ai/.venv/bin/python` and
  `neurosense-ai/.venv/bin/streamlit`.

### Run / lint / test
- Run (dev): from `neurosense-ai/`, `.venv/bin/streamlit run app.py` (serves on
  port 8501). In a headless/cloud VM, add
  `--server.address 0.0.0.0 --server.headless true`.
- Lint: the repo defines no linter/formatter config. Use
  `.venv/bin/python -m compileall app.py src` as a basic syntax check.
- Test: there is no test suite. Validate end-to-end by exercising the core
  pipeline (load data -> `split_data` -> `train_models` -> `select_best_model` ->
  `save_model`/`load_model` -> `predict_risk`) or by clicking through the UI
  (Train Model -> Risk Prediction).

### Gotchas
- `app.py` inserts its own directory onto `sys.path` so `import src...` works
  regardless of CWD, but Streamlit must still be launched from `neurosense-ai/`
  so relative paths to `data/` and `models/` resolve.
- The "Train Model" page overwrites `models/trained_model.pkl` on disk. That file
  is tracked in git, so retraining will show up as a local diff — do not commit it
  unless intentionally updating the bundled model.
