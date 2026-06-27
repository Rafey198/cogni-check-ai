# 🧠 NeuroSense AI

### A Multimodal Digital Biomarker System for Early Cognitive Risk Screening

NeuroSense AI is a **research/academic demonstration** built with
[Streamlit](https://streamlit.io/). It showcases a complete, transparent machine
learning workflow — from a multimodal dataset, through preprocessing and model
training, to prediction, explainability and report generation — for the task of
**early cognitive risk screening**.

> ⚠️ **Disclaimer:** This tool is for academic/research demonstration only and is
> **not** a medical diagnosis or clinical decision system. The bundled dataset is
> fully synthetic and contains no real patient data.

---

## ✨ Features

The app provides seven pages in the sidebar:

1. **Home** – project overview and disclaimer.
2. **Dataset** – explore the bundled synthetic dataset or upload your own CSV.
3. **Train Model** – train & compare Logistic Regression and Random Forest,
   view accuracy / precision / recall / F1, confusion matrix and classification
   report, and save the best model.
4. **Risk Prediction** – interactive form to predict **Low / Moderate / High**
   cognitive risk with a confidence score and a careful, non-medical interpretation.
5. **Explainability** – Random Forest feature importance with a bar chart and a
   plain-language summary.
6. **Research Report** – generate and download a concise text report.
7. **About** – background, extensions and technology.

### Multimodal feature set

| Modality        | Features                                                        |
| --------------- | --------------------------------------------------------------- |
| Demographic     | `age`, `gender`, `education_years`                              |
| Cognitive       | `memory_score`, `attention_score`                              |
| Behavioral      | `reaction_time`, `sleep_hours`, `activity_level`               |
| Sensor / Gait   | `walking_stability`                                            |
| Speech/Language | `speech_pause_rate`, `lexical_diversity`                       |
| Clinical-style  | `mmse_score`                                                   |
| **Target**      | `cognitive_risk_label` → `0 = Low`, `1 = Moderate`, `2 = High` |

---

## 📁 Project structure

```
neurosense-ai/
├── app.py                       # Streamlit application (7 pages)
├── requirements.txt             # Python dependencies
├── README.md                    # This file
├── .gitignore
├── data/
│   └── sample_cognitive_data.csv  # Bundled synthetic dataset
├── src/
│   ├── __init__.py
│   ├── data_loader.py           # Load / generate dataset
│   ├── preprocessing.py         # Imputation, encoding, scaling, split
│   ├── model.py                 # Train / evaluate / save / predict
│   ├── explainability.py        # Random Forest feature importance
│   └── report.py                # Text report generation
├── models/
│   └── trained_model.pkl        # Persisted best model (regenerated on demand)
└── assets/
```

---

## 🚀 Local setup & run

> Requires **Python 3.9+**.

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>/neurosense-ai

# 2. (Recommended) create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate        # On Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the app
streamlit run app.py
```

The app opens at `http://localhost:8501`. The bundled dataset and a pre-trained
model are included, so all pages work immediately. You can also retrain models
from the **Train Model** page at any time.

---

## ☁️ Live Deployment on Streamlit Community Cloud

This project is ready for free hosting on
[Streamlit Community Cloud](https://streamlit.io/cloud). All file paths are
**relative** and deployment-safe (no absolute Windows/Mac paths), and the sample
dataset is bundled so the app works without any external downloads.

### Deployment steps

1. **Push this project to a public GitHub repository.**
   - Create a public repo on GitHub.
   - Commit and push the project (see the exact commands in the next section).
2. **Open Streamlit Community Cloud** at
   [share.streamlit.io](https://share.streamlit.io/).
3. **Sign in with GitHub** and authorize Streamlit to access your repositories.
4. Click **"Create app" / "New app"**, then **select the repository, branch, and
   the `app.py` file**:
   - **Repository:** `your-username/your-repo`
   - **Branch:** `main`
   - **Main file path:** `neurosense-ai/app.py`
     *(If you place the project files at the repository root instead, use
     `app.py`.)*
5. **Click "Deploy".** Streamlit installs the packages from
   `requirements.txt` and launches the app at a public `*.streamlit.app` URL.

> **Tip:** The app retrains and recreates `models/trained_model.pkl` on demand
> via the **Train Model** page, so even if the cloud environment resets the
> file system, you can regenerate the model with one click.

---

## 🐙 Exact GitHub upload steps

From the repository root (the folder containing `neurosense-ai/`):

```bash
# Initialize git (skip if the folder is already a repo)
git init
git add .
git commit -m "Add NeuroSense AI Streamlit research demo"

# Point to your new GitHub repository (replace the URL)
git remote add origin https://github.com/<your-username>/<your-repo>.git
git branch -M main

# Push
git push -u origin main
```

If the repository already exists locally with a remote configured, simply:

```bash
git add .
git commit -m "Update NeuroSense AI"
git push
```

---

## 🧪 Dataset explanation

The bundled file `data/sample_cognitive_data.csv` is a **fully synthetic**
dataset of 600 simulated participants. A latent "risk" signal is constructed as
a weighted combination of features (e.g. lower memory/MMSE scores, slower
reaction times and higher speech-pause rates trend toward higher risk), with
added noise, then discretised into three balanced-ish risk classes.

A small number of missing values are intentionally injected so the preprocessing
pipeline (median/most-frequent imputation) is exercised realistically.

You can also **upload your own CSV** on the Dataset page. To train on it, the
file must contain the same feature columns plus the `cognitive_risk_label`
target.

---

## ⚠️ Limitations

- The dataset is **synthetic** and does not represent real patients.
- There is **no clinical validation**, regulatory approval, or peer review.
- Speech, gait and sensor features are **simulated**, not extracted from raw
  signals.
- Predictions reflect patterns in the demo data only and must **not** guide care.
- Real-world performance would differ substantially from the demo metrics.

---

## 🔭 Possible extensions

- Real **speech/audio** biomarker extraction from recordings.
- **Computer vision** for facial/gait analysis from video.
- **Wearable sensor** integration (activity, sleep, heart rate).
- Prospective **clinical validation** against gold-standard assessments.

---

## 🛠️ Technology

Python · Streamlit · pandas · NumPy · scikit-learn · matplotlib · plotly · joblib
