"""
Diabetes Risk Predictor — Flask Web App
Converts the Streamlit app/app.py into a simple Flask + HTML application.
"""

from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = BASE_DIR / "model"

app = Flask(__name__)

# ── Load model artefacts once at startup ────────────────────────────────────
with open(MODEL_DIR / "xgb_model.pkl", "rb") as f:
    model = pickle.load(f)

with open(MODEL_DIR / "feature_names.pkl", "rb") as f:
    feature_names = pickle.load(f)

# ── Field metadata ────────────────────────────────────────────────────────────
FIELD_META = {
    "HighBP":              {"label": "High Blood Pressure",                    "type": "bool"},
    "HighChol":            {"label": "High Cholesterol",                       "type": "bool"},
    "CholCheck":           {"label": "Cholesterol Checked (last 5 yrs)",       "type": "bool"},
    "BMI":                 {"label": "BMI",                                    "type": "slider", "min": 10, "max": 80, "default": 27},
    "Stroke":              {"label": "Ever Had a Stroke",                      "type": "bool"},
    "HeartDiseaseorAttack":{"label": "Heart Disease or Attack",                "type": "bool"},
    "Smoker":              {"label": "Smoker (100+ cigarettes lifetime)",       "type": "bool"},
    "PhysActivity":        {"label": "Physically Active (last 30 days)",       "type": "bool"},
    "Fruits":              {"label": "Eats Fruit Daily",                       "type": "bool"},
    "Veggies":             {"label": "Eats Vegetables Daily",                  "type": "bool"},
    "HvyAlcoholConsump":   {"label": "Heavy Alcohol Consumption",              "type": "bool"},
    "AnyHealthcare":       {"label": "Has Health Insurance",                   "type": "bool"},
    "NoDocbcCost":         {"label": "Could Not See Doctor Due to Cost",       "type": "bool"},
    "GenHlth":             {"label": "General Health (1=Excellent, 5=Poor)",   "type": "slider", "min": 1, "max": 5,  "default": 3},
    "MentHlth":            {"label": "Poor Mental Health Days (last 30 days)", "type": "slider", "min": 0, "max": 30, "default": 0},
    "PhysHlth":            {"label": "Poor Physical Health Days (last 30 days)","type": "slider","min": 0, "max": 30, "default": 0},
    "DiffWalk":            {"label": "Difficulty Walking",                     "type": "bool"},
    "Sex":                 {"label": "Sex (0=Female, 1=Male)",                 "type": "slider", "min": 0, "max": 1,  "default": 0},
    "Age":                 {"label": "Age Group (1=18-24 to 13=80+)",          "type": "slider", "min": 1, "max": 13, "default": 7},
    "Education":           {"label": "Education Level (1=None, 6=College)",    "type": "slider", "min": 1, "max": 6,  "default": 4},
    "Income":              {"label": "Income Level (1=<$10K, 8=>$75K)",        "type": "slider", "min": 1, "max": 8,  "default": 5},
}


def build_input(form) -> pd.DataFrame:
    row = {}
    for feat in feature_names:
        meta = FIELD_META[feat]
        raw = form.get(feat, meta.get("default", 0))
        row[feat] = float(raw)
    return pd.DataFrame([row])[feature_names]


def risk_band(prob: float) -> dict:
    if prob < 0.30:
        return {"label": "LOW RISK",      "emoji": "🟢", "css": "low",      "msg": "Low diabetes risk detected."}
    elif prob < 0.60:
        return {"label": "MODERATE RISK", "emoji": "🟡", "css": "moderate", "msg": "Moderate diabetes risk detected."}
    else:
        return {"label": "HIGH RISK",     "emoji": "🔴", "css": "high",     "msg": "High diabetes risk detected."}


@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    form_values = {feat: FIELD_META[feat].get("default", 0) for feat in feature_names}

    if request.method == "POST":
        form_values = {feat: request.form.get(feat, FIELD_META[feat].get("default", 0))
                       for feat in feature_names}
        input_df = build_input(request.form)
        prob = float(model.predict_proba(input_df)[0][1])
        band = risk_band(prob)

        summary = [
            {"feature": feat,
             "label":   FIELD_META[feat]["label"],
             "value":   form_values[feat]}
            for feat in feature_names
        ]

        result = {
            "prob":       prob,
            "prob_pct":   f"{prob:.1%}",
            "confidence": f"{max(prob, 1 - prob):.1%}",
            "band":       band,
            "bar_width":  int(prob * 100),
            "summary":    summary,
        }

    return render_template("index.html",
                           fields=FIELD_META,
                           feature_names=feature_names,
                           form_values=form_values,
                           result=result)


if __name__ == "__main__":
    app.run(debug=True, port=5000)
