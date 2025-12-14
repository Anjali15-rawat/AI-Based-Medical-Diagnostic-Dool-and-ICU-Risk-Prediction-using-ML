# model_engine.py
# Core AI engine for MedAI:
# - Synthetic XGBoost model for ICU vitals risk
# - Google Gemini 2.5 Flash for multimodal report/file analysis
# - Optional offline training pipeline for real Sepsis PSV data

import os
import re
import json
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier

import google.generativeai as genai


# =========================
#  GEMINI CONFIG
# =========================

MODEL_NAME = "gemini-2.5-flash-lite"  # or "gemini-2.5-flash-lite"

# 1) Try environment variable first
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# 2) Local fallback for demo ONLY
#    >>> REPLACE THIS STRING WITH YOUR REAL API KEY ON YOUR MACHINE <<<
if not GOOGLE_API_KEY:
    GOOGLE_API_KEY = "AIzaSyAFjgFoFoGxBw3HcqtgMgh-4DHg4ScZVZk"  # <-- put your key here locally

if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)
    print("🔑 Gemini API key loaded successfully.")
else:
    print(
        "⚠️ GOOGLE_API_KEY is not set. "
        "Gemini-based report analysis will be disabled until you configure it."
    )


class MedicalAITool:
    def __init__(self):
        self.model_path = "models"
        os.makedirs(self.model_path, exist_ok=True)

        print("⚡ Initializing Medical AI Engine (XGBoost + Gemini)...")


        # Synthetic vitals XGBoost model (for fast demo)
        self.vitals_model = self._load_or_train_vitals()

        # Optional: real sepsis model trained from PSV
        self.sepsis_model = None
        sepsis_model_file = os.path.join(self.model_path, "sepsis_xgb_real.pkl")
        if os.path.exists(sepsis_model_file):
            try:
                self.sepsis_model = joblib.load(sepsis_model_file)
                print("✅ Loaded real Sepsis XGBoost model.")
            except Exception as e:
                print(f"⚠️ Failed to load sepsis model: {e}")

        # Gemini multimodal model (if API key provided)
        self.vision_model = (
            genai.GenerativeModel(MODEL_NAME) if GOOGLE_API_KEY else None
        )

    # ------------------------------------------------------------------
    #  Synthetic ICU vitals model (fast, for live demo)
    # ------------------------------------------------------------------
    def _load_or_train_vitals(self):
        """Load or train a small XGBoost classifier for ICU vitals risk."""
        model_file = os.path.join(self.model_path, "vitals_xgb.pkl")

        if os.path.exists(model_file):
            print("✅ Loaded existing vitals XGBoost model.")
            return joblib.load(model_file)

        print("🚀 Training vitals XGBoost model (first run only)...")
        np.random.seed(42)
        n = 1200  # enough for variety but trains quickly

        df = pd.DataFrame(
            {
                "age": np.random.normal(65, 15, n),
                "heart_rate": np.random.normal(85, 20, n),
                "systolic_bp": np.random.normal(130, 25, n),
                "diastolic_bp": np.random.normal(80, 15, n),
                "respiratory_rate": np.random.normal(18, 6, n),
                "temperature": np.random.normal(37.2, 1.2, n),
                "oxygen_saturation": np.random.normal(96, 4, n),
                "wbc_count": np.random.normal(8, 3, n),
            }
        )

        # Stricter, clinically-inspired rule so we actually get "low risk" examples
        score = np.zeros(len(df))

        # Severe tachycardia
        score += (df["heart_rate"] > 110).astype(int) * 2
        # Significant hypoxia
        score += (df["oxygen_saturation"] < 90).astype(int) * 2
        # High fever
        score += (df["temperature"] > 39.0).astype(int) * 1
        # Very low systolic BP
        score += (df["systolic_bp"] < 90).astype(int) * 2

        # Binary target: high risk only if combined abnormalities are strong
        y = (score >= 3).astype(int)

        model = XGBClassifier(
            n_estimators=80,
            max_depth=4,
            learning_rate=0.12,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=42,
            eval_metric="logloss",
            use_label_encoder=False,
        )
        model.fit(df, y)

        joblib.dump(model, model_file)
        print(f"💾 Saved vitals model to {model_file}")
        return model

    # ------------------------------------------------------------------
    #  Rule-based + XGBoost ICU vitals analysis
    # ------------------------------------------------------------------
    def analyze_vitals(self, data: dict) -> dict:
        """
        Analyze ICU vitals using a rule-based clinical score (primary)
        plus a small XGBoost adjustment (secondary).
        """

        # ---- 1) Parse inputs safely ----
        age = float(data.get("age", 0))
        hr = float(data.get("heart_rate", 0))
        sbp = float(data.get("systolic_bp", 0))
        dbp = float(data.get("diastolic_bp", 0))
        rr = float(data.get("respiratory_rate", 0))
        temp = float(data.get("temperature", 0))
        spo2 = float(data.get("oxygen_saturation", 0))
        wbc = float(data.get("wbc_count", 0))

        # ---- 2) Rule-based clinical score (primary) ----
        score = 0.0
        max_score = 20.0  # used to normalize to 0–100

        # Age
        if age >= 80:
            score += 3
        elif age >= 65:
            score += 2
        elif age >= 50:
            score += 1

        # Heart Rate (bpm)
        if hr >= 140 or hr <= 40:
            score += 4
        elif hr >= 120 or hr <= 50:
            score += 3
        elif hr >= 100:
            score += 2

        # Systolic BP (mmHg)
        if sbp < 80:
            score += 4
        elif sbp < 90:
            score += 3
        elif sbp < 100:
            score += 2

        # Respiratory Rate (breaths/min)
        if rr >= 30 or rr <= 8:
            score += 3
        elif rr >= 24:
            score += 2
        elif rr >= 20:
            score += 1

        # Temperature (°C)
        if temp >= 40.0 or temp <= 35.0:
            score += 3
        elif temp >= 38.5 or temp <= 36.0:
            score += 2
        elif temp >= 37.5:
            score += 1

        # Oxygen Saturation (%)
        if spo2 < 85:
            score += 4
        elif spo2 < 90:
            score += 3
        elif spo2 < 94:
            score += 2

        # WBC count (10^9/L)
        if wbc >= 20 or wbc <= 3:
            score += 3
        elif wbc >= 15 or wbc <= 4:
            score += 2
        elif wbc >= 12:
            score += 1

        # Normalize to 0–100
        rule_risk = max(0.0, min(100.0, round((score / max_score) * 100.0, 1)))

        # ---- 3) XGBoost adjustment (secondary, small effect) ----
        ml_adjust = 0.0
        if self.vitals_model is not None:
            x = np.array([[age, hr, sbp, dbp, rr, temp, spo2, wbc]])
            try:
                prob = float(self.vitals_model.predict_proba(x)[0][1])
                # Convert prob (0–1) centered at 0.5 into a small +/- adjustment
                # Range approx -10 to +10
                ml_adjust = (prob - 0.5) * 20.0
            except Exception as e:
                print(f"⚠️ XGBoost vitals adjustment failed: {e}")
                ml_adjust = 0.0

        # Final risk combines rule-based score + small ML nudge
        risk_score = rule_risk + ml_adjust
        risk_score = max(0.0, min(100.0, round(risk_score, 1)))

        # ---- 4) Convert to label ----
        if risk_score >= 75:
            risk_label = "High Risk"
        elif risk_score >= 40:
            risk_label = "Moderate"
        else:
            risk_label = "Low"

        # ---- 5) Human-readable findings ----
        alerts = []

        if spo2 < 94:
            if spo2 < 85:
                alerts.append("Severe hypoxia (very low oxygen saturation).")
            elif spo2 < 90:
                alerts.append("Significant hypoxia (low oxygen saturation).")
            else:
                alerts.append("Mild reduction in oxygen saturation.")

        if hr >= 100:
            if hr >= 140:
                alerts.append("Severe tachycardia (very high heart rate).")
            elif hr >= 120:
                alerts.append("Moderate tachycardia (elevated heart rate).")
            else:
                alerts.append("Mild tachycardia.")

        if sbp < 100:
            if sbp < 80:
                alerts.append("Severely low systolic blood pressure – risk of shock.")
            elif sbp < 90:
                alerts.append(
                    "Low systolic blood pressure – possible hemodynamic instability."
                )
            else:
                alerts.append("Borderline low systolic blood pressure.")

        if rr >= 20 or rr <= 8:
            if rr >= 30 or rr <= 8:
                alerts.append("Marked respiratory distress or suppression.")
            elif rr >= 24:
                alerts.append("Increased respiratory effort.")
            else:
                alerts.append("Slightly elevated respiratory rate.")

        if temp >= 38.5:
            if temp >= 40.0:
                alerts.append("Very high fever – severe systemic stress.")
            else:
                alerts.append("Fever – possible infection or inflammatory process.")
        elif temp <= 36.0:
            alerts.append("Low body temperature – possible hypothermia or sepsis.")

        if wbc >= 12 or wbc <= 4:
            if wbc >= 20 or wbc <= 3:
                alerts.append(
                    "Marked abnormal WBC count – strong infection/inflammation signal."
                )
            else:
                alerts.append(
                    "Abnormal WBC count – possible infection or stress response."
                )

        if not alerts:
            alerts = [
                "Vitals are largely within acceptable range.",
                "Continue routine monitoring and trend observation.",
            ]

        return {
            "risk_score": risk_score,
            "risk_label": risk_label,
            "findings": alerts,
            "source": "Rule-based ICU Score + XGBoost Adjustment",
        }

    # ------------------------------------------------------------------
    #  Gemini multimodal file analysis
    # ------------------------------------------------------------------
    def _build_scan_context(self, scan_type: str) -> str:
        """Map dropdown scan_type to clinical context text for Gemini."""
        mapping = {
            "normal": "Routine checkup, expected mostly normal findings.",
            "xray_fracture": "Bone X-Ray with suspected fracture.",
            "xray_pneumonia": (
                "Chest X-Ray to evaluate for pneumonia or viral infection."
            ),
            "mri_tumor": "Brain MRI with suspected tumor or mass lesion.",
            "ecg_arrhythmia": "12-lead ECG to check for arrhythmia or ischemia.",
            "blood_infection": "Blood test to detect infection or sepsis.",
        }
        return mapping.get(scan_type, "General diagnostic medical report.")

    def analyze_file(self, file_path: str, scan_type: str) -> dict:
        """
        Analyze an uploaded report/image/PDF using Google Gemini 2.5 Flash.
        Returns a structured JSON-like dict for the frontend.
        """
        if not GOOGLE_API_KEY or self.vision_model is None:
            return {
                "risk_score": 0,
                "risk_label": "Error",
                "findings": [
                    "Gemini API key not configured.",
                    "Set GOOGLE_API_KEY or hard-code the key to enable cloud analysis.",
                ],
                "source": "System Fallback",
            }

        try:
            print(f"🤖 Uploading file to Gemini: {file_path}")
            uploaded_file = genai.upload_file(path=file_path)

            context = self._build_scan_context(scan_type)

            prompt = f"""
You are an expert medical diagnostic AI. Analyze the attached medical file
(X-ray / MRI / CT / ECG / blood report / PDF) with the clinical context below:

Clinical context: {context}

Tasks:
1. Identify key abnormalities, diseases, or risk indicators.
2. Estimate a risk_score from 0–100 indicating severity.
3. Choose a risk_label from: "Low", "Moderate", "High", "Critical".
4. Provide 2–4 short clinical findings in simple language.

Return ONLY raw JSON (no markdown, no commentary). EXACT structure:

{{
  "risk_score": 85.0,
  "risk_label": "Critical",
  "findings": ["Finding 1", "Finding 2", "Finding 3"],
  "source": "Gemini 2.5 Flash"
}}
"""

            # For google-generativeai, passing a list as first arg is treated as "contents"
            response = self.vision_model.generate_content([prompt, uploaded_file])
            raw_text = response.text.strip()

            # Strip possible markdown fences
            raw_text = raw_text.replace("```json", "").replace("```", "").strip()

            # Attempt to extract the first JSON object
            match = re.search(r"{.*}", raw_text, re.DOTALL)
            json_text = match.group(0) if match else raw_text

            result = json.loads(json_text)

            # Normalize keys
            result.setdefault("risk_score", 0)
            result.setdefault("risk_label", "Unknown")
            result.setdefault("findings", ["No findings returned by Gemini."])
            result.setdefault("source", "Gemini 2.5 Flash")

            try:
                result["risk_score"] = float(result["risk_score"])
                result["risk_score"] = max(0, min(100, result["risk_score"]))
            except Exception:
                result["risk_score"] = 0.0

            return result

        except Exception as e:
            print(f"❌ Gemini error: {e}")
            return {
                "risk_score": 0,
                "risk_label": "Error",
                "findings": [
                    "AI analysis failed. Please verify API key, model name, and network connectivity.",
                    f"Internal error: {str(e)}",
                ],
                "source": "System Error",
            }

    # ------------------------------------------------------------------
    #  Optional: real Sepsis XGBoost training from PSV files
    # ------------------------------------------------------------------
    def train_real_sepsis_from_psv(self, dataset_root: str = "clinical_data"):
        """
        Offline training pipeline for Sepsis prediction using
        PhysioNet-style PSV files (training_setA/training/p000001.psv).

        Run from CLI or notebook:
            python model_engine.py
        AFTER you extract the dataset to `dataset_root`.
        """
        extract_path = dataset_root
        training_setA_path = os.path.join(extract_path, "training_setA", "training")
        patient_file = os.path.join(training_setA_path, "p000001.psv")

        if not os.path.exists(patient_file):
            raise FileNotFoundError(
                f"Could not find {patient_file}. "
                f"Make sure you extracted the Sepsis dataset to '{dataset_root}'."
            )

        print("\n--- Starting Sepsis Data Preprocessing (Real PSV) ---")
        df_sepsis = pd.read_csv(patient_file, sep="|")
        print(f"Loaded PSV sample: {df_sepsis.shape} rows")

        # LOCF imputation on core vitals/labs
        vitals_labs = ["HR", "MAP", "Temp", "O2Sat", "Lactate", "WBC"]
        df_sepsis[vitals_labs] = df_sepsis[vitals_labs].ffill()
        print("LOCF imputation done on vitals and labs.")

        # Feature engineering
        WINDOW = 6
        df_sepsis["Delta_MAP_6hr"] = df_sepsis["MAP"].diff(periods=WINDOW).fillna(0)
        df_sepsis["Max_Lactate_Value"] = df_sepsis["Lactate"].max()
        print("Delta_MAP_6hr & Max_Lactate_Value created.")

        # Normalization for selected features
        scaler = MinMaxScaler()
        df_sepsis[["HR", "MAP", "Delta_MAP_6hr"]] = scaler.fit_transform(
            df_sepsis[["HR", "MAP", "Delta_MAP_6hr"]]
        )
        print("Scaled HR, MAP, Delta_MAP_6hr.")

        feature_columns = [
            "HR",
            "O2Sat",
            "Temp",
            "MAP",
            "Resp",
            "BaseExcess",
            "HCO3",
            "pH",
            "PaCO2",
            "Lactate",
            "WBC",
            "Age",
            "Delta_MAP_6hr",
            "Max_Lactate_Value",
        ]

        missing_cols = [c for c in feature_columns if c not in df_sepsis.columns]
        if missing_cols:
            print(
                f"⚠️ Columns missing from PSV file and filled with 0: {missing_cols}"
            )
            for c in missing_cols:
                df_sepsis[c] = 0

        X = df_sepsis[feature_columns].copy().fillna(0)
        y = df_sepsis["SepsisLabel"].copy()

        print(f"Final training samples: {X.shape[0]}")

        if y.nunique() < 2:
            print(
                "⚠️ Only one class present in SepsisLabel. "
                "Cannot train a classifier on a single-class sample."
            )
            return

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        xgb_model = XGBClassifier(
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=42,
        )
        xgb_model.fit(X_train, y_train)

        y_pred_proba = xgb_model.predict_proba(X_test)[:, 1]
        auc_score = roc_auc_score(y_test, y_pred_proba)

        print(f"✅ XGBoost Sepsis model trained. AUC-ROC: {auc_score:.4f}")

        model_file = os.path.join(self.model_path, "sepsis_xgb_real.pkl")
        joblib.dump(xgb_model, model_file)
        self.sepsis_model = xgb_model
        print(f"💾 Saved real Sepsis model to {model_file}")


if __name__ == "__main__":
    # Optional offline training entry point
    ai = MedicalAITool()
    # ai.train_real_sepsis_from_psv(dataset_root="clinical_data")
