import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import joblib
import os

# -------------------
# Load Data
# -------------------
df = pd.read_csv("Child_data.csv")

# -------------------
# Preprocessing & Feature Engineering
# -------------------
raw_cols = ["Sex", "Age_in_months", "Weight_kg", "Height_cm", "MUAC_cm"]
df = df[raw_cols + ["Growth_status"]].copy()

# Encode Sex
df["Sex"] = df["Sex"].astype(str)
le_sex = LabelEncoder()
df["Sex"] = le_sex.fit_transform(df["Sex"])

# Feature engineering
df["Height_m"] = df["Height_cm"] / 100
df["BMI"] = df["Weight_kg"] / (df["Height_m"] ** 2)
df["MUAC_per_month"] = df["MUAC_cm"] / df["Age_in_months"]

# Encode target
target_le = LabelEncoder()
df["Growth_status_encoded"] = target_le.fit_transform(df["Growth_status"])

# Features and target
feature_cols = raw_cols + ["BMI", "MUAC_per_month"]
X = df[feature_cols]
y = df["Growth_status_encoded"]

# Fill missing values
X = X.fillna(X.median(numeric_only=True))

# Scale numeric features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -------------------
# Train/Test Split
# -------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# -------------------
# Hyperparameter tuning for Random Forest
# -------------------
rf = RandomForestClassifier(class_weight="balanced", random_state=42, n_jobs=-1)

param_dist = {
    "n_estimators": [500, 700, 1000],
    "max_depth": [None, 8, 10],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "max_features": ["sqrt", "log2", None]
}

search = RandomizedSearchCV(
    rf, param_distributions=param_dist, n_iter=20, scoring="accuracy", cv=5, verbose=1, random_state=42
)

search.fit(X_train, y_train)
model = search.best_estimator_

# -------------------
# Predictions
# -------------------
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)
normal_class_idx = np.where(target_le.classes_ == "Normal")[0][0]
risk_scores = 1 - y_prob[:, normal_class_idx]

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=target_le.classes_))

# -------------------
# WHO Verification Step (with clipping fix)
# -------------------
def calc_zscore(x, L, M, S):
    # avoid negative/zero values for fractional powers
    x_safe = np.maximum(x, 0.001)
    M_safe = np.maximum(M, 0.001)
    if L != 0:
        return ((x_safe / M_safe) ** L - 1) / (L * S)
    else:
        return np.log(x_safe / M_safe) / S

def load_who_table(indicator, sex, age_months):
    sex_str = "boys" if sex == 1 else "girls"

    if indicator == "wfa":
        fname = f"who_reference/wfa_{sex_str}_0-to-5-years_zscores.xlsx"
    elif indicator == "hfa":
        fname = f"who_reference/lhfa_{sex_str}_0-to-2-years_zscores.xlsx" if age_months <= 24 else f"who_reference/lhfa_{sex_str}_2-to-5-years_zscores.xlsx"
    elif indicator == "bfa":
        fname = f"who_reference/bmi_{sex_str}_0-to-2-years_zscores.xlsx" if age_months <= 24 else f"who_reference/bmi_{sex_str}_2-to-5-years_zscores.xlsx"
    elif indicator == "wfh":
        fname = f"who_reference/wfl_{sex_str}_0-to-2-years_zscores.xlsx" if age_months <= 24 else f"who_reference/wfh_{sex_str}_2-to-5-years_zscores.xlsx"
    else:
        raise ValueError(f"Unknown indicator: {indicator}")

    if not os.path.exists(fname):
        fname = fname.replace("zscores", "zcores")
        if not os.path.exists(fname):
            raise FileNotFoundError(f"WHO file not found: {fname}")

    df = pd.read_excel(fname)
    df.columns = [c.strip() for c in df.columns]
    if "Length" in df.columns: df = df.rename(columns={"Length": "Month"})
    if "Height" in df.columns: df = df.rename(columns={"Height": "Month"})
    if "M       " in df.columns: df = df.rename(columns={"M       ": "M"})
    if "SD" in df.columns: df = df.drop(columns=["SD"])
    return df

def get_who_status(row):
    sex = row["Sex"]
    age = row["Age_in_months"]
    weight = row["Weight_kg"]
    height = row["Height_cm"]

    wfa_table = load_who_table("wfa", sex, age)
    hfa_table = load_who_table("hfa", sex, age)
    wfh_table = load_who_table("wfh", sex, age)

    wfa_row = wfa_table.iloc[(wfa_table['Month'] - age).abs().argsort()[:1]]
    hfa_row = hfa_table.iloc[(hfa_table['Month'] - age).abs().argsort()[:1]]
    wfh_row = wfh_table.iloc[(wfh_table['Month'] - height).abs().argsort()[:1]]

    wfa_z = calc_zscore(weight, wfa_row['L'].values[0], wfa_row['M'].values[0], wfa_row['S'].values[0])
    hfa_z = calc_zscore(height, hfa_row['L'].values[0], hfa_row['M'].values[0], hfa_row['S'].values[0])
    wfh_z = calc_zscore(weight, wfh_row['L'].values[0], wfh_row['M'].values[0], wfh_row['S'].values[0])

    if hfa_z < -2:
        return "Stunted"
    elif wfa_z < -2:
        return "Underweight"
    elif wfh_z < -2:
        return "Wasted"
    else:
        return "Normal"

# WHO verification
df_raw_test = pd.DataFrame(X_test, columns=feature_cols)
who_status = df_raw_test.apply(get_who_status, axis=1)

# -------------------
# Output
# -------------------
output = pd.DataFrame({
    "True_Label": target_le.inverse_transform(y_test),
    "Predicted_Label": target_le.inverse_transform(y_pred),
    "Risk_Score": risk_scores,
    "Probabilities": list(y_prob),
    "WHO_Verified_Status": who_status
})

print(output.head())

print("\n--- Confusion Matrix (Model vs WHO) ---")
cm = confusion_matrix(output["Predicted_Label"], output["WHO_Verified_Status"], labels=target_le.classes_)
cm_df = pd.DataFrame(cm, index=target_le.classes_, columns=target_le.classes_)
print(cm_df)

agreement = (output["Predicted_Label"] == output["WHO_Verified_Status"]).mean()
print(f"\nAgreement between Model and WHO: {agreement:.2%}")

# -------------------
# Save Model + Scaler + LabelEncoder
# -------------------
joblib.dump(model, "health_form_rf.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(target_le, "label_encoder.pkl")

print("\n✅ Random Forest model, scaler, and label encoder exported successfully!")
