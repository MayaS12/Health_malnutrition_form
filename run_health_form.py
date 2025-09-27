import streamlit as st
import joblib
import numpy as np
import pandas as pd
import gdown
import os

# -------------------
# Page config
# -------------------
st.set_page_config(layout="wide")

@st.cache_resource
def load_model():
    files = {
        "health_form_rf.pkl": "14CaWIkRbQRLDeyR6lHB0qgjQ09XWciws",
        "scaler.pkl": "1dhyIiDyz36SODfU4ITdTJmuYKbn1-p5a",
        "label_encoder.pkl": "1a_4ZC7zFE4Qc0k8L501rBfnhynJ56a7G"
    }

    for fname, file_id in files.items():
        if not os.path.exists(fname):
            url = f"https://drive.google.com/uc?id={file_id}"
            gdown.download(url, fname, quiet=False)

    model = joblib.load("health_form_rf.pkl")
    scaler = joblib.load("scaler.pkl")
    target_le = joblib.load("label_encoder.pkl")
    return model, scaler, target_le

model, scaler, target_le = load_model()


# -------------------
# Multi-language dictionary
# -------------------
translations = {
    "en": {"title": "Malnutrition Risk Prediction", "age": "Age (in months)", "sex": "Sex",
           "male": "Male", "female": "Female", "height": "Height (cm)", "weight": "Weight (kg)",
           "muac": "MUAC (cm)", "predict": "Predict", "results": "Prediction Results",
           "recommend": "Recommendations", "stunted": "Stunted", "wasted": "Wasted",
           "underweight": "Underweight", "normal": "Normal Growth",
           "probability": "Probability (%)",
           "input_warning": "⚠️ Entered values are unusual, please double-check.",
           "prediction_is": "Prediction is"},
    "hi": {"title": "कुपोषण जोखिम भविष्यवाणी", "age": "आयु (महीनों में)", "sex": "लिंग",
           "male": "लड़का", "female": "लड़की", "height": "लंबाई (सेमी)", "weight": "वजन (किग्रा)",
           "muac": "MUAC (सेमी)", "predict": "भविष्यवाणी करें", "results": "परिणाम",
           "recommend": "शिफ़ारिशें", "stunted": "ठिगनेपन", "wasted": "क्षीणता",
           "underweight": "कम वजन", "normal": "सामान्य विकास",
           "probability": "संभावना (%)",
           "input_warning": "⚠️ दर्ज किए गए मान असामान्य हैं, कृपया जाँच करें।",
           "prediction_is": "भविष्यवाणी है"},
    "kn": {"title": "ಪೋಷಣಾ ಕೊರತೆ ಅಪಾಯ ಭವಿಷ್ಯವಾಣಿ", "age": "ವಯಸ್ಸು (ತಿಂಗಳುಗಳಲ್ಲಿ)", "sex": "ಲಿಂಗ",
           "male": "ಹುಡುಗ", "female": "ಹುಡುಗಿ", "height": "ಎತ್ತರ (ಸೆಂಮೀ)", "weight": "ತೂಕ (ಕೆಜಿ)",
           "muac": "MUAC (ಸೆಂಮೀ)", "predict": "ಭವಿಷ್ಯವಾಣಿ ಮಾಡಿ", "results": "ಫಲಿತಾಂಶ",
           "recommend": "ಶಿಫಾರಸುಗಳು", "stunted": "ಹೆಬ್ಬಿನ ಅಪಾಯ", "wasted": "ತೂಕ ಕಳೆದುಕೊಳ್ಳುವ ಅಪಾಯ",
           "underweight": "ಕಡಿಮೆ ತೂಕ ಅಪಾಯ", "normal": "ಸಾಮಾನ್ಯ ಬೆಳವಣಿಗೆ",
           "probability": "ಸಂಭಾವ್ಯತೆ (%)",
           "input_warning": "⚠️ ನಮೂದಿಸಿದ ಮೌಲ್ಯಗಳು ಅಸಾಮಾನ್ಯವಾಗಿವೆ, ದಯವಿಟ್ಟು ಪರಿಶೀಲಿಸಿ.",
           "prediction_is": "ಭವಿಷ್ಯವಾಣಿ"}
}

# -------------------
# Sidebar language names in their own language
# -------------------
lang_options = {"en": "English", "hi": "हिन्दी", "kn": "ಕನ್ನಡ"}
lang = st.sidebar.selectbox("Language / भाषा / ಭಾಷೆ", list(lang_options.keys()),
                            format_func=lambda x: lang_options[x])
t = translations[lang]

# -------------------
# Centered App Heading
# -------------------
app_heading = {
    "en": "Malnutrition Form",
    "hi": "कुपोषण फॉर्म",
    "kn": "ಪೋಷಣಾ ಕೊರತೆ ಫಾರ್ಮ್"
}

st.markdown(f"<h1 style='text-align: center;'>{app_heading[lang]}</h1>", unsafe_allow_html=True)

# -------------------
# Numbers translation maps
# -------------------
num_map_hi = str.maketrans("0123456789.", "०१२३४५६७८९.")
num_map_kn = str.maketrans("0123456789.", "೦೧೨೩೪೫೬೭೮೯.")

def translate_number(n, lang):
    """Translate numeric value to Hindi/Kannada numerals"""
    n_str = f"{n:.2f}"
    if lang == "hi":
        return n_str.translate(num_map_hi)
    elif lang == "kn":
        return n_str.translate(num_map_kn)
    else:
        return n_str

# -------------------
# Input form with localized numbers
# -------------------
def get_localized_number_input(label, lang, value=0.0, step=1.0):
    """
    Uses text_input to allow localized numerals and converts to float internally
    """
    # Display number in localized form
    display_value = translate_number(value, lang)
    input_str = st.text_input(label, value=display_value)
    
    # Convert Hindi/Kannada numerals back to float for calculation
    if lang == "hi":
        input_str_std = input_str.translate(str.maketrans("०१२३४५६७८९", "0123456789"))
    elif lang == "kn":
        input_str_std = input_str.translate(str.maketrans("೦೧೨೩೪೫೬೭೮೯", "0123456789"))
    else:
        input_str_std = input_str
    
    try:
        return float(input_str_std)
    except:
        return 0.0

with st.form("child_form"):
    age = get_localized_number_input(t["age"], lang, value=12, step=1)
    height = get_localized_number_input(t["height"], lang, value=90, step=0.1)
    weight = get_localized_number_input(t["weight"], lang, value=12, step=0.1)
    muac = get_localized_number_input(t["muac"], lang, value=12, step=0.1)
    sex = st.radio(t["sex"], [t["male"], t["female"]])
    submit = st.form_submit_button(t["predict"])


# -------------------
# Recommendations
# -------------------
recs = {
    "en": {
        "Stunted": "📌 Stunting: low height for age due to chronic malnutrition. Monitor height monthly. Feed ragi mudde, egg curry, milk, pulses, and greens.",
        "Wasted": "📌 Wasting: low weight for height, often from illness. Provide energy-dense foods like idli with ghee, chapati with dal, banana. Treat infections promptly.",
        "Underweight": "📌 Underweight: low weight for age. Feed frequent balanced meals with cereals (ragi, rice), pulses, dairy, vegetables. Check anemia.",
        "Normal": "✅ Growth appears normal. Maintain balanced diet and monitor regularly."
    },
    "hi": {
        "Stunted": "📌 ठिगनेपन: उम्र के अनुसार लंबाई कम, लंबे समय के कुपोषण से। लंबाई मासिक देखें। रागी मुड्डे, अंडा करी, दूध, दाल और हरी सब्जियाँ दें।",
        "Wasted": "📌 क्षीणता: लंबाई के अनुसार वजन कम, अक्सर बीमारी से। इडली घी के साथ, चपाती दाल के साथ, केला दें। संक्रमण शीघ्र इलाज करें।",
        "Underweight": "📌 कम वजन: उम्र के अनुसार वजन कम। बार-बार संतुलित भोजन दें: रागी, चावल, दाल, दूध, सब्जियाँ। एनीमिया जांचें।",
        "Normal": "✅ विकास सामान्य। संतुलित आहार बनाए रखें और नियमित निगरानी करें।"
    },
    "kn": {
        "Stunted": "📌 ಹೆಬ್ಬಿನ ಅಪಾಯ: ವಯಸ್ಸಿಗೆ ತಕ್ಕ ಎತ್ತರ ಕಡಿಮೆ, ದೀರ್ಘಕಾಲೀನ ಪೋಷಣಾ ಕೊರತೆ. ಎತ್ತರವನ್ನು ಮಾಸಿಕವಾಗಿ ಗಮನಿಸಿ. ರಾಗಿ ಮುಡ್ಡೆ, ಮೊಟ್ಟೆ करी, ಹಾಲು, ದಾಲ್ ಮತ್ತು ಹಸಿರು ತರಕಾರಿಗಳು ನೀಡಿ.",
        "Wasted": "📌 ತೂಕ ಕಳೆದುಕೊಳ್ಳುವ ಅಪಾಯ: ಎತ್ತರಕ್ಕೆ ತಕ್ಕ ತೂಕ ಕಡಿಮೆ, ಸಾಮಾನ್ಯವಾಗಿ ಅಸ್ವಸ್ಥತೆ. ಶಕ್ತಿ-ಸಮೃದ್ಧ ಆಹಾರ: ಇಡ್ಲಿ ತುಪ್ಪದೊಂದಿಗೆ, ಚಪಾತಿ ದಾಲ್ ಜೊತೆ, ಬಾಳೆಹಣ್ಣು. ಸೋಂಕುಗಳನ್ನು ತಕ್ಷಣ ಚಿಕಿತ್ಸೆ ಮಾಡಿ.",
        "Underweight": "📌 ಕಡಿಮೆ ತೂಕ: ವಯಸ್ಸಿಗೆ ತಕ್ಕ ತೂಕ ಕಡಿಮೆ. ಸಮತೋಲ ಆಹಾರ: ಪುನಃ ಪುನಃ ಆಹಾರ, ರಾಗಿ, ಅಕ್ಕಿ, ದಾಲ್, ಹಾಲು, ಹಸಿರು ತರಕಾರಿಗಳು. ಅನೀಮಿಯಾ ಪರಿಶೀಲಿಸಿ.",
        "Normal": "✅ ಬೆಳವಣಿಗೆ ಸಾಮಾನ್ಯವಾಗಿದೆ. ಸಮತೋಲ ಆಹಾರ ಮತ್ತು ನಿಯಮಿತ ಮಾನಿಟರಿಂಗ್ ಮುಂದುವರಿಸಿ."
    }
}

# -------------------
# Risk color
# -------------------
def risk_color(pred_label):
    if pred_label in ["Stunted", "Wasted", "Underweight"]:
        return "🔴"
    return "🟢"

# -------------------
# Prediction
# -------------------
if submit:
    try:
        # Basic validation
        if age <= 0 or height <= 0 or weight <= 0 or muac <= 0:
            st.warning(t["input_warning"])
        else:
            sex_num = 1 if sex == t["male"] else 0
            height_m = height / 100
            bmi = weight / (height_m ** 2)
            muac_per_month = muac / max(age, 1)

            features = np.array([[sex_num, age, weight, height, muac, bmi, muac_per_month]])
            features_scaled = scaler.transform(features)

            probs = model.predict_proba(features_scaled)[0]
            pred_class = model.predict(features_scaled)[0]
            pred_label = target_le.inverse_transform([pred_class])[0]

            # Translate labels
            label_map = {
                "Stunted": t["stunted"], "Wasted": t["wasted"],
                "Underweight": t["underweight"], "Normal": t["normal"]
            }
            max_idx = np.argmax(probs)
            max_pred_label = target_le.inverse_transform([model.classes_[max_idx]])[0]
            max_pred_label_translated = label_map[max_pred_label]

            # Show prediction
            st.markdown(
                f"<h2 style='color:blue'>{t['prediction_is']}: {max_pred_label_translated}</h2>",
                unsafe_allow_html=True
            )

            # Table with localized numbers
            probs_translated = [translate_number(p * 100, lang) for p in probs]
            table_labels = [label_map[label] for label in target_le.inverse_transform(model.classes_)]
            results_df = pd.DataFrame({t["results"]: table_labels, t["probability"]: probs_translated})

            st.subheader(t["results"])
            st.table(results_df)
            st.markdown(f"{risk_color(pred_label)} **{t['recommend']}**")
            st.write(recs[lang].get(pred_label, recs[lang]["Normal"]))

    except Exception as e:
        st.error(f"⚠️ {t['input_warning']}")
        st.exception(e)  # For debugging only, you can remove in production
