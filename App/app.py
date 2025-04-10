import streamlit as st
import joblib
import numpy as np
import pandas as pd  # Import Pandas
import os
from pathlib import Path

# --- Page Configuration (MUST BE THE FIRST st command) ---
st.set_page_config(page_title="ASD Screening Tool", layout="wide")

# --- Configuration & Model Loading ---
try:
    script_path = Path(__file__).resolve()
    project_dir = script_path.parent.parent
    MODEL_PATH = project_dir / 'Models' / 'best_model.joblib'
except NameError:
    MODEL_PATH = Path('Models') / 'best_model.joblib'

print(f"Attempting to load model from: {MODEL_PATH}")
print(f"Does the model file exist at this path? {os.path.exists(MODEL_PATH)}")

@st.cache_resource
def load_model(path):
    """Loads the joblib model file safely."""
    try:
        path_str = str(path)
        if os.path.exists(path_str):
            model_info = joblib.load(path_str)
            if 'model' in model_info:
                print("Model loaded successfully via cache.")
                return model_info['model']
            else:
                print("Error: 'model' key not found in the loaded joblib file.")
                return None
        else:
            print(f"Error: Model file not found at the specified path: {path_str}")
            return None
    except Exception as e:
        print(f"An unexpected error occurred during model loading: {e}")
        return None

model = load_model(MODEL_PATH)

# --- Static Data: Mappings & Questions ---
ethnicity_map = {
    "Asian": 0.0597, "Black": 0.1277, "Hispanic": 0.2222, "Latino": 0.2353,
    "Middle Eastern": 0.0619, "Others": 0.0383, "Pasifika": 0.1875,
    "South Asian": 0.0882, "Turkish": 0.0000, "White-European": 0.4708,
}
relation_map = {
    "Self": 0.2073, "Parent": 0.1538, "Relative": 0.1538,
    "Health care professional": 0.1538, "Others": 0.1538,
}

# Replace the existing region_map definition with the new country-based mapping
region_map = {
    'United States': 'North America', 'Canada': 'North America', 'Mexico': 'North America',
    'Nicaragua': 'North America', 'AmericanSamoa': 'North America', 'Bahamas': 'North America',
    'Brazil': 'South America', 'Argentina': 'South America', 'Bolivia': 'South America',
    'United Kingdom': 'Europe', 'Austria': 'Europe', 'Ukraine': 'Europe',
    'France': 'Europe', 'Netherlands': 'Europe', 'Italy': 'Europe',
    'Ireland': 'Europe', 'Russia': 'Europe', 'Serbia': 'Europe',
    'Sweden': 'Europe', 'Iceland': 'Europe', 'Germany': 'Europe',
    'Spain': 'Europe', 'Czech Republic': 'Europe', 'Romania': 'Europe',
    'Cyprus': 'Europe', 'Belgium': 'Europe',
    'Jordan': 'Middle East', 'United Arab Emirates': 'Middle East', 'Iraq': 'Middle East',
    'Oman': 'Middle East', 'Saudi Arabia': 'Middle East', 'Iran': 'Middle East',
    'Azerbaijan': 'Middle East', 'Armenia': 'Middle East',
    'India': 'Asia', 'Malaysia': 'Asia', 'Vietnam': 'Asia',
    'Sri Lanka': 'Asia', 'Hong Kong': 'Asia', 'China': 'Asia',
    'Pakistan': 'Asia', 'Japan': 'Asia', 'Bangladesh': 'Asia',
    'South Africa': 'Africa', 'Egypt': 'Africa', 'Ethiopia': 'Africa',
    'Angola': 'Africa', 'Sierra Leone': 'Africa', 'Niger': 'Africa',
    'Burundi': 'Africa',
    'New Zealand': 'Oceania', 'Australia': 'Oceania', 'Tonga': 'Oceania',
    'Kazakhstan': 'Central Asia',
    'Aruba': 'Caribbean'
}

# Add this new dictionary for region frequencies
region_frequencies = {
    'Africa': 0.1875,
    'Asia': 0.0612,
    'Caribbean': 0.5000,
    'Central Asia': 0.3000,
    'Europe': 0.2470,
    'Middle East': 0.0909,
    'North America': 0.4341,
    'Oceania': 0.0690,
    'South America': 0.1250,
    'Other': 0.1364
}

aq_questions_config = {
    "A1": {"text": "I often notice small sounds when others do not.", "scoring": "forward"},
    "A2": {"text": "When I’m reading a story, I find it difficult to work out the characters’ intentions.", "scoring": "forward"},
    "A3": {"text": "I find it easy to 'read between the lines' when someone is talking to me.", "scoring": "reverse"},
    "A4": {"text": "I usually concentrate more on the whole picture, rather than the small details.", "scoring": "reverse"},
    "A5": {"text": "I know how to tell if someone listening to me is getting bored.", "scoring": "reverse"},
    "A6": {"text": "I find it easy to do more than one thing at once.", "scoring": "reverse"},
    "A7": {"text": "I find it easy to work out what someone is thinking or feeling just by looking at their face.", "scoring": "reverse"},
    "A8": {"text": "If there is an interruption, I can switch back to what I was doing very quickly.", "scoring": "reverse"},
    "A9": {"text": "I like to collect information about categories of things.", "scoring": "forward"},
    "A10": {"text": "I find it difficult to work out people’s intentions.", "scoring": "forward"},
}
aq_options = ["Definitely Disagree", "Slightly Disagree", "Slightly Agree", "Definitely Agree"]

# Define the exact feature names expected by the trained model IN ORDER
EXPECTED_FEATURE_NAMES = [
    'A1_Score', 'A2_Score', 'A3_Score', 'A4_Score', 'A5_Score',
    'A6_Score', 'A7_Score', 'A8_Score', 'A9_Score', 'A10_Score',
    'age', 'gender', 'jaundice', 'austim', 'used_app_before', 'result',
    'ethnicity_target', 'relation_target', 'region_target'
]


# --- Streamlit App Layout & Logic ---
st.title("Autism Spectrum Disorder (ASD) Screening Tool")
st.markdown("Please answer the following questions based on how you typically feel or behave. This is a screening tool and **not** a diagnostic assessment. Consult a qualified healthcare provider for diagnosis.")
st.markdown("---")

if model is None:
    st.error("Critical Error: The prediction model could not be loaded. Please ensure the 'Models/best_model.joblib' file exists relative to the project root and is valid. The application cannot proceed.")
    st.stop()

user_inputs = {}

# --- Section 1: Screening Questions (AQ-10) ---
st.subheader("Screening Questions")
st.caption("Please select the option that best describes you for each statement.")
for i, (key_suffix, config) in enumerate(aq_questions_config.items()):
    question_key = f"aq_{key_suffix}"
    input_key = key_suffix + "_Score"
    user_inputs[input_key] = st.radio(
        f"{i+1}. {config['text']}",
        options=aq_options,
        key=question_key,
        horizontal=True,
        index=None
    )
    st.markdown("---")

# --- Section 2: Demographic and Background Information ---
st.subheader("Demographic & Background Information")
user_inputs['age'] = st.number_input("Age (years):", min_value=1, max_value=120, value=None, step=1, placeholder="Enter your age")
user_inputs['gender'] = st.radio("Gender:", options=["Female", "Male"], key="gender_radio", index=None, horizontal=True)
user_inputs['jaundice'] = st.radio("Born with Jaundice?", options=["No", "Yes"], key="jaundice_radio", index=None, horizontal=True)
user_inputs['austim'] = st.radio("Family member with Autism Spectrum Disorder?", options=["No", "Yes"], key="autism_radio", index=None, horizontal=True)
user_inputs['used_app_before'] = st.radio("Used a screening app like this before?", options=["No", "Yes"], key="used_app_radio", index=None, horizontal=True)

placeholder_text = "-- Select --"
ethnicity_options = [placeholder_text] + list(ethnicity_map.keys())
relation_options = [placeholder_text] + list(relation_map.keys())
user_inputs['ethnicity'] = st.selectbox("Ethnicity:", options=ethnicity_options, index=0)
user_inputs['relation'] = st.selectbox("Your relation to the person being screened:", options=relation_options, index=0)

# Modify the region selection part
country_options = [placeholder_text] + sorted(list(region_map.keys()))
selected_country = st.selectbox("Country of Residence:", options=country_options, index=0)

# --- Submit Button and Prediction Logic ---
st.markdown("---")
submit_button = st.button("Get Screening Result")

if submit_button:
    # --- Input Validation ---
    missing_fields = []
    for i in range(1, 11):
        key = f"A{i}_Score"
        if user_inputs[key] is None: missing_fields.append(f"Question {i}")
    if user_inputs['age'] is None: missing_fields.append("Age")
    if user_inputs['gender'] is None: missing_fields.append("Gender")
    if user_inputs['jaundice'] is None: missing_fields.append("Jaundice")
    if user_inputs['austim'] is None: missing_fields.append("Autism in Family") 
    if user_inputs['used_app_before'] is None: missing_fields.append("Used App Before")
    if user_inputs['ethnicity'] == placeholder_text: missing_fields.append("Ethnicity")
    if user_inputs['relation'] == placeholder_text: missing_fields.append("Relation")
    if selected_country == placeholder_text: missing_fields.append("Country")

    if missing_fields:
        st.error(f"Please fill in all fields. Missing: {', '.join(missing_fields)}")
    else:
        try:
            # --- Process Inputs for Model ---
            a_scores = []
            agree_responses = ["Slightly Agree", "Definitely Agree"]
            disagree_responses = ["Slightly Disagree", "Definitely Disagree"]

            for i in range(1, 11):
                input_key = f"A{i}_Score"
                config_key = f"A{i}"
                config = aq_questions_config[config_key]
                value = user_inputs[input_key]
                score = 0
                if config['scoring'] == 'forward':
                    if value in agree_responses: score = 1
                elif config['scoring'] == 'reverse':
                    if value in disagree_responses: score = 1
                a_scores.append(score)

            result_sum = sum(a_scores) # This corresponds to the 'result' feature

            age = float(user_inputs['age'])
            gender = 1 if user_inputs['gender'] == 'Male' else 0
            jaundice = 1 if user_inputs['jaundice'] == 'Yes' else 0
            autism_in_family = 1 if user_inputs['austim'] == 'Yes' else 0
            used_app_before = 1 if user_inputs['used_app_before'] == 'Yes' else 0

            # Get mapped float values for dropdowns (these correspond to the '_freq' features)
            ethnicity_val = ethnicity_map[user_inputs['ethnicity']]
            relation_val = relation_map[user_inputs['relation']]
            
            # Update how region_val is calculated
            if selected_country != placeholder_text:
                region = region_map[selected_country]
                region_val = region_frequencies[region]

            # --- Create Feature List (ORDER MUST MATCH EXPECTED_FEATURE_NAMES) ---
            features_in_order = [
                *a_scores,          
                age,                
                gender,             
                jaundice,           
                autism_in_family,   
                used_app_before,    
                result_sum,         
                ethnicity_val,      
                relation_val,       
                region_val          
            ]

            # --- Create Pandas DataFrame ---
            if len(features_in_order) != len(EXPECTED_FEATURE_NAMES):
                st.error(f"Critical Error: Feature count mismatch. Generated {len(features_in_order)}, expected {len(EXPECTED_FEATURE_NAMES)}. Check code.")
                print("Generated Features:", features_in_order)
                print("Expected Names:", EXPECTED_FEATURE_NAMES)
                st.stop()

            # Create the DataFrame with one row of data and the correct column names
            input_df = pd.DataFrame([features_in_order], columns=EXPECTED_FEATURE_NAMES)
            

            # --- Prediction (using DataFrame) ---
            prediction = model.predict(input_df)
            predicted_class = prediction[0]

            # --- Confidence Score Alternatives ---
            confidence_metrics = {}
            
            if hasattr(model, 'decision_function'):
                try:
                    # Get distance from decision boundary
                    decision_scores = model.decision_function(input_df)
                    # Normalize the score to a 0-1 scale using sigmoid function
                    confidence = 1 / (1 + np.exp(-abs(decision_scores[0])))
                    confidence_metrics['Distance-Based Score'] = f"{confidence:.1%}"
                except Exception as dec_e:
                    print(f"Error calculating decision function: {dec_e}")
                    confidence_metrics['Distance-Based Score'] = "Calculation Error"
            
            if hasattr(model, 'predict_proba'):
                try:
                    probabilities = model.predict_proba(input_df)
                    prob_confidence = probabilities[0][predicted_class]
                    confidence_metrics['Probability Score'] = f"{prob_confidence:.1%}"
                except Exception as proba_e:
                    print(f"Error calculating probability: {proba_e}")
                    confidence_metrics['Probability Score'] = "Calculation Error"

            # --- Display Results with Multiple Metrics ---
            st.subheader("Screening Outcome")
            if predicted_class == 1:
                st.warning("Result: Potential indicators of ASD detected based on screening answers.")
            else:
                st.success("Result: Does not show significant indicators of ASD based on this screening.")

            # Display available confidence metrics
            st.subheader("Model Confidence Metrics")
            for metric_name, value in confidence_metrics.items():
                st.info(f"{metric_name}: {value}")
            
            if not confidence_metrics:
                st.info("No confidence metrics available for this model type")

            st.caption("Disclaimer: This is a preliminary screening tool based on a machine learning model and does not substitute for a professional diagnosis. Consult with a qualified healthcare provider for any health concerns.")

        except Exception as e:
            st.error(f"An unexpected error occurred during prediction: {e}")
            print(f"Prediction Error details: {e}")