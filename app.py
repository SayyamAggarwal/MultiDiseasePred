import streamlit as st  # type: ignore
import pickle
import numpy as np
from streamlit_option_menu import option_menu  # type: ignore

# === Load Models and Scalers ===
diabetes_model = pickle.load(open("notebooks/diabetes_disease/models/diabetes_model.pkl", 'rb'))
heart_model = pickle.load(open("notebooks/heart_disease/models/heart_model.pkl", 'rb'))
parkinsons_model = pickle.load(open("notebooks/parkinsons_disease/models/parkinsons_model.pkl", 'rb'))

diabetes_scaler = pickle.load(open("notebooks/diabetes_disease/models/diabetes_scaler.pkl", 'rb'))
heart_scaler = pickle.load(open("notebooks/heart_disease/models/heart_scaler.pkl", 'rb'))
parkinsons_scaler = pickle.load(open("notebooks/parkinsons_disease/models/parkinsons_scaler.pkl", 'rb'))

# === Sidebar Navigation ===
with st.sidebar:
    selected = option_menu("Multiple Disease Prediction System",
                           ['Diabetes Prediction', 'Heart Prediction', 'Parkinsons Prediction'],
                           menu_icon="hospital-fill",
                           icons=['activity', 'heart', 'person'],
                           default_index=0)

# === Diabetes Prediction ===
if selected == "Diabetes Prediction":
    st.title("Diabetes Prediction Using Machine Learning")
    with st.expander("📋 Show Expected Input Columns for Diabetes Prediction"):
        st.markdown("""
        - **Glucose**: Blood glucose level  
        - **BloodPressure**: Diastolic blood pressure (mm Hg)  
        - **SkinThickness**: Triceps skin fold thickness (mm)  
        - **Insulin**: 2-Hour serum insulin (mu U/ml)  
        - **BMI**: Body Mass Index (weight in kg / height in m²)  
        - **DiabetesPedigreeFunction**: Diabetes heritage measure  
        - **Age**: Age in years
        """)


    # Input fields
    Glucose = st.text_input('Glucose Level')
    BloodPressure = st.text_input('Blood Pressure value')
    SkinThickness = st.text_input('Skin Thickness value')
    Insulin = st.text_input('Insulin Level')
    BMI = st.text_input('BMI value')
    DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function value')
    Age = st.text_input('Age of the Person')

    diab_diag = ""

    if st.button("Diabetes Test Result"):
        try:
            input_data = np.array([[float(Glucose), float(BloodPressure), float(SkinThickness),
                                    float(Insulin), float(BMI), float(DiabetesPedigreeFunction), float(Age)]])
            scaled_input = diabetes_scaler.transform(input_data)
            diab_prediction = diabetes_model.predict(scaled_input)

            if diab_prediction[0] == 1:
                diab_diag = "⚠️ The person is likely diabetic."
            else:
                diab_diag = "✅ The person is not diabetic."
            st.success(diab_diag)
        except ValueError:
            st.error("Please enter valid numerical values.")

# === Heart Disease Prediction ===
elif selected == "Heart Prediction":
    st.title("Heart Disease Prediction Using Machine Learning")
    with st.expander("📋 Show Expected Input Columns for Heart Disease Prediction"):
        st.markdown("""
        - **age**: Age in years  
        - **sex**: 0 = female, 1 = male  
        - **cp**: Chest pain type (0–3)  
        - **trestbps**: Resting blood pressure (mm Hg)  
        - **chol**: Serum cholesterol in mg/dl  
        - **fbs**: Fasting blood sugar > 120 mg/dl (1 = true; 0 = false)  
        - **restecg**: Resting electrocardiographic results (0–2)  
        - **thalach**: Maximum heart rate achieved  
        - **exang**: Exercise-induced angina (1 = yes; 0 = no)  
        - **oldpeak**: ST depression induced by exercise  
        - **slope**: Slope of the peak exercise ST segment (0–2)  
        - **ca**: Number of major vessels (0–4) colored by fluoroscopy  
        - **thal**: 0 = normal; 1 = fixed defect; 2 = reversible defect
        """)


    # Input fields
    age = st.text_input('Age')
    sex = st.text_input('Sex (0 = Female, 1 = Male)')
    cp = st.text_input('Chest Pain Type (0-3)')
    trestbps = st.text_input('Resting Blood Pressure')
    chol = st.text_input('Cholesterol')
    fbs = st.text_input('Fasting Blood Sugar (1=True, 0=False)')
    restecg = st.text_input('Resting ECG (0-2)')
    thalach = st.text_input('Max Heart Rate')
    exang = st.text_input('Exercise Induced Angina (1/0)')
    oldpeak = st.text_input('Oldpeak')
    slope = st.text_input('Slope (0-2)')
    ca = st.text_input('CA (0-4)')
    thal = st.text_input('Thal (0,1,2)')

    heart_diag = ""

    if st.button("Heart Disease Test Result"):
        try:
            input_data = np.array([[float(age), float(sex), float(cp), float(trestbps), float(chol), float(fbs),
                                    float(restecg), float(thalach), float(exang), float(oldpeak), float(slope),
                                    float(ca), float(thal)]])
            scaled_input = heart_scaler.transform(input_data)
            heart_prediction = heart_model.predict(scaled_input)

            if heart_prediction[0] == 1:
                heart_diag = "⚠️ The person has a high chance of heart disease."
            else:
                heart_diag = "✅ The person likely does not have heart disease."
            st.success(heart_diag)
        except ValueError:
            st.error("Please enter valid numerical values.")

# === Parkinson’s Prediction ===
elif selected == "Parkinsons Prediction":
    st.title("Parkinson's Disease Prediction Using Machine Learning")
    with st.expander("📋 Show Expected Input Columns for Parkinson's Prediction"):
        st.markdown("""
        - **MDVP:Fo(Hz)**  
        - **MDVP:Fhi(Hz)**  
        - **MDVP:Flo(Hz)**  
        - **MDVP:Jitter(%)**  
        - **MDVP:Jitter(Abs)**  
        - **MDVP:RAP**  
        - **MDVP:PPQ**  
        - **Jitter:DDP**  
        - **MDVP:Shimmer**  
        - **MDVP:Shimmer(dB)**  
        - **Shimmer:APQ3**  
        - **Shimmer:APQ5**  
        - **MDVP:APQ**  
        - **Shimmer:DDA**  
        - **NHR**  
        - **HNR**  
        - **RPDE**  
        - **DFA**  
        - **spread1**  
        - **spread2**  
        - **D2**  
        - **PPE**
        """)


    # Input fields
    fields = ['MDVP:Fo(Hz)', 'MDVP:Fhi(Hz)', 'MDVP:Flo(Hz)', 'MDVP:Jitter(%)', 'MDVP:Jitter(Abs)',
              'MDVP:RAP', 'MDVP:PPQ', 'Jitter:DDP', 'MDVP:Shimmer', 'MDVP:Shimmer(dB)',
              'Shimmer:APQ3', 'Shimmer:APQ5', 'MDVP:APQ', 'Shimmer:DDA', 'NHR', 'HNR', 'RPDE',
              'DFA', 'spread1', 'spread2', 'D2', 'PPE']
    inputs = []

    for field in fields:
        inputs.append(st.text_input(field))

    park_diag = ""

    if st.button("Parkinson's Test Result"):
        try:
            input_data = np.array([[float(i) for i in inputs]])
            scaled_input = parkinsons_scaler.transform(input_data)
            park_prediction = parkinsons_model.predict(scaled_input)

            if park_prediction[0] == 1:
                park_diag = "⚠️ The person is likely to have Parkinson's disease."
            else:
                park_diag = "✅ The person likely does not have Parkinson's disease."
            st.success(park_diag)
        except ValueError:
            st.error("Please enter all values as numbers.")
