import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder

# Load the trained model
model = joblib.load("best_model.pkl")
required_features = joblib.load('feature_names.pkl')

st.set_page_config(page_title="Employee Salary Classification", page_icon="💼", layout="centered")

st.title("💼🤖📊 AI Salary Sensei App")
st.markdown("Predict whether an employee earns >50K or ≤50K based on input features.")

# Sidebar inputs (these must match your training feature columns)
st.sidebar.header("Input Employee Details")

# ✨ Replace these fields with your dataset's actual input columns
age = st.sidebar.slider("Age", 18, 65, 30)
# education = st.sidebar.selectbox("Education Level", [
#     "Bachelors", "Masters", "PhD", "HS-grad", "Assoc", "Some-college"
# ])
educational_num = st.sidebar.slider("Education Level (Numerical)", 1, 16, 10)
fnlwgt = st.sidebar.number_input("Final Weight (fnlwgt)", value=100000, step=1000)
capital_gain = st.sidebar.number_input("Capital Gain", value=0, step=100)
capital_loss = st.sidebar.number_input("Capital Loss", value=0, step=100)
occupation = st.sidebar.selectbox("Job Role", [
    "Tech-support", "Craft-repair", "Other-service", "Sales",
    "Exec-managerial", "Prof-specialty", "Handlers-cleaners", "Machine-op-inspct",
    "Adm-clerical", "Farming-fishing", "Transport-moving", "Priv-house-serv",
    "Protective-serv", "Armed-Forces"
])
hours_per_week = st.sidebar.slider("Hours per week", 1, 80, 40)
gender = st.sidebar.radio("Gender", ["Male", "Female"])

# experience = st.sidebar.slider("Years of Experience", 0, 40, 5)

# Build input DataFrame (⚠️ must match preprocessing of your training data)
input_df = pd.DataFrame({
    'age': [age],
    'educational-num': [educational_num],
    'fnlwgt': [fnlwgt],
    'capital-gain': [capital_gain],
    'capital-loss': [capital_loss],
    'hours-per-week': [hours_per_week],
    'gender': [gender],
    'occupation': [occupation]
})
encoder = LabelEncoder()

input_df['occupation'] = encoder.fit_transform(input_df['occupation'])

input_df['gender'] = encoder.fit_transform(input_df['gender'])


for col in required_features:
    if col not in input_df.columns:
        input_df[col] = 0

input_df = input_df[required_features]

st.write("### 📥 Input Data")
st.write(input_df)

# Predict button
if st.button("Predict Salary Class"):
    prediction = model.predict(input_df)
    st.success(f"🔮✅ Prediction: {prediction[0]}")

# Batch prediction
st.markdown("---")
st.markdown("#### 📂🔁 Batch Prediction")
uploaded_file = st.file_uploader("Upload a CSV file for batch prediction", type="csv")

if uploaded_file is not None:
    batch_data = pd.read_csv(uploaded_file)

    for col in required_features:
        if col not in batch_data.columns:
            batch_data[col] = 0
    batch_data = batch_data[required_features]
    
    st.write("Uploaded data preview:", batch_data.head())
    batch_preds = model.predict(batch_data)
    batch_data['PredictedClass'] = batch_preds
    st.write("✅ Predictions:")
    st.write(batch_data.head())
    csv = batch_data.to_csv(index=False).encode('utf-8')
    st.download_button("Download Predictions CSV", csv, file_name='predicted_classes.csv', mime='text/csv')

