import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import numpy as np

# Set page configuration
st.set_page_config(page_title="Titanic EDA and Survival Prediction", layout="wide")

# Load data and models
@st.cache_data
def load_data_and_models():
    df = pd.read_csv('titanic_processed.csv')
    rf_model = joblib.load('titanic_rf_model.pkl')
    le_sex = joblib.load('le_sex.pkl')
    le_embarked = joblib.load('le_embarked.pkl')
    return df, rf_model, le_sex, le_embarked

df, rf_model, le_sex, le_embarked = load_data_and_models()

# Title and introduction
st.title("Titanic Dataset: Exploratory Data Analysis and Survival Prediction")
st.write("This app presents an Exploratory Data Analysis (EDA) of the Titanic dataset and allows you to predict passenger survival using a Random Forest model.")

# EDA Section
st.header("Exploratory Data Analysis")

# Dataset preview
st.subheader("Dataset Preview")
st.write(df.head())

# Summary statistics
st.subheader("Summary Statistics")
st.write(df.describe())

# Visualizations
st.subheader("Key Visualizations")

# Age distribution
fig1, ax1 = plt.subplots()
sns.histplot(df['Age'], bins=30, kde=True, ax=ax1)
ax1.set_title('Age Distribution of Passengers')
ax1.set_xlabel('Age')
ax1.set_ylabel('Count')
st.pyplot(fig1)

# Survival by Pclass and Sex
fig2 = sns.catplot(x='Pclass', hue='Survived', col='Sex', kind='count', data=df, height=4, aspect=1)
st.pyplot(fig2)

# Survival Prediction Section
st.header("Predict Passenger Survival")
st.write("Enter passenger details to predict survival probability.")

# Input form
with st.form("prediction_form"):
    pclass = st.selectbox("Passenger Class", [1, 2, 3])
    sex = st.selectbox("Sex", ["male", "female"])
    age = st.slider("Age", 0, 100, 30)
    sibsp = st.number_input("Siblings/Spouses Aboard", min_value=0, max_value=10, value=0)
    parch = st.number_input("Parents/Children Aboard", min_value=0, max_value=10, value=0)
    fare = st.number_input("Fare", min_value=0.0, max_value=600.0, value=30.0)
    embarked = st.selectbox("Embarked", ["S", "C", "Q"])
    submit = st.form_submit_button("Predict")

    if submit:
        # Prepare input data
        input_data = pd.DataFrame({
            'Pclass': [pclass],
            'Sex': [le_sex.transform([sex])[0]],
            'Age': [age],
            'SibSp': [sibsp],
            'Parch': [parch],
            'Fare': [fare],
            'Embarked': [le_embarked.transform([embarked])[0]]
        })
        # Predict
        prediction = rf_model.predict(input_data)[0]
        probability = rf_model.predict_proba(input_data)[0][1]
        st.write(f"**Prediction**: {'Survived' if prediction == 1 else 'Did not survive'}")
        st.write(f"**Survival Probability**: {probability:.2%}")

# Footer
st.write("---")
st.write("Created by John Brown Ouma for Titanic EDA and Prediction Project")