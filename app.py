import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score
import base64
from io import BytesIO

# Set page configuration
st.set_page_config(page_title="Titanic Explorer & Survival Predictor", layout="wide", initial_sidebar_state="expanded")

# Custom CSS for Titanic-themed styling
st.markdown("""
    <style>
    .main { background-color: #f0f6ff; }
    .stButton>button { background-color: #003087; color: white; border-radius: 5px; }
    .stTabs { background-color: #e6f3ff; padding: 10px; border-radius: 10px; }
    .stSidebar { background-color: #003087; color: white; }
    .stSidebar .stSelectbox, .stSidebar .stSlider { background-color: white; color: black; }
    h1, h2, h3 { color: #003087; }
    .tooltip {
        position: relative;
        display: inline-block;
        cursor: pointer;
    }
    .tooltip .tooltiptext {
        visibility: hidden;
        width: 200px;
        background-color: #555;
        color: #fff;
        text-align: center;
        border-radius: 6px;
        padding: 5px;
        position: absolute;
        z-index: 1;
        bottom: 125%;
        left: 50%;
        margin-left: -100px;
        opacity: 0;
        transition: opacity 0.3s;
    }
    .tooltip:hover .tooltiptext {
        visibility: visible;
        opacity: 1;
    }
             /* Prediction tab specific styling */
    .prediction-tab { 
        background-color: #e6f3ff; 
        padding: 25px; 
        border-radius: 12px; 
        border: 2px solid #003087; 
        box-shadow: 0 4px 12px rgba(0,0,0,0.15); 
        margin-bottom: 20px; 
    }
    .prediction-form { 
        background-color: white; 
        padding: 20px; 
        border-radius: 10px; 
        border: 2px solid #003087; 
        box-shadow: 0 2px 8px rgba(0,0,0,0.1); 
    }
    .prediction-output-survived { 
        background-color: #c3e6cb; /* Lighter green for better contrast */
        color: #155724; 
        padding: 15px; 
        border-radius: 8px; 
        border: 1px solid #155724; 
        font-size: 18px; 
        font-weight: bold; 
        text-align: center; 
        margin-top: 10px; 
    }
    .prediction-output-not-survived { 
        background-color: blue; /* Lighter red for better contrast */
        color: #721c24; 
        padding: 15px; 
        border-radius: 8px; 
        border: 1px solid #721c24; 
        font-size: 18px; 
        font-weight: bold; 
        text-align: center; 
        margin-top: 10px; 
    }
    </style>
""", unsafe_allow_html=True)

# Load data and models
@st.cache_data
def load_data_and_models():
    df = pd.read_csv('titanic_processed.csv')
    rf_model = joblib.load('titanic_rf_model.pkl')
    le_sex = joblib.load('le_sex.pkl')
    le_embarked = joblib.load('le_embarked.pkl')
    return df, rf_model, le_sex, le_embarked

df, rf_model, le_sex, le_embarked = load_data_and_models()

# Sidebar for navigation and filters
st.sidebar.title("Titanic Explorer")
st.sidebar.markdown("Navigate through the app or filter visualizations.")
app_mode = st.sidebar.selectbox("Choose Section", ["Home", "EDA", "Prediction", "Model Insights", "Dataset"])

# EDA filters
st.sidebar.subheader("EDA Filters")
pclass_filter = st.sidebar.multiselect("Passenger Class", options=[1, 2, 3], default=[1, 2, 3])
sex_filter = st.sidebar.multiselect("Sex", options=["male", "female"], default=["male", "female"])

# Filter dataset based on user selection
filtered_df = df[df['Pclass'].isin(pclass_filter) & df['Sex'].isin(sex_filter)]

# Home tab
if app_mode == "Home":
    st.title("Titanic Explorer & Survival Predictor")
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/f/fd/RMS_Titanic_3.jpg/1200px-RMS_Titanic_3.jpg", caption="RMS Titanic", width=400)
    st.markdown("""
        Welcome to the **Titanic Explorer & Survival Predictor**! This app provides:
        - **Exploratory Data Analysis (EDA)**: Visualize passenger demographics and survival patterns.
        - **Survival Prediction**: Predict survival for individual passengers or batch inputs.
        - **Model Insights**: View Random Forest model performance and feature importance.
        - **Dataset Access**: Download the processed Titanic dataset.
        Use the sidebar to navigate and filter visualizations.
    """)

# EDA tab
elif app_mode == "EDA":
    st.header("Exploratory Data Analysis")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Age Distribution")
        fig1, ax1 = plt.subplots()
        sns.histplot(filtered_df['Age'], bins=30, kde=True, ax=ax1)
        ax1.set_title('Age Distribution of Passengers')
        ax1.set_xlabel('Age')
        ax1.set_ylabel('Count')
        st.pyplot(fig1)

        st.subheader("Fare Distribution")
        fig2, ax2 = plt.subplots()
        sns.histplot(filtered_df['Fare'], bins=30, kde=True, ax=ax2)
        ax2.set_title('Fare Distribution')
        ax2.set_xlabel('Fare')
        ax2.set_ylabel('Count')
        st.pyplot(fig2)

    with col2:
        st.subheader("Survival by Pclass and Sex")
        fig3 = sns.catplot(x='Pclass', hue='Survived', col='Sex', kind='count', data=filtered_df, height=4, aspect=1)
        st.pyplot(fig3)

        st.subheader("Survival by Embarked")
        fig4, ax4 = plt.subplots()
        sns.countplot(x='Embarked', hue='Survived', data=filtered_df, ax=ax4)
        ax4.set_title('Survival Count by Embarked Location')
        ax4.set_xlabel('Embarked')
        ax4.set_ylabel('Count')
        st.pyplot(fig4)

# Prediction tab
elif app_mode == "Prediction":
    st.header("Predict Passenger Survival")
    st.markdown("Enter passenger details or upload a CSV for batch predictions.")

    # Tabs for single vs batch prediction
    pred_tab1, pred_tab2 = st.tabs(["Single Prediction", "Batch Prediction"])

    with pred_tab1:
        with st.form("single_prediction_form"):
            st.markdown('<div class="tooltip">Passenger Details<span class="tooltiptext">Enter details to predict survival probability.</span></div>', unsafe_allow_html=True)
            pclass = st.selectbox("Passenger Class", [1, 2, 3], help="1 = First Class, 2 = Second Class, 3 = Third Class")
            sex = st.selectbox("Sex", ["male", "female"])
            age = st.slider("Age", 0, 100, 30, help="Age of the passenger")
            sibsp = st.number_input("Siblings/Spouses Aboard", min_value=0, max_value=10, value=0, help="Number of siblings or spouses")
            parch = st.number_input("Parents/Children Aboard", min_value=0, max_value=10, value=0, help="Number of parents or children")
            fare = st.number_input("Fare", min_value=0.0, max_value=600.0, value=30.0, help="Ticket fare in pounds")
            embarked = st.selectbox("Embarked", ["S", "C", "Q"], help="S = Southampton, C = Cherbourg, Q = Queenstown")
            submit = st.form_submit_button("Predict")

            if submit:
                with st.spinner("Predicting..."):
                    input_data = pd.DataFrame({
                        'Pclass': [pclass],
                        'Sex': [le_sex.transform([sex])[0]],
                        'Age': [age],
                        'SibSp': [sibsp],
                        'Parch': [parch],
                        'Fare': [fare],
                        'Embarked': [le_embarked.transform([embarked])[0]]
                    })
                    prediction = rf_model.predict(input_data)[0]
                    probability = rf_model.predict_proba(input_data)[0][1]
                    st.success(f"**Prediction**: {'Survived' if prediction == 1 else 'Did not survive'}")
                    st.info(f"**Survival Probability**: {probability:.2%}")

    with pred_tab2:
        st.subheader("Batch Prediction")
        uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"], help="CSV must have columns: Pclass, Sex, Age, SibSp, Parch, Fare, Embarked")
        if uploaded_file:
            batch_df = pd.read_csv(uploaded_file)
            required_columns = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
            if all(col in batch_df.columns for col in required_columns):
                with st.spinner("Processing batch predictions..."):
                    progress_bar = st.progress(0)
                    batch_df['Sex'] = le_sex.transform(batch_df['Sex'])
                    batch_df['Embarked'] = le_embarked.transform(batch_df['Embarked'])
                    predictions = rf_model.predict(batch_df[required_columns])
                    probabilities = rf_model.predict_proba(batch_df[required_columns])[:, 1]
                    batch_df['Predicted_Survived'] = predictions
                    batch_df['Survival_Probability'] = probabilities
                    progress_bar.progress(100)
                    st.write("Batch Predictions:")
                    st.write(batch_df)
                    # Download button for batch results
                    csv = batch_df.to_csv(index=False)
                    b64 = base64.b64encode(csv.encode()).decode()
                    href = f'<a href="data:file/csv;base64,{b64}" download="batch_predictions.csv">Download Batch Predictions</a>'
                    st.markdown(href, unsafe_allow_html=True)
            else:
                st.error("CSV must contain columns: Pclass, Sex, Age, SibSp, Parch, Fare, Embarked")

# Model Insights tab
elif app_mode == "Model Insights":
    st.header("Model Insights")
    st.markdown("Explore the Random Forest model's performance and feature importance.")

    # Model performance
    st.subheader("Model Performance")
    X = df[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']].copy()
    X['Sex'] = le_sex.transform(X['Sex'])
    X['Embarked'] = le_embarked.transform(X['Embarked'])
    y = df['Survived']
    y_pred = rf_model.predict(X)
    accuracy = accuracy_score(y, y_pred)
    st.write(f"**Model Accuracy (on full dataset)**: {accuracy:.2%}")
    cm = confusion_matrix(y, y_pred)
    fig5, ax5 = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax5)
    ax5.set_title('Confusion Matrix')
    ax5.set_xlabel('Predicted')
    ax5.set_ylabel('Actual')
    st.pyplot(fig5)

    # Feature importance
    st.subheader("Feature Importance")
    feature_importance = pd.DataFrame({
        'Feature': ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked'],
        'Importance': rf_model.feature_importances_
    }).sort_values(by='Importance', ascending=False)
    fig6, ax6 = plt.subplots()
    sns.barplot(x='Importance', y='Feature', data=feature_importance, ax=ax6)
    ax6.set_title('Feature Importance in Random Forest Model')
    st.pyplot(fig6)

# Dataset tab
elif app_mode == "Dataset":
    st.header("Dataset Insights")
    st.markdown("View and download the processed Titanic dataset.")

    # Dataset preview
    st.subheader("Dataset Preview")
    st.write(filtered_df.head())

    # Summary statistics
    st.subheader("Summary Statistics")
    st.write(filtered_df.describe())

    # Download dataset
    st.subheader("Download Processed Dataset")
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="titanic_processed.csv">Download Processed Dataset</a>'
    st.markdown(href, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("Created by [Your Name] for Titanic EDA, Machine Learning, and Streamlit App Project")