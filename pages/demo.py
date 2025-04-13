import streamlit as st
import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import gspread
from google.oauth2.service_account import Credentials

st.markdown(
    """
    <style>
        [data-testid="stSidebar"] {display: none;} /* Hide sidebar */
        [data-testid="stSidebarNavToggle"] {
            visibility: hidden;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

if st.button("🏠 Home"):
    st.switch_page("Landing_page.py")  # Ensure the filename matches your landing page

# Constants
MODEL_PATH = os.path.join(os.path.dirname(__file__), "catboost_model.cbm")
SERVICE_ACCOUNT_FILE = "regal-station-452514-t8-42ab438bf0cc.json"
TRAINING_DATA_SHEET_ID = "1f11K9QkJF3w2Qk3xpbEtV7YS9eHmsMsCNAKOTwCuF2Q"
PREDICTION_SHEET_ID = "1Usg8iMlBAwMxFf9y-60Gg5gDE30rgH1Vh4_WsHAX0Es"
SCOPES = ["https://www.googleapis.com/auth/spreadsheets"]

@st.cache_resource
def connect_to_sheets(sheet_id):
    """Establish connection to Google Sheets."""
    try:
        creds = Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE, scopes=SCOPES)
        client = gspread.authorize(creds)
        sheet = client.open_by_key(sheet_id).sheet1
        return sheet
    except Exception as e:
        st.error(f"⚠ Error connecting to Google Sheets: {e}")
        return None

training_data_sheet = connect_to_sheets(TRAINING_DATA_SHEET_ID)
prediction_sheet = connect_to_sheets(PREDICTION_SHEET_ID)

# Load training data
def load_training_data():
    if training_data_sheet:
        data = training_data_sheet.get_all_records()
        return pd.DataFrame(data)
    return None

training_data = load_training_data()

if training_data is None or training_data.empty:
    st.error("⚠ No training data available. Please check your Google Sheets connection.")
else:
    # Preprocessing
    def preprocess_data(df):
        boolean_columns = ['Tremor', 'Rigidity', 'Bradykinesia', 'PosturalInstability', 'Depression', 'Diabetes', 'Diagnosis']
        for col in boolean_columns:
            df[col] = df[col].astype(int)
        df['Age'] = df['Age'].astype(int)
        label_encoder = LabelEncoder()
        df['Diagnosis'] = label_encoder.fit_transform(df['Diagnosis'])
        numeric_df = df.select_dtypes(include=['number'])
        correlation = numeric_df.corr()['Diagnosis'].abs().sort_values(ascending=False)
        important_features = correlation.index[1:11].tolist()
        X = df[important_features].copy()
        y = df['Diagnosis']
        return X, y, important_features

    X, y, important_features = preprocess_data(training_data)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    @st.cache_resource
    def train_model():
        model = CatBoostClassifier(
            iterations=1000,
            learning_rate=0.03,
            depth=8,
            l2_leaf_reg=3,
            loss_function='Logloss',
            eval_metric='Accuracy',
            random_seed=42,
            verbose=100
        )
        model.fit(X_train_scaled, y_train, eval_set=(X_test_scaled, y_test), early_stopping_rounds=50)
        
        # Cross-validation accuracy
        cross_val = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cross_val_scores = cross_val_score(model, X_train_scaled, y_train, cv=cross_val, scoring='accuracy')

        # Test accuracy
        test_predictions = model.predict(X_test_scaled)
        test_accuracy = accuracy_score(y_test, test_predictions)
        
        # Confusion matrix
        conf_matrix = confusion_matrix(y_test, test_predictions)

        # Display results
        st.write(f"Cross-Validation Accuracy: {cross_val_scores.mean():.4f}")
        st.write(f"Test Accuracy: {test_accuracy:.4f}")

        # Plot confusion matrix
        fig, ax = plt.subplots()
        sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=["No", "Yes"], yticklabels=["No", "Yes"])
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.title("Confusion Matrix")
        st.pyplot(fig)

        model.save_model(MODEL_PATH)
        return model

    model = train_model()

    # Streamlit UI
    st.title("🩺 Disease Diagnosis Prediction")
    st.write("Provide the required inputs to predict the diagnosis:")

    # Description of boolean columns
    st.write("""
    **For the boolean columns (Tremor, Rigidity, Bradykinesia, Postural Instability, Depression, Diabetes, Diagnosis):**
    - Enter `0` if absent.
    - Enter `1` if present.
    """)

    numeric_responses = []
    for feature in important_features:
        if feature in ['Tremor', 'Rigidity', 'Bradykinesia', 'PosturalInstability', 'Depression', 'Diabetes', 'Diagnosis']:
            value = st.radio(f"{feature} (0 = Absent, 1 = Present):", [0, 1])
        elif feature == 'Age':
            value = st.number_input(f"{feature} (integer value):", min_value=0, step=1, format="%d")
        else:
            value = st.number_input(f"{feature} (float value):", min_value=0.0, step=0.000001, format="%.6f")
        numeric_responses.append(value)

    if st.button("🔍 Predict Diagnosis"):
        if model is None:
            st.error("⚠ Model not loaded.")
        else:
            input_df = pd.DataFrame([numeric_responses], columns=important_features)
            input_df_scaled = scaler.transform(input_df)
            prediction = model.predict(input_df_scaled)[0]
            output = "Yes" if prediction == 1 else "No"

            if prediction_sheet:
                try:
                    prediction_sheet.append_row(numeric_responses + [1 if output == "Yes" else 0])
                    st.success(f"✅ Prediction: {output} (Saved to Google Sheets)")
                except Exception as e:
                    st.error(f"⚠ Failed to save data: {e}")

            st.subheader(f"🩺 Prediction: {output}")

    st.markdown("""
    **Explanation:**
    - The model predicts whether the individual has Parkinson's disease based on the provided symptoms and medical history.
    - The `Diagnosis` column is the output prediction: `1` means "Yes" (Parkinson's detected), and `0` means "No" (No Parkinson's detected).
    """)
