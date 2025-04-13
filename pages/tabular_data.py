
import streamlit as st
import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split,StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix
import gspread
from google.oauth2.service_account import Credentials

st.markdown(
    """
    <style>
        [data-testid="stSidebar"] {display: none;}
        [data-testid="stSidebarNavToggle"] {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True,
)
if st.button("🏠 Home"):
     st.switch_page("Landing_page.py")

# Constants
MODEL_PATH = os.path.join(os.path.dirname(__file__), "catboost_model.cbm")
SERVICE_ACCOUNT_FILE = "regal-station-452514-t8-42ab438bf0cc.json"
TRAINING_DATA_SHEET_ID = "1f11K9QkJF3w2Qk3xpbEtV7YS9eHmsMsCNAKOTwCuF2Q"
PREDICTION_SHEET_ID = "1Usg8iMlBAwMxFf9y-60Gg5gDE30rgH1Vh4_WsHAX0Es"
SCOPES = ["https://www.googleapis.com/auth/spreadsheets"]

print(MODEL_PATH)
@st.cache_resource
def connect_to_sheets(sheet_id):
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

def load_data(sheet):
    if sheet:
        data = sheet.get_all_records()
        return pd.DataFrame(data)
    return None

training_data = load_data(training_data_sheet)
prediction_data = load_data(prediction_sheet)

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

if training_data is not None and not training_data.empty:
    X, y, important_features = preprocess_data(training_data)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42, stratify=y)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    if prediction_data is not None and not prediction_data.empty:
        X_test, y_test, _ = preprocess_data(prediction_data)
        X_test = X_test[X_train.columns]
        X_test_scaled = scaler.transform(X_test)
    else:
        X_test_scaled, y_test = None, None
    
    @st.cache_resource
    def train_model():
        model = CatBoostClassifier(
            class_weights=[5, 1],
            random_strength=1.61,
            border_count=256,  
            iterations=1000,
            learning_rate=0.03,
            depth=8,
            l2_leaf_reg=2,
            loss_function='Logloss',
            eval_metric='Accuracy',
            random_seed=42,
            verbose=100
        )
        model.fit(X_train_scaled, y_train, eval_set=(X_val_scaled, y_val), early_stopping_rounds=50)

        cross_val = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cross_val_scores = cross_val_score(model, X_train_scaled, y_train, cv=cross_val, scoring='accuracy')

        
        train_predictions = model.predict(X_train_scaled)
        val_predictions = model.predict(X_val_scaled)
        train_conf_matrix = confusion_matrix(y_train, train_predictions)
        val_conf_matrix = confusion_matrix(y_val, val_predictions)
        train_accuracy = accuracy_score(y_train, train_predictions)
        val_accuracy = accuracy_score(y_val, val_predictions)
        
        if X_test_scaled is not None and y_test is not None:
            test_predictions = model.predict(X_test_scaled)
            test_conf_matrix = confusion_matrix(y_test, test_predictions)
            test_accuracy = accuracy_score(y_test, test_predictions)
        else:
            test_conf_matrix, test_accuracy = None, None
        
        model.save_model(MODEL_PATH)
        return model, train_conf_matrix, val_conf_matrix, test_conf_matrix, train_accuracy, val_accuracy, test_accuracy, cross_val_scores.mean()

    
    model, train_conf_matrix, val_conf_matrix, test_conf_matrix, train_accuracy, val_accuracy, test_accuracy, cross_val_accuracy = train_model()

    
    st.title("🩺 Disease Diagnosis Prediction")
    st.write(f"Training Accuracy: {train_accuracy:.4f}")
    st.write(f"Validation Accuracy: {val_accuracy:.4f}")
    st.write(f"Cross-Validation Accuracy: {cross_val_accuracy:.4f}")
    if test_accuracy is not None:
        st.write(f"Testing Accuracy: {test_accuracy:.4f}")

    show_matrices = st.button("📊 Show Confusion Matrices and Accuracy Calculation")
    hide_matrices = st.button("Hide Confusion Matrices")

    if show_matrices:
        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            st.subheader("🔍 Training Set Confusion Matrix")
            fig, ax = plt.subplots(figsize=(2, 2))
            sns.heatmap(train_conf_matrix, annot=True, fmt="d", cmap="Blues", ax=ax, xticklabels=["No", "Yes"], yticklabels=["No", "Yes"])
            st.pyplot(fig)
            st.write("""
            **Formula for Accuracy:**
            $$
            Accuracy = \\frac{TP + TN}{TP + TN + FP + FN}
            $$
            **Calculation:**
            """)
            TP, FN, FP, TN = train_conf_matrix.ravel()
            train_acc_calc = (TP + TN) / (TP + TN + FP + FN)
            st.write(f"Accuracy = ({TP} + {TN}) / ({TP} + {TN} + {FP} + {FN}) = **{train_acc_calc:.4f}**")

        with col3:
            st.subheader("🔍 Validation Set Confusion Matrix")
            fig, ax = plt.subplots(figsize=(2, 2))
            sns.heatmap(val_conf_matrix, annot=True, fmt="d", cmap="Oranges", ax=ax, xticklabels=["No", "Yes"], yticklabels=["No", "Yes"])
            st.pyplot(fig)
            st.write("""
            **Formula for Accuracy:**
            $$
            Accuracy = \\frac{TP + TN}{TP + TN + FP + FN}
            $$
            **Calculation:**
            """)
            TP, FN, FP, TN = val_conf_matrix.ravel()
            val_acc_calc = (TP + TN) / (TP + TN + FP + FN)
            st.write(f"Accuracy = ({TP} + {TN}) / ({TP} + {TN} + {FP} + {FN}) = **{val_acc_calc:.4f}**")

        with col5:
            st.subheader("🔍 Testing Set Confusion Matrix")
            fig, ax = plt.subplots(figsize=(2, 2))
            sns.heatmap(test_conf_matrix, annot=True, fmt="d", cmap="Greens", ax=ax, xticklabels=["No", "Yes"], yticklabels=["No", "Yes"])
            st.pyplot(fig)
            st.write("""
            **Formula for Accuracy:**
            $$
            Accuracy = \\frac{TP + TN}{TP + TN + FP + FN}
            $$
            **Calculation:**
            """)
            TP, FN, FP, TN = test_conf_matrix.ravel()
            test_acc_calc = (TP + TN) / (TP + TN + FP + FN)
            st.write(f"Accuracy = ({TP} + {TN}) / ({TP} + {TN} + {FP} + {FN}) = **{test_acc_calc:.4f}**")
        
        st.markdown("""
        **🔹 Interpretation:**
        - **TP (True Positive)** = Model correctly predicted Parkinson’s.
        - **TN (True Negative)** = Model correctly predicted no Parkinson’s.
        - **FP (False Positive)** = Model falsely predicted Parkinson’s.
        - **FN (False Negative)** = Model falsely predicted no Parkinson’s.
        - Accuracy formula and calculation shown below each confusion matrix.
        """)

    # Prediction part
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
            input_df = input_df[X_train.columns]  # Ensure correct column order
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


if st.button("📈 Show Accuracy Graph"):
    fig, ax = plt.subplots()
    categories = ['Training', 'Validation', 'Testing']
    accuracies = [train_accuracy, val_accuracy, test_accuracy]

    bars = sns.barplot(x=categories, y=accuracies, palette="Blues", ax=ax)
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1)

    # Add accuracy percentage on top of each bar
    for bar, accuracy in zip(bars.patches, accuracies):
        ax.annotate(f'{accuracy:.2%}',  # Format as percentage
                    (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    ha='center', va='center',
                    size=10, xytext=(0, -20),
                    textcoords='offset points')

    st.pyplot(fig)
