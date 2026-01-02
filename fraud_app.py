import streamlit as st
import numpy as np
import pandas as pd
import joblib
import pickle
import time

from streamlit_cookies_manager import EncryptedCookieManager
import matplotlib.pyplot as plt
import plotly.express as px


# -----------------------------
# Authentication setup
# -----------------------------
cookies = EncryptedCookieManager(
    prefix="hr_app",
    password=st.secrets["cookie"]["password"]
)
if not cookies.ready():
    st.stop()

SESSION_TIMEOUT = 60 * 30  # 30 minutes

def login():
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False

    # Check cookie
    if cookies.get("login_time"):
        login_time = float(cookies.get("login_time"))
        if time.time() - login_time < SESSION_TIMEOUT:
            st.session_state.logged_in = True

    if st.session_state.logged_in:
        return True

    st.title("🔐 Secure Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        users = dict(st.secrets.get("auth", {}).get("users", {}))
        if username in users and password == users[username]:
            st.session_state.logged_in = True
            cookies["login_time"] = str(time.time())
            cookies.save()
            st.success("Login successful. Refreshing...")
            st.rerun()
        else:
            st.error("Invalid credentials")
    return False

if not login():
    st.stop()


# Example logout button
if st.sidebar.button("Logout"):
    st.session_state.logged_in = False
    cookies["login_time"] = ""  # clear cookie
    cookies.save()
    st.rerun()


# -------------------------------
# Load Trained Model
# -------------------------------
model = joblib.load("GradientBoostingClassifier.pkl")

# # Optional backup with pickle
# with open("pickle_GradientBoostingClassifier.pkl", "rb") as f:
#     model_pickle = pickle.load(f)

# -------------------------------
# Fraud Detection Function
# -------------------------------
def fraud_detection(fraud_data):
    prediction = model.predict(fraud_data)
    probability = np.round(model.predict_proba(fraud_data), 3)
    return prediction, probability

# -------------------------------
# Streamlit App Configuration
# -------------------------------
st.set_page_config(page_title="💳 Fraud Detection App", page_icon="🕵️", layout="wide")

st.title("💳 Credit Card Fraud Detection")

# ⚠️ Alert about PCA
st.warning(
    """
    ⚠️ **Important Note:**  
    This model was trained on **PCA-transformed features** (V1–V28, plus Time and Amount).  
    Please ensure that any input data has already been transformed in the same way as the training dataset.  
    """
)

# -------------------------------
# Model Description Section
# -------------------------------
with st.expander("ℹ️ About the Model"):
    st.markdown(
        """
        - **Algorithm:** Gradient Boosting Classifier  
        - **Why Gradient Boosting?** It combines multiple weak learners (decision trees) into a strong ensemble, 
          making it effective for imbalanced classification problems like fraud detection.  
        - **Evaluation Metrics:** Recall and Precision were prioritized over raw accuracy.  
        - **Performance:**  
          - Recall ≈ 81% (captures ~8 out of 10 frauds)  
          - Precision ≈ 88% (most flagged cases are truly fraud)   
        """
    )

# -------------------------------
# Tabs for Single vs Batch Prediction
# -------------------------------
tab1, tab2 = st.tabs(["🔍 Single Transaction", "📂 Batch Upload"])

# --- Single Transaction Prediction ---
with tab1:
    st.sidebar.header("🔧 Input Transaction Features")

    feature_names = [
        "Time","V1","V2","V3","V4","V5","V6","V7","V8","V9",
        "V10","V11","V12","V13","V14","V15","V16","V17","V18","V19",
        "V20","V21","V22","V23","V24","V25","V26","V27","V28","Amount"
    ]

    default_values = [
        438.000000, 1.491574, -1.088278, 0.552852, -1.289876, -1.391175, -0.101768,
        -1.156643, -0.023064, -1.706012, 1.419577, 0.625897, 0.472241, 1.262940,
        -0.776938, -1.593226, -0.735048, 0.554896, 0.203064, 0.558620, -0.294562,
        -0.224231, 0.008611, -0.090743, 0.045121, 0.593903, -0.100271, 0.052103,
        0.004537, 2.000000
    ]

    user_inputs = []
    for name, default in zip(feature_names, default_values):
        val = st.sidebar.number_input(f"{name}", value=float(default), format="%.6f")
        user_inputs.append(val)

    input_data = pd.DataFrame([user_inputs], columns=feature_names)

    st.subheader("📊 Transaction Data Preview")
    st.write(input_data)

    if st.button("🔍 Predict Fraud", key="single"):
        pred, prob = fraud_detection(input_data)
        if pred[0] == 1:
            st.error(f"⚠️ Fraudulent Transaction Detected! (Confidence: {prob[0][1]*100:.2f}%)")
        else:
            st.success(f"✅ Legitimate Transaction (Confidence: {prob[0][0]*100:.2f}%)")

        st.write("### Prediction Probabilities")
        st.write(pd.DataFrame(prob, columns=["Non-Fraud", "Fraud"]))

# --- Batch Upload Prediction ---
with tab2:
    st.subheader("📂 Upload Batch Transactions")
    uploaded_file = st.file_uploader("Upload a CSV file with the same feature structure (exclude 'Class')", type=["csv"])

    if uploaded_file is not None:
        batch_data = pd.read_csv(uploaded_file)

        # 🔑 Drop target column if present
        expected_features = [
            "Time","V1","V2","V3","V4","V5","V6","V7","V8","V9",
            "V10","V11","V12","V13","V14","V15","V16","V17","V18","V19",
            "V20","V21","V22","V23","V24","V25","V26","V27","V28","Amount"
        ]

        # Common aliases for target column
        target_aliases = ["Class", "class", "target", "Target", "Fraud", "fraud"]

        # Drop any target aliases if present
        to_drop = [col for col in batch_data.columns if col in target_aliases]
        if to_drop:
            st.info(f"ℹ️ Dropped target column(s): {', '.join(to_drop)} before prediction.")
            batch_data = batch_data.drop(columns=to_drop)

        # Keep only expected features
        batch_data = batch_data[[col for col in batch_data.columns if col in expected_features]]

        # Check for missing features
        missing = set(expected_features) - set(batch_data.columns)
        if missing:
            st.warning(f"⚠️ Missing expected features: {', '.join(missing)}. "
                    "Predictions may not be reliable unless all required features are present.")


        st.write("### Preview of Uploaded Data")
        st.dataframe(batch_data.head())

        if st.button("🚀 Run Batch Prediction", key="batch"):
            preds, probs = fraud_detection(batch_data)
            results = batch_data.copy()
            results["Prediction"] = ["Fraud" if p == 1 else "Non-Fraud" for p in preds]
            results["Fraud_Prob"] = probs[:, 1]

            st.write("### Prediction Results")
            st.dataframe(results.head(20))

            total = len(preds)

            fraud_count = int((preds == 1).sum())
            nonfraud_count = total - fraud_count
            fraud_rate = (fraud_count / total) * 100

            st.markdown("### 📌 Batch Summary")

            # Conditional headline
            if fraud_count > 0:
                st.error(
                    f"⚠️ **{fraud_count} fraudulent transactions detected** "
                    f"out of {total} uploaded ({fraud_rate:.2f}%)."
                )
            else:
                st.success(
                    f"✅ No fraudulent transactions detected out of {total} uploaded."
                )


            # 📊 Create summary DataFrame
            summary_df = pd.DataFrame({
                "Class": ["Non-Fraud", "Fraud"],
                "Count": [nonfraud_count, fraud_count]
            })

            # Interactive Pie Chart
            fig = px.pie(
                summary_df,
                names="Class",
                values="Count",
                color="Class",
                color_discrete_map={"Non-Fraud": "#4CAF50", "Fraud": "#F44336"},
                hole=0.3,  # makes it a donut chart (optional)
            )

            fig.update_traces(
                textinfo="label+percent",
                pull=[0, 0.1],  # slightly "explode" the fraud slice for emphasis
            )

            fig.update_layout(
                title_text="Fraud vs Non-Fraud Distribution",
                title_x=0.5
            )

            st.plotly_chart(fig, use_container_width=True)



# -------------------------------
# Footer
# -------------------------------
st.markdown("---")
st.caption("Developed as part of a Machine Learning project on Credit Card Fraud Detection.")
