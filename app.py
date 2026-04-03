# ===============================
# 🚗 CAR INSURANCE AI SYSTEM
# ===============================

import streamlit as st
import joblib
import pandas as pd
import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score

# -----------------------
# PAGE CONFIG
# -----------------------
st.set_page_config(page_title="Car Insurance AI", layout="centered")

st.title("🚗 Car Insurance Claim Prediction System")
st.markdown("### AI-Based Risk & Fraud Detection")

# -----------------------
# LOAD NUMERIC MODEL
# -----------------------
@st.cache_resource
def load_numeric_model():
    return joblib.load("car_claim_numeric_model.pkl")

numeric_model = load_numeric_model()

train_df = pd.read_csv("train.csv")
template = train_df.drop(columns=["is_claim"]).iloc[0:1].copy()

# -----------------------
# LOAD IMAGE MODEL
# -----------------------
@st.cache_resource
def load_image_model():
    device = torch.device("cpu")

    model = torchvision.models.alexnet(weights=None)
    model.classifier[6] = nn.Linear(4096, 2)

    state_dict = torch.load("best_image_model.pth", map_location=device)
    model.load_state_dict(state_dict)

    model.eval()
    return model

model = load_image_model()

# -----------------------
# IMAGE TRANSFORM
# -----------------------
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
])

classes = ["Not Fraud", "Fraud"]

# -----------------------
# MODEL EVALUATION
# -----------------------
@st.cache_data
def evaluate_model():
    df = pd.read_csv("train.csv")

    # -----------------------
    # FEATURE ENGINEERING (SAME AS TRAINING)
    # -----------------------
    df["max_power"] = pd.to_numeric(df.get("max_power", 100), errors="coerce")
    df["gross_weight"] = pd.to_numeric(df.get("gross_weight", 1000), errors="coerce")

    df["engine_per_weight"] = df["max_power"] / (df["gross_weight"] + 1)
    df["age_ratio"] = df["age_of_car"] / (df["age_of_policyholder"] + 1)

    df.fillna(0, inplace=True)

    # -----------------------
    # SPLIT
    # -----------------------
    X = df.drop("is_claim", axis=1)
    y = df["is_claim"]

    # -----------------------
    # PREDICT
    # -----------------------
    y_pred = numeric_model.predict(X)

    # -----------------------
    # METRICS
    # -----------------------
    cm = confusion_matrix(y, y_pred)
    acc = accuracy_score(y, y_pred)
    prec = precision_score(y, y_pred)
    rec = recall_score(y, y_pred)

    return cm, acc, prec, rec, y

# -----------------------
# USER INPUT
# -----------------------
st.header("👤 Enter Customer Details")

age = st.slider("Age of Policyholder", 18, 80, 30)
car_age = st.slider("Age of Car", 0, 20, 2)
fuel = st.selectbox("Fuel Type", ["Petrol", "Diesel", "CNG"])
airbags = st.slider("Number of Airbags", 0, 10, 2)

# -----------------------
# IMAGE INPUT
# -----------------------
st.header("📷 Upload Car Image")
uploaded_file = st.file_uploader("Upload Image", type=["jpg","png","jpeg"])

# -----------------------
# PREDICT
# -----------------------
if st.button("🔍 Predict"):

    # =======================
    # NUMERIC MODEL
    # =======================
    input_data = template.copy()

    input_data["age_of_policyholder"] = age
    input_data["age_of_car"] = car_age
    input_data["fuel_type"] = fuel
    input_data["airbags"] = airbags

    input_data["max_power"] = pd.to_numeric(input_data.get("max_power", 100), errors="coerce")
    input_data["gross_weight"] = pd.to_numeric(input_data.get("gross_weight", 1000), errors="coerce")

    input_data["engine_per_weight"] = input_data["max_power"] / (input_data["gross_weight"] + 1)
    input_data["age_ratio"] = input_data["age_of_car"] / (input_data["age_of_policyholder"] + 1)

    input_data.fillna(0, inplace=True)

    claim_prob = numeric_model.predict_proba(input_data)[:,1][0]

    # Risk logic
    if claim_prob < 0.30:
        risk = "🟢 Low Risk"
        decision_numeric = "Auto Approve"
    elif claim_prob < 0.60:
        risk = "🟡 Medium Risk"
        decision_numeric = "Manual Review"
    else:
        risk = "🔴 High Risk"
        decision_numeric = "Investigate"

    st.subheader("📊 Claim Risk Analysis")
    st.progress(int(claim_prob * 100))
    st.write(f"**Claim Probability:** {claim_prob*100:.2f}%")
    st.write(f"**Risk Level:** {risk}")
    st.write(f"**Decision:** {decision_numeric}")

    # 📈 Risk Chart
    st.markdown("### 📈 Risk Visualization")
    risk_df = pd.DataFrame({
        "Type": ["Safe", "Risk"],
        "Value": [1 - claim_prob, claim_prob]
    })
    st.bar_chart(risk_df.set_index("Type"))

    # =======================
    # IMAGE MODEL
    # =======================
    image_pred_class = "Not Provided"
    confidence = 0

    if uploaded_file is not None:
        img = Image.open(uploaded_file).convert("RGB")
        st.image(img, caption="Uploaded Image", use_container_width=True)

        img_tensor = transform(img).unsqueeze(0)

        with torch.no_grad():
            output = model(img_tensor)
            probs = torch.softmax(output, dim=1)
            pred = torch.argmax(probs, 1).item()
            confidence = probs[0][pred].item()

        threshold = 0.75

        if confidence < threshold:
            image_pred_class = "❌ Uncertain (Low Confidence)"
        else:
            image_pred_class = classes[pred]

        st.subheader("🧠 Image Fraud Detection")
        st.write(f"Prediction: {image_pred_class}")
        st.write(f"Confidence: {confidence*100:.2f}%")
        st.progress(int(confidence * 100))

        # 📉 Confidence Chart
        conf_df = pd.DataFrame({
            "Confidence": [confidence]
        })
        st.line_chart(conf_df)

    else:
        st.warning("⚠️ Please upload image")

    # =======================
    # FINAL DECISION
    # =======================
    st.subheader("🎯 Final Decision")

    final_score = (claim_prob * 0.6) + (confidence * 0.4)

    if image_pred_class == "Fraud":
        final = "🚫 Reject Claim"
    elif image_pred_class == "❌ Uncertain (Low Confidence)":
        final = "⚠️ Need Better Image"
    elif final_score > 0.6:
        final = "🔍 Investigate"
    else:
        final = "✅ Approve Claim"

    st.success(final)
    st.write(f"Combined Score: {final_score*100:.2f}%")

# -----------------------
# 📊 MODEL PERFORMANCE DASHBOARD
# -----------------------
st.subheader("📊 Model Performance Dashboard")

try:
    cm, acc, prec, rec, y = evaluate_model()

    # Metrics
    col1, col2, col3 = st.columns(3)
    col1.metric("Accuracy", f"{acc*100:.2f}%")
    col2.metric("Precision", f"{prec*100:.2f}%")
    col3.metric("Recall", f"{rec*100:.2f}%")

    # Confusion Matrix
    st.markdown("### 🔲 Confusion Matrix")
    fig1, ax1 = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["No Claim", "Claim"],
                yticklabels=["No Claim", "Claim"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    st.pyplot(fig1)

    # Distribution Chart
    st.markdown("### 📊 Claim Distribution")
    fig2, ax2 = plt.subplots()
    pd.Series(y).value_counts().plot(kind="bar", ax=ax2)
    ax2.set_title("Claim vs No Claim")
    ax2.set_xlabel("Class")
    ax2.set_ylabel("Count")
    st.pyplot(fig2)

except Exception as e:
    st.error(f"Error generating dashboard: {e}")