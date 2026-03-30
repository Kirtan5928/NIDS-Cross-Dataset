import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.model_selection import train_test_split

# ── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NIDS Dashboard",
    page_icon="🛡️",
    layout="wide"
)

# ── Load models and data ─────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    rf  = joblib.load("models/random_forest.pkl")
    xgb = joblib.load("models/xgboost.pkl")
    return rf, xgb

@st.cache_data
def load_data():
    df = pd.read_csv("data/merged_dataset.csv")
    return df

rf, xgb = load_models()
df      = load_data()

# ── Sidebar navigation ───────────────────────────────────────────────────────
st.sidebar.title("🛡️ NIDS Dashboard")
st.sidebar.markdown("Cross-Dataset Network Intrusion Detection System")
page = st.sidebar.radio(
    "Navigate",
    ["Model Performance", "Live Classification", "Dataset Distribution"]
)

# ── Feature columns ──────────────────────────────────────────────────────────
FEATURES = ['dur', 'spkts', 'dpkts', 'sbytes', 'dbytes', 'rate',
            'sload', 'dload', 'sinpkt', 'dinpkt', 'sjit', 'djit',
            'swin', 'dwin', 'smean', 'dmean', 'proto', 'service', 'state']

# ════════════════════════════════════════════════════════════════════════════
# PAGE 1 — MODEL PERFORMANCE
# ════════════════════════════════════════════════════════════════════════════
if page == "Model Performance":
    st.title("📊 Model Performance")
    st.markdown("Evaluation of Random Forest and XGBoost on the merged test set.")

    # Prepare test data
    X = df[FEATURES]
    y = df["Label"]
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    with st.spinner("Generating predictions..."):
        rf_preds  = rf.predict(X_test)
        xgb_preds = xgb.predict(X_test)

    rf_acc  = round(accuracy_score(y_test, rf_preds)  * 100, 2)
    xgb_acc = round(accuracy_score(y_test, xgb_preds) * 100, 2)

    # ── Accuracy cards ───────────────────────────────────────────────────────
    st.subheader("Accuracy Scores")
    col1, col2 = st.columns(2)
    col1.metric("🌲 Random Forest", f"{rf_acc}%")
    col2.metric("⚡ XGBoost",        f"{xgb_acc}%")

    # ── Confusion matrices ───────────────────────────────────────────────────
    st.subheader("Confusion Matrices")
    col1, col2 = st.columns(2)

    for col, preds, name in [
        (col1, rf_preds,  "Random Forest"),
        (col2, xgb_preds, "XGBoost")
    ]:
        cm = confusion_matrix(y_test, preds)
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues", ax=ax,
            xticklabels=["Benign", "Attack"],
            yticklabels=["Benign", "Attack"]
        )
        ax.set_title(f"{name} Confusion Matrix")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        col.pyplot(fig)
        plt.close()

    # ── Classification reports ───────────────────────────────────────────────
    st.subheader("Classification Reports")
    col1, col2 = st.columns(2)

    for col, preds, name in [
        (col1, rf_preds,  "Random Forest"),
        (col2, xgb_preds, "XGBoost")
    ]:
        report = classification_report(
            y_test, preds,
            target_names=["Benign", "Attack"],
            output_dict=True
        )
        report_df = pd.DataFrame(report).transpose().round(2)
        col.markdown(f"**{name}**")
        col.dataframe(report_df)

# ════════════════════════════════════════════════════════════════════════════
# PAGE 2 — LIVE CLASSIFICATION
# ════════════════════════════════════════════════════════════════════════════
elif page == "Live Classification":
    st.title("🔍 Live Traffic Classification")
    st.markdown("Enter network traffic values below to classify as Benign or Attack.")

    st.subheader("Input Traffic Features")

    col1, col2, col3 = st.columns(3)

    with col1:
        dur    = st.number_input("Flow Duration (dur)",    value=0.0)
        spkts  = st.number_input("Source Packets (spkts)", value=0)
        dpkts  = st.number_input("Dest Packets (dpkts)",   value=0)
        sbytes = st.number_input("Source Bytes (sbytes)",  value=0)
        dbytes = st.number_input("Dest Bytes (dbytes)",    value=0)
        rate   = st.number_input("Transfer Rate (rate)",   value=0.0)

    with col2:
        sload  = st.number_input("Source Load (sload)",    value=0.0)
        dload  = st.number_input("Dest Load (dload)",      value=0.0)
        sinpkt = st.number_input("Src Inter-Packet (sinpkt)", value=0.0)
        dinpkt = st.number_input("Dst Inter-Packet (dinpkt)", value=0.0)
        sjit   = st.number_input("Source Jitter (sjit)",   value=0.0)
        djit   = st.number_input("Dest Jitter (djit)",     value=0.0)

    with col3:
        swin   = st.number_input("Src TCP Window (swin)",  value=0)
        dwin   = st.number_input("Dst TCP Window (dwin)",  value=0)
        smean  = st.number_input("Src Mean Pkt Size (smean)", value=0.0)
        dmean  = st.number_input("Dst Mean Pkt Size (dmean)", value=0.0)
        proto   = st.number_input("Protocol (proto)",      value=0)
        service = st.number_input("Service (service)",     value=0)
        state   = st.number_input("State (state)",         value=0)

    if st.button("🔍 Classify Traffic", use_container_width=True):
        input_data = pd.DataFrame([[
            dur, spkts, dpkts, sbytes, dbytes, rate,
            sload, dload, sinpkt, dinpkt, sjit, djit,
            swin, dwin, smean, dmean, proto, service, state
        ]], columns=FEATURES)

        # Get predictions and probabilities
        rf_pred   = rf.predict(input_data)[0]
        rf_prob   = rf.predict_proba(input_data)[0]
        xgb_pred  = xgb.predict(input_data)[0]
        xgb_prob  = xgb.predict_proba(input_data)[0]

        # Risk score = average attack probability from both models
        risk_score = round((rf_prob[1] + xgb_prob[1]) / 2 * 100, 1)

        st.markdown("---")
        st.subheader("Classification Result")

        col1, col2, col3 = st.columns(3)

        # Risk score
        if risk_score >= 70:
            col1.error(f"⚠️ Risk Score: {risk_score}%")
        elif risk_score >= 40:
            col1.warning(f"⚠️ Risk Score: {risk_score}%")
        else:
            col1.success(f"✅ Risk Score: {risk_score}%")

        # Model verdicts
        col2.metric(
            "🌲 Random Forest",
            "ATTACK 🚨" if rf_pred == 1 else "BENIGN ✅",
            f"Confidence: {round(max(rf_prob) * 100, 1)}%"
        )
        col3.metric(
            "⚡ XGBoost",
            "ATTACK 🚨" if xgb_pred == 1 else "BENIGN ✅",
            f"Confidence: {round(max(xgb_prob) * 100, 1)}%"
        )

        # Risk bar
        st.markdown("**Attack Probability Breakdown**")
        prob_df = pd.DataFrame({
            "Model": ["Random Forest", "XGBoost"],
            "Attack Probability (%)": [
                round(rf_prob[1]  * 100, 1),
                round(xgb_prob[1] * 100, 1)
            ]
        })
        st.bar_chart(prob_df.set_index("Model"))

# ════════════════════════════════════════════════════════════════════════════
# PAGE 3 — DATASET DISTRIBUTION
# ════════════════════════════════════════════════════════════════════════════
elif page == "Dataset Distribution":
    st.title("📈 Dataset Distribution")

    col1, col2 = st.columns(2)

    # Overall label distribution
    with col1:
        st.subheader("Benign vs Attack")
        label_counts = df["Label"].value_counts()
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.pie(
            label_counts,
            labels=["Benign", "Attack"],
            autopct="%1.1f%%",
            colors=["#2ecc71", "#e74c3c"],
            startangle=90
        )
        ax.set_title("Overall Label Distribution")
        st.pyplot(fig)
        plt.close()

    # Source distribution
    with col2:
        st.subheader("Dataset Source Split")
        source_counts = df["source"].value_counts()
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.bar(
            ["CICIDS2017", "UNSW-NB15"],
            source_counts.values,
            color=["#3498db", "#9b59b6"]
        )
        ax.set_title("Rows per Dataset")
        ax.set_ylabel("Number of Rows")
        for i, v in enumerate(source_counts.values):
            ax.text(i, v + 10000, f"{v:,}", ha="center", fontweight="bold")
        st.pyplot(fig)
        plt.close()

    # Feature distributions
    st.subheader("Feature Statistics")
    st.dataframe(df[FEATURES].describe().round(2))