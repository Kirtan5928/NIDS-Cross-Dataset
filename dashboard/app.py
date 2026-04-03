import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.model_selection import train_test_split

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NIDS · Threat Intelligence",
    page_icon="🛡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Global CSS — futuristic dark theme ───────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;600;700;800&display=swap');

/* ── Base ── */
html, body, [class*="css"] {
    font-family: 'Syne', sans-serif;
    background-color: #050A0F;
    color: #E2E8F0;
}

/* ── Hide streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 2rem 2.5rem 2rem 2.5rem; }

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: #080E16;
    border-right: 1px solid #0FF4C620;
}
section[data-testid="stSidebar"] .stRadio label {
    font-family: 'Space Mono', monospace;
    font-size: 0.8rem;
    color: #94A3B8;
    letter-spacing: 0.05em;
}
section[data-testid="stSidebar"] .stRadio [data-checked="true"] label {
    color: #0FF4C6 !important;
}

/* ── Page title ── */
.nids-title {
    font-family: 'Syne', sans-serif;
    font-weight: 800;
    font-size: 2.4rem;
    letter-spacing: -0.02em;
    background: linear-gradient(135deg, #0FF4C6 0%, #3B82F6 50%, #A855F7 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 0.2rem;
}
.nids-subtitle {
    font-family: 'Space Mono', monospace;
    font-size: 0.75rem;
    color: #475569;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-bottom: 2rem;
}

/* ── Metric cards ── */
.metric-card {
    background: linear-gradient(135deg, #0D1B2A 0%, #0A1628 100%);
    border: 1px solid #0FF4C620;
    border-radius: 16px;
    padding: 1.5rem;
    position: relative;
    overflow: hidden;
}
.metric-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, #0FF4C6, #3B82F6, #A855F7);
}
.metric-label {
    font-family: 'Space Mono', monospace;
    font-size: 0.65rem;
    color: #475569;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    margin-bottom: 0.5rem;
}
.metric-value {
    font-family: 'Syne', sans-serif;
    font-weight: 800;
    font-size: 2rem;
    color: #0FF4C6;
}
.metric-sub {
    font-family: 'Space Mono', monospace;
    font-size: 0.65rem;
    color: #64748B;
    margin-top: 0.3rem;
}

/* ── Section headers ── */
.section-header {
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
    color: #0FF4C6;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    border-left: 3px solid #0FF4C6;
    padding-left: 0.75rem;
    margin: 2rem 0 1rem 0;
}

/* ── Verdict box ── */
.verdict-safe {
    background: linear-gradient(135deg, #052E16 0%, #064E3B 100%);
    border: 1px solid #10B98140;
    border-left: 4px solid #10B981;
    border-radius: 12px;
    padding: 1.5rem 2rem;
    text-align: center;
}
.verdict-danger {
    background: linear-gradient(135deg, #1F0505 0%, #3B0E0E 100%);
    border: 1px solid #EF444440;
    border-left: 4px solid #EF4444;
    border-radius: 12px;
    padding: 1.5rem 2rem;
    text-align: center;
}
.verdict-warning {
    background: linear-gradient(135deg, #1C1005 0%, #38200A 100%);
    border: 1px solid #F59E0B40;
    border-left: 4px solid #F59E0B;
    border-radius: 12px;
    padding: 1.5rem 2rem;
    text-align: center;
}
.verdict-label {
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    margin-bottom: 0.5rem;
    color: #94A3B8;
}
.verdict-value {
    font-family: 'Syne', sans-serif;
    font-weight: 800;
    font-size: 1.8rem;
}

/* ── Input fields ── */
.stNumberInput input {
    background: #0D1B2A !important;
    border: 1px solid #1E3A5F !important;
    color: #E2E8F0 !important;
    border-radius: 8px !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 0.8rem !important;
}
.stNumberInput input:focus {
    border-color: #0FF4C6 !important;
    box-shadow: 0 0 0 1px #0FF4C640 !important;
}

/* ── Button ── */
.stButton > button {
    background: linear-gradient(135deg, #0FF4C6 0%, #3B82F6 100%);
    color: #050A0F;
    font-family: 'Space Mono', monospace;
    font-size: 0.8rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    border: none;
    border-radius: 10px;
    padding: 0.75rem 2rem;
    width: 100%;
    cursor: pointer;
    transition: all 0.2s;
}
.stButton > button:hover {
    opacity: 0.9;
    transform: translateY(-1px);
    box-shadow: 0 8px 25px #0FF4C630;
}

/* ── Divider ── */
.nids-divider {
    border: none;
    height: 1px;
    background: linear-gradient(90deg, transparent, #0FF4C620, transparent);
    margin: 2rem 0;
}

/* ── Spinner ── */
.stSpinner > div { border-top-color: #0FF4C6 !important; }

/* ── Dataframe ── */
.dataframe { font-family: 'Space Mono', monospace; font-size: 0.75rem; }

/* ── Sidebar logo ── */
.sidebar-logo {
    font-family: 'Syne', sans-serif;
    font-weight: 800;
    font-size: 1.3rem;
    background: linear-gradient(135deg, #0FF4C6, #3B82F6);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 0.2rem;
}
.sidebar-version {
    font-family: 'Space Mono', monospace;
    font-size: 0.6rem;
    color: #334155;
    letter-spacing: 0.1em;
    margin-bottom: 2rem;
}
</style>
""", unsafe_allow_html=True)

# ── Matplotlib dark theme ─────────────────────────────────────────────────────
plt.rcParams.update({
    'figure.facecolor':  '#080E16',
    'axes.facecolor':    '#080E16',
    'axes.edgecolor':    '#1E3A5F',
    'axes.labelcolor':   '#94A3B8',
    'xtick.color':       '#475569',
    'ytick.color':       '#475569',
    'text.color':        '#E2E8F0',
    'grid.color':        '#0F2035',
    'grid.linestyle':    '--',
    'grid.alpha':        0.5,
    'font.family':       'monospace',
})

ACCENT   = '#0FF4C6'
BLUE     = '#3B82F6'
PURPLE   = '#A855F7'
RED      = '#EF4444'
AMBER    = '#F59E0B'
BG       = '#080E16'
CARD_BG  = '#0D1B2A'

# ── Load models and data ──────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_models():
    rf  = joblib.load("models/random_forest.pkl")
    xgb = joblib.load("models/xgboost.pkl")
    return rf, xgb

@st.cache_data(show_spinner=False)
def load_data():
    return pd.read_csv("data/merged_dataset.csv")

@st.cache_data(show_spinner=False)
def get_test_predictions(_rf, _xgb, df):
    FEATURES = ['dur','spkts','dpkts','sbytes','dbytes','rate','sload','dload',
                'sinpkt','dinpkt','sjit','djit','swin','dwin','smean','dmean',
                'proto','service','state']
    X = df[FEATURES]
    y = df["Label"]
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    rf_preds  = _rf.predict(X_test)
    xgb_preds = _xgb.predict(X_test)
    return y_test, rf_preds, xgb_preds

rf, xgb = load_models()
df = load_data()

FEATURES = ['dur','spkts','dpkts','sbytes','dbytes','rate','sload','dload',
            'sinpkt','dinpkt','sjit','djit','swin','dwin','smean','dmean',
            'proto','service','state']

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sidebar-logo">⬡ NIDS</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-version">THREAT INTELLIGENCE PLATFORM · v2.0</div>', unsafe_allow_html=True)
    st.markdown('<hr class="nids-divider">', unsafe_allow_html=True)
    page = st.radio(
        "NAVIGATE",
        ["Overview", "Live Classifier", "Dataset Intel"],
        label_visibility="visible"
    )
    st.markdown('<hr class="nids-divider">', unsafe_allow_html=True)
    st.markdown("""
    <div style='font-family: Space Mono, monospace; font-size: 0.6rem; color: #334155; line-height: 1.8;'>
    DATASETS<br>
    · UNSW-NB15<br>
    · CICIDS2017<br><br>
    MODELS<br>
    · Random Forest<br>
    · XGBoost<br><br>
    SAMPLES<br>
    · 2,778,471 total
    </div>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
if page == "Overview":
    st.markdown('<div class="nids-title">Threat Intelligence Dashboard</div>', unsafe_allow_html=True)
    st.markdown('<div class="nids-subtitle">Cross-Dataset Network Intrusion Detection · UNSW-NB15 + CICIDS2017</div>', unsafe_allow_html=True)

    with st.spinner(""):
        y_test, rf_preds, xgb_preds = get_test_predictions(rf, xgb, df)

    rf_acc  = round(accuracy_score(y_test, rf_preds)  * 100, 2)
    xgb_acc = round(accuracy_score(y_test, xgb_preds) * 100, 2)
    rf_report  = classification_report(y_test, rf_preds,  output_dict=True)
    xgb_report = classification_report(y_test, xgb_preds, output_dict=True)

    # ── Top metric cards ──
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Random Forest</div>
            <div class="metric-value">{rf_acc}%</div>
            <div class="metric-sub">Overall Accuracy</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">XGBoost</div>
            <div class="metric-value" style="color:#3B82F6">{xgb_acc}%</div>
            <div class="metric-sub">Overall Accuracy</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Test Samples</div>
            <div class="metric-value" style="color:#A855F7">555K</div>
            <div class="metric-sub">20% holdout set</div>
        </div>""", unsafe_allow_html=True)
    with c4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Attack Recall · RF</div>
            <div class="metric-value" style="color:#F59E0B">{round(rf_report['1']['recall']*100,1)}%</div>
            <div class="metric-sub">True positive rate</div>
        </div>""", unsafe_allow_html=True)

    st.markdown('<div class="nids-divider"></div>', unsafe_allow_html=True)

    # ── Confusion matrices ──
    st.markdown('<div class="section-header">Confusion Matrices</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)

    for col, preds, name, color in [
        (col1, rf_preds,  "Random Forest", ACCENT),
        (col2, xgb_preds, "XGBoost",       BLUE)
    ]:
        cm = confusion_matrix(y_test, preds)
        fig, ax = plt.subplots(figsize=(5, 4))
        fig.patch.set_facecolor(BG)
        cmap = sns.light_palette(color, as_cmap=True)
        sns.heatmap(
            cm, annot=True, fmt=",d", cmap=cmap, ax=ax,
            xticklabels=["Benign", "Attack"],
            yticklabels=["Benign", "Attack"],
            linewidths=0.5, linecolor='#0A1628',
            annot_kws={"size": 13, "weight": "bold", "color": "#050A0F"}
        )
        ax.set_title(name, fontsize=11, color=color,
                     fontfamily='monospace', pad=12, fontweight='bold')
        ax.set_xlabel("Predicted", labelpad=8)
        ax.set_ylabel("Actual",    labelpad=8)
        plt.tight_layout()
        col.pyplot(fig, use_container_width=True)
        plt.close()

    st.markdown('<div class="nids-divider"></div>', unsafe_allow_html=True)

    # ── Precision / Recall / F1 bar chart ──
    st.markdown('<div class="section-header">Model Metrics Comparison</div>', unsafe_allow_html=True)

    metrics_data = {
        'Metric': ['Precision\n(Benign)', 'Recall\n(Benign)', 'F1\n(Benign)',
                   'Precision\n(Attack)', 'Recall\n(Attack)', 'F1\n(Attack)'],
        'Random Forest': [
            rf_report['0']['precision'],  rf_report['0']['recall'],  rf_report['0']['f1-score'],
            rf_report['1']['precision'],  rf_report['1']['recall'],  rf_report['1']['f1-score'],
        ],
        'XGBoost': [
            xgb_report['0']['precision'], xgb_report['0']['recall'], xgb_report['0']['f1-score'],
            xgb_report['1']['precision'], xgb_report['1']['recall'], xgb_report['1']['f1-score'],
        ]
    }
    mdf = pd.DataFrame(metrics_data)

    fig, ax = plt.subplots(figsize=(11, 4))
    fig.patch.set_facecolor(BG)
    x     = np.arange(len(mdf['Metric']))
    width = 0.35
    bars1 = ax.bar(x - width/2, mdf['Random Forest'], width,
                   color=ACCENT,  alpha=0.85, label='Random Forest', zorder=3)
    bars2 = ax.bar(x + width/2, mdf['XGBoost'],       width,
                   color=BLUE,    alpha=0.85, label='XGBoost',       zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels(mdf['Metric'], fontsize=8)
    ax.set_ylim(0.88, 1.02)
    ax.set_ylabel("Score", labelpad=8)
    ax.yaxis.grid(True, zorder=0)
    ax.set_axisbelow(True)
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                f'{bar.get_height():.2f}', ha='center', va='bottom',
                fontsize=7.5, color=ACCENT, fontweight='bold')
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                f'{bar.get_height():.2f}', ha='center', va='bottom',
                fontsize=7.5, color=BLUE,  fontweight='bold')
    ax.legend(facecolor=CARD_BG, edgecolor='#1E3A5F',
              labelcolor='#E2E8F0', fontsize=9)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close()

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — LIVE CLASSIFIER
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Live Classifier":
    st.markdown('<div class="nids-title">Live Traffic Classifier</div>', unsafe_allow_html=True)
    st.markdown('<div class="nids-subtitle">Input network flow features · Get instant threat verdict</div>', unsafe_allow_html=True)

    with st.form("classifier_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown('<div class="section-header">Flow Metrics</div>', unsafe_allow_html=True)
            dur    = st.number_input("Flow Duration (dur)",       value=0.0, format="%.4f")
            spkts  = st.number_input("Source Packets (spkts)",    value=0,   step=1)
            dpkts  = st.number_input("Dest Packets (dpkts)",      value=0,   step=1)
            sbytes = st.number_input("Source Bytes (sbytes)",     value=0,   step=1)
            dbytes = st.number_input("Dest Bytes (dbytes)",       value=0,   step=1)
            rate   = st.number_input("Transfer Rate (rate)",      value=0.0, format="%.4f")
        with c2:
            st.markdown('<div class="section-header">Load & Timing</div>', unsafe_allow_html=True)
            sload  = st.number_input("Source Load (sload)",       value=0.0, format="%.4f")
            dload  = st.number_input("Dest Load (dload)",         value=0.0, format="%.4f")
            sinpkt = st.number_input("Src Inter-Packet (sinpkt)", value=0.0, format="%.4f")
            dinpkt = st.number_input("Dst Inter-Packet (dinpkt)", value=0.0, format="%.4f")
            sjit   = st.number_input("Source Jitter (sjit)",      value=0.0, format="%.4f")
            djit   = st.number_input("Dest Jitter (djit)",        value=0.0, format="%.4f")
        with c3:
            st.markdown('<div class="section-header">Window & Protocol</div>', unsafe_allow_html=True)
            swin    = st.number_input("Src TCP Window (swin)",    value=0,   step=1)
            dwin    = st.number_input("Dst TCP Window (dwin)",    value=0,   step=1)
            smean   = st.number_input("Src Mean Pkt (smean)",     value=0.0, format="%.4f")
            dmean   = st.number_input("Dst Mean Pkt (dmean)",     value=0.0, format="%.4f")
            proto   = st.number_input("Protocol (proto)",         value=0,   step=1)
            service = st.number_input("Service (service)",        value=0,   step=1)
            state   = st.number_input("State (state)",            value=0,   step=1)

        submitted = st.form_submit_button("⬡  ANALYSE TRAFFIC")

    if submitted:
        input_df = pd.DataFrame([[
            dur, spkts, dpkts, sbytes, dbytes, rate,
            sload, dload, sinpkt, dinpkt, sjit, djit,
            swin, dwin, smean, dmean, proto, service, state
        ]], columns=FEATURES)

        rf_pred   = rf.predict(input_df)[0]
        rf_prob   = rf.predict_proba(input_df)[0]
        xgb_pred  = xgb.predict(input_df)[0]
        xgb_prob  = xgb.predict_proba(input_df)[0]
        risk      = round((rf_prob[1] + xgb_prob[1]) / 2 * 100, 1)

        st.markdown('<div class="nids-divider"></div>', unsafe_allow_html=True)
        st.markdown('<div class="section-header">Threat Verdict</div>', unsafe_allow_html=True)

        # ── Verdict box ──
        if risk >= 70:
            st.markdown(f"""
            <div class="verdict-danger">
                <div class="verdict-label">Threat Status</div>
                <div class="verdict-value" style="color:#EF4444">⚠ MALICIOUS TRAFFIC</div>
                <div style="font-family:Space Mono,monospace;font-size:0.7rem;color:#94A3B8;margin-top:0.5rem;">
                Risk Score: {risk}% · High Confidence Attack Detected
                </div>
            </div>""", unsafe_allow_html=True)
        elif risk >= 40:
            st.markdown(f"""
            <div class="verdict-warning">
                <div class="verdict-label">Threat Status</div>
                <div class="verdict-value" style="color:#F59E0B">◈ SUSPICIOUS TRAFFIC</div>
                <div style="font-family:Space Mono,monospace;font-size:0.7rem;color:#94A3B8;margin-top:0.5rem;">
                Risk Score: {risk}% · Requires Manual Investigation
                </div>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="verdict-safe">
                <div class="verdict-label">Threat Status</div>
                <div class="verdict-value" style="color:#10B981">✓ BENIGN TRAFFIC</div>
                <div style="font-family:Space Mono,monospace;font-size:0.7rem;color:#94A3B8;margin-top:0.5rem;">
                Risk Score: {risk}% · No Threat Detected
                </div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Model confidence cards ──
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Risk Score</div>
                <div class="metric-value" style="color:{'#EF4444' if risk>=70 else '#F59E0B' if risk>=40 else '#10B981'}">{risk}%</div>
                <div class="metric-sub">Ensemble average</div>
            </div>""", unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Random Forest</div>
                <div class="metric-value" style="color:{ACCENT}">{'ATTACK' if rf_pred==1 else 'BENIGN'}</div>
                <div class="metric-sub">Confidence: {round(max(rf_prob)*100,1)}%</div>
            </div>""", unsafe_allow_html=True)
        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">XGBoost</div>
                <div class="metric-value" style="color:{BLUE}">{'ATTACK' if xgb_pred==1 else 'BENIGN'}</div>
                <div class="metric-sub">Confidence: {round(max(xgb_prob)*100,1)}%</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Attack probability gauge chart ──
        st.markdown('<div class="section-header">Attack Probability</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(7, 3))
        fig.patch.set_facecolor(BG)
        models  = ['Random Forest', 'XGBoost']
        probs   = [round(rf_prob[1]*100,1), round(xgb_prob[1]*100,1)]
        colors  = [ACCENT, BLUE]
        bars    = ax.barh(models, probs, color=colors, alpha=0.85, height=0.4)
        ax.set_xlim(0, 100)
        ax.axvline(x=70, color=RED,   linestyle='--', alpha=0.5, linewidth=1, label='High risk (70%)')
        ax.axvline(x=40, color=AMBER, linestyle='--', alpha=0.5, linewidth=1, label='Medium risk (40%)')
        for bar, prob in zip(bars, probs):
            ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                    f'{prob}%', va='center', fontsize=11, fontweight='bold',
                    color='#E2E8F0')
        ax.legend(facecolor=CARD_BG, edgecolor='#1E3A5F',
                  labelcolor='#94A3B8', fontsize=8, loc='lower right')
        ax.set_xlabel("Attack Probability (%)")
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — DATASET INTEL
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Dataset Intel":
    st.markdown('<div class="nids-title">Dataset Intelligence</div>', unsafe_allow_html=True)
    st.markdown('<div class="nids-subtitle">Merged dataset · UNSW-NB15 + CICIDS2017 · 2,778,471 samples</div>', unsafe_allow_html=True)

    # ── Top stats ──
    total    = len(df)
    benign   = int((df["Label"] == 0).sum())
    attacks  = int((df["Label"] == 1).sum())
    unsw_c   = int((df["source"] == 0).sum())
    cicids_c = int((df["source"] == 1).sum())

    c1, c2, c3, c4 = st.columns(4)
    for col, label, val, color in [
        (c1, "Total Samples",  f"{total:,}",   ACCENT),
        (c2, "Benign Flows",   f"{benign:,}",  '#10B981'),
        (c3, "Attack Flows",   f"{attacks:,}", RED),
        (c4, "Attack Ratio",   f"{round(attacks/total*100,1)}%", AMBER),
    ]:
        with col:
            col.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">{label}</div>
                <div class="metric-value" style="color:{color}">{val}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown('<div class="nids-divider"></div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    # ── Donut — benign vs attack ──
    with col1:
        st.markdown('<div class="section-header">Traffic Distribution</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(5, 5))
        fig.patch.set_facecolor(BG)
        sizes  = [benign, attacks]
        colors_pie = [ACCENT, RED]
        labels_pie = [f'Benign\n{benign:,}', f'Attack\n{attacks:,}']
        wedges, texts, autotexts = ax.pie(
            sizes,
            labels=labels_pie,
            colors=colors_pie,
            autopct='%1.1f%%',
            startangle=90,
            pctdistance=0.78,
            wedgeprops=dict(width=0.55, edgecolor=BG, linewidth=3)
        )
        for t in texts:
            t.set_color('#94A3B8')
            t.set_fontsize(9)
        for at in autotexts:
            at.set_color('#050A0F')
            at.set_fontsize(9)
            at.set_fontweight('bold')
        ax.text(0, 0, f'{round(attacks/total*100,1)}%\nattacks',
                ha='center', va='center',
                fontsize=13, fontweight='bold', color=RED,
                fontfamily='monospace')
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

    # ── Dataset source split ──
    with col2:
        st.markdown('<div class="section-header">Dataset Source Split</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(5, 5))
        fig.patch.set_facecolor(BG)
        sizes2  = [cicids_c, unsw_c]
        colors2 = [BLUE, PURPLE]
        labels2 = [f'CICIDS2017\n{cicids_c:,}', f'UNSW-NB15\n{unsw_c:,}']
        wedges2, texts2, autotexts2 = ax.pie(
            sizes2,
            labels=labels2,
            colors=colors2,
            autopct='%1.1f%%',
            startangle=90,
            pctdistance=0.78,
            wedgeprops=dict(width=0.55, edgecolor=BG, linewidth=3)
        )
        for t in texts2:
            t.set_color('#94A3B8')
            t.set_fontsize(9)
        for at in autotexts2:
            at.set_color('#050A0F')
            at.set_fontsize(9)
            at.set_fontweight('bold')
        ax.text(0, 0, '2 datasets\nmerged',
                ha='center', va='center',
                fontsize=10, fontweight='bold', color='#94A3B8',
                fontfamily='monospace')
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

    st.markdown('<div class="nids-divider"></div>', unsafe_allow_html=True)

    # ── Feature statistics table ──
    st.markdown('<div class="section-header">Feature Statistics</div>', unsafe_allow_html=True)
    stats_df = df[FEATURES].describe().round(3)
    st.dataframe(
        stats_df,
        use_container_width=True,
        height=280
    )