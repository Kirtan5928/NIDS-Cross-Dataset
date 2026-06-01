# dashboard/app.py — NIDS Cross-Dataset · v3.1
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import time, random
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.model_selection import train_test_split

st.set_page_config(
    page_title="NIDS · Threat Intelligence",
    page_icon="🛡",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

section[data-testid="stSidebar"] {
    transform: none !important; width:260px !important; min-width:260px !important;
    background:#161B27 !important; border-right:1px solid #252D3D !important;
}
section[data-testid="stSidebar"] > div { width:260px !important; }
button[data-testid="collapsedControl"] { display:none !important; }

.stApp { background:#0F1117; font-family:'Inter',sans-serif; }
body, p, div, span { color:#C9D4E0; }
h1,h2,h3 { color:#F0F4F8 !important; font-family:'Inter',sans-serif !important; font-weight:700 !important; }

div[data-testid="metric-container"] {
    background:#161B27 !important; border:1px solid #252D3D !important;
    border-radius:8px !important; padding:16px 20px !important;
}
div[data-testid="metric-container"] label {
    color:#5A6A82 !important; font-family:'JetBrains Mono',monospace !important;
    font-size:0.68rem !important; letter-spacing:0.8px !important; text-transform:uppercase !important;
}
div[data-testid="metric-container"] div[data-testid="stMetricValue"] {
    color:#F0F4F8 !important; font-family:'Inter',sans-serif !important;
    font-size:1.8rem !important; font-weight:700 !important;
}

.sb-header  { font-family:'Inter',sans-serif; font-size:0.95rem; font-weight:700; color:#F0F4F8; }
.sb-sub     { font-family:'JetBrains Mono',monospace; font-size:0.63rem; color:#3D5068; }
.sb-section { font-family:'JetBrains Mono',monospace; font-size:0.63rem; color:#3D5068;
              letter-spacing:1.5px; text-transform:uppercase; margin:14px 0 6px; }
.sb-stat    { font-family:'JetBrains Mono',monospace; font-size:0.68rem; color:#5A6A82;
              padding:5px 0; border-bottom:1px solid #1E2535; line-height:1.5; }
.sb-stat b  { color:#60A5FA; }

.alert-row {
    display:flex; align-items:center; gap:10px; padding:7px 12px;
    border-radius:6px; margin:3px 0; background:#0F1117;
    border:1px solid #1E2535; font-family:'JetBrains Mono',monospace; font-size:0.7rem;
}
.dot-red   { width:7px; height:7px; border-radius:50%; background:#EF4444; flex-shrink:0; }
.dot-green { width:7px; height:7px; border-radius:50%; background:#22C55E; flex-shrink:0; }

.verdict { border-radius:8px; padding:16px 20px; margin:10px 0; font-family:'Inter',sans-serif; }
.verdict-attack { background:#1A0F0F; border:1px solid #3D1515; border-left:4px solid #EF4444; }
.verdict-benign { background:#0F1A13; border:1px solid #153D1E; border-left:4px solid #22C55E; }
.verdict-warn   { background:#1A170F; border:1px solid #3D3115; border-left:4px solid #EAB308; }
.verdict-label  { font-family:'JetBrains Mono',monospace; font-size:0.65rem; color:#5A6A82;
                  text-transform:uppercase; letter-spacing:1.5px; margin-bottom:6px; }
.verdict-title  { font-size:1.2rem; font-weight:700; color:#F0F4F8; margin:4px 0 6px; }

.warn-box {
    background:#1A170F; border:1px solid #3D3115; border-left:4px solid #EAB308;
    border-radius:8px; padding:14px 18px; margin:12px 0;
    font-family:'Inter',sans-serif; font-size:0.85rem; color:#A8B8CC; line-height:1.7;
}
.info-box {
    background:#0F1520; border:1px solid #1E3A5F; border-left:4px solid #60A5FA;
    border-radius:8px; padding:14px 18px; margin:12px 0;
    font-family:'Inter',sans-serif; font-size:0.85rem; color:#A8B8CC; line-height:1.7;
}

/* Remediation cards */
.remed-card {
    background:#161B27; border:1px solid #252D3D; border-radius:8px;
    padding:16px 20px; margin:8px 0; position:relative; overflow:hidden;
}
.remed-card::before {
    content:''; position:absolute; left:0; top:0; bottom:0;
    width:3px; border-radius:8px 0 0 8px;
}
.remed-scan::before    { background:#EF4444; }
.remed-dos::before     { background:#F97316; }
.remed-brute::before   { background:#EAB308; }
.remed-exfil::before   { background:#A78BFA; }
.remed-generic::before { background:#60A5FA; }

.remed-title { font-family:'Inter',sans-serif; font-size:0.9rem; font-weight:700; color:#F0F4F8; margin-bottom:4px; }
.remed-sub   { font-family:'JetBrains Mono',monospace; font-size:0.65rem; color:#5A6A82; margin-bottom:10px; }
.remed-body  { font-family:'Inter',sans-serif; font-size:0.83rem; color:#A8B8CC; line-height:1.7; }
.remed-step  { display:flex; gap:10px; align-items:flex-start; margin:4px 0; }
.remed-num   { font-family:'JetBrains Mono',monospace; font-size:0.7rem; color:#60A5FA;
               font-weight:600; flex-shrink:0; padding-top:2px; }

/* Feature explain bar */
.feat-bar-wrap { margin:6px 0; }
.feat-label { font-family:'JetBrains Mono',monospace; font-size:0.68rem; color:#5A6A82; margin-bottom:2px; }
.feat-bar-bg { background:#0F1117; border-radius:4px; height:10px; overflow:hidden; }
.feat-bar-fill { height:100%; border-radius:4px; }

.div { border:none; border-top:1px solid #1E2535; margin:18px 0; }

.stButton > button {
    background:#1E3A5F !important; border:1px solid #2A5280 !important;
    color:#93C5FD !important; font-family:'Inter',sans-serif !important;
    font-weight:600 !important; border-radius:6px !important;
}
.stTabs [data-baseweb="tab"] { font-family:'Inter',sans-serif !important; color:#3D5068 !important; }
.stTabs [aria-selected="true"] { color:#F0F4F8 !important; font-weight:700 !important; }
.stTabs [data-baseweb="tab-highlight"] { background:#60A5FA !important; }
.streamlit-expanderHeader {
    background:#161B27 !important; border:1px solid #252D3D !important;
    border-radius:6px !important; color:#C9D4E0 !important;
}
.streamlit-expanderContent {
    background:#0F1117 !important; border:1px solid #252D3D !important; border-top:none !important;
}
div[data-testid="stAlert"] {
    background:#14352A !important; border-color:#1A5038 !important;
    color:#34D399 !important; border-radius:6px !important;
}
code { background:#0F1117 !important; color:#7A9AB8 !important; }
</style>
""", unsafe_allow_html=True)

plt.rcParams.update({
    'figure.facecolor':'#161B27','axes.facecolor':'#161B27',
    'axes.edgecolor':'#252D3D','axes.labelcolor':'#5A6A82',
    'xtick.color':'#3D5068','ytick.color':'#3D5068',
    'text.color':'#C9D4E0','grid.color':'#1E2535',
    'grid.linestyle':'--','grid.alpha':0.5,'font.family':'monospace',
})

FEATURES = ['dur','spkts','dpkts','sbytes','dbytes','rate','sload','dload',
            'sinpkt','dinpkt','sjit','djit','swin','dwin','smean','dmean',
            'proto','service','state']

# ── Attack inference from flow features ──────────────────────────────────────
def infer_attack_types(attack_flows_df):
    """
    Infer likely attack categories from flow feature patterns.
    Returns list of (attack_type, count, confidence, css_class) tuples.
    """
    results = {}
    for _, row in attack_flows_df.iterrows():
        # Port scan: many packets, small bytes, short duration
        if row['spkts'] > 50 and row['sbytes'] < 500 and row['dur'] < 1.0:
            results['Port Scan'] = results.get('Port Scan', 0) + 1
        # DoS/DDoS: very high rate or load
        elif row['rate'] > 10000 or row['sload'] > 500000:
            results['DoS / DDoS'] = results.get('DoS / DDoS', 0) + 1
        # Brute force: many small packets, high inter-packet time variance
        elif row['spkts'] > 20 and row['sbytes'] < 2000 and row['sjit'] > 100:
            results['Brute Force'] = results.get('Brute Force', 0) + 1
        # Data exfiltration: large bytes, low packet count
        elif row['sbytes'] > 100000 and row['spkts'] < 20:
            results['Data Exfiltration'] = results.get('Data Exfiltration', 0) + 1
        # Generic attack
        else:
            results['Generic Attack'] = results.get('Generic Attack', 0) + 1

    total = sum(results.values())
    return [(k, v, round(v/total*100,1)) for k,v in sorted(results.items(), key=lambda x:-x[1])]

REMEDIATION = {
    'Port Scan': {
        'css': 'remed-scan',
        'signals': 'High packet count · Low byte volume · Short duration · Sequential port access',
        'desc': 'Attacker is mapping open ports and services on your network. Typically a precursor to targeted exploitation.',
        'steps': [
            'Block source IP at perimeter firewall immediately',
            'Enable port scan detection on IDS/IPS (threshold: >20 ports/sec)',
            'Review firewall rules — close all non-essential ports',
            'Deploy honeypot ports to detect and fingerprint scanners',
            'Log and alert on half-open (SYN) connections from single sources',
        ]
    },
    'DoS / DDoS': {
        'css': 'remed-dos',
        'signals': 'Extremely high transfer rate · High source load · Abnormal packet burst',
        'desc': 'Denial of Service attack flooding your network. Aim is to exhaust bandwidth or overwhelm servers.',
        'steps': [
            'Rate-limit incoming traffic per source IP at border router',
            'Enable SYN cookie protection to handle SYN floods',
            'Contact upstream ISP for traffic scrubbing if volumetric',
            'Deploy CDN or DDoS mitigation service (Cloudflare, AWS Shield)',
            'Identify and null-route attacking source ranges',
            'Monitor CPU/memory on affected servers — prepare to failover',
        ]
    },
    'Brute Force': {
        'css': 'remed-brute',
        'signals': 'Repeated small packets · High jitter · Many source packets · Auth service port',
        'desc': 'Attacker is attempting to guess credentials via repeated authentication attempts.',
        'steps': [
            'Enforce account lockout after 5 failed attempts',
            'Deploy MFA on all internet-facing services immediately',
            'Block source IP after threshold breach (fail2ban or equivalent)',
            'Audit logs for successful logins from attacking IP',
            'Rotate credentials for targeted accounts',
            'Move SSH/RDP to non-standard ports to reduce automated attacks',
        ]
    },
    'Data Exfiltration': {
        'css': 'remed-exfil',
        'signals': 'Large outbound byte volume · Low packet count · Unusual destination',
        'desc': 'Large data transfer to external destination — possible data theft in progress.',
        'steps': [
            'Block destination IP at firewall immediately',
            'Isolate source host from network pending investigation',
            'Capture and preserve network traffic for forensic analysis',
            'Audit what data the source host had access to',
            'Check for C2 callback patterns from the same host',
            'Notify security team and initiate incident response procedure',
            'Review DLP policies — set outbound transfer size alerts',
        ]
    },
    'Generic Attack': {
        'css': 'remed-generic',
        'signals': 'Anomalous flow features — does not match specific attack signature',
        'desc': 'Flow classified as malicious by ML model but does not match a specific attack pattern. Requires manual investigation.',
        'steps': [
            'Capture full packet trace from source IP for manual analysis',
            'Cross-reference source IP against threat intelligence feeds',
            'Check SIEM for correlated events from same source',
            'Apply temporary block on source and monitor for recurrence',
            'Escalate to Tier 2 analyst if activity persists',
        ]
    }
}

FEATURE_EXPLAIN = {
    'Port Scan':        {'spkts':0.9,'dpkts':0.7,'dur':0.8,'rate':0.5,'sbytes':0.3,'dbytes':0.2,'sload':0.2,'dload':0.1},
    'DoS / DDoS':       {'rate':0.95,'sload':0.9,'sbytes':0.7,'spkts':0.8,'dur':0.5,'dpkts':0.3,'dbytes':0.2,'dload':0.3},
    'Brute Force':      {'spkts':0.85,'sjit':0.9,'sbytes':0.6,'dpkts':0.5,'dur':0.4,'rate':0.4,'sload':0.3,'dbytes':0.2},
    'Data Exfiltration':{'sbytes':0.95,'dbytes':0.7,'spkts':0.3,'dpkts':0.4,'rate':0.5,'sload':0.6,'dur':0.6,'dload':0.4},
    'Generic Attack':   {'dur':0.5,'rate':0.5,'sbytes':0.5,'dbytes':0.5,'spkts':0.5,'dpkts':0.5,'sload':0.4,'dload':0.4},
}

@st.cache_resource(show_spinner=False)
def load_models():
    return joblib.load("models/random_forest.pkl"), joblib.load("models/xgboost.pkl")

@st.cache_data(show_spinner=False)
def load_data():
    return pd.read_csv("data/merged_dataset.csv")

@st.cache_data(show_spinner=False)
def get_test_split(_rf, _xgb, df):
    X=df[FEATURES]; y=df["Label"]
    _,X_t,_,y_t = train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)
    return y_t, _rf.predict(X_t), _xgb.predict(X_t)

@st.cache_data(show_spinner=False)
def get_cross_dataset_acc(_rf, df):
    unsw=df[df['source']==0]; cicids=df[df['source']==1]
    return (round(accuracy_score(unsw['Label'],  _rf.predict(unsw[FEATURES]))*100,2),
            round(accuracy_score(cicids['Label'],_rf.predict(cicids[FEATURES]))*100,2))

rf, xgb = load_models()
df      = load_data()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sb-header">NIDS Platform</div>', unsafe_allow_html=True)
    st.markdown('<div class="sb-sub">Cross-Dataset Intrusion Detection · v3.1</div>', unsafe_allow_html=True)
    st.markdown('<hr style="border:none;border-top:1px solid #1E2535;margin:12px 0 4px;">', unsafe_allow_html=True)

    st.markdown('<div class="sb-section">Navigate</div>', unsafe_allow_html=True)
    page = st.radio("Navigation", ["Overview","Live Stream","Manual Classifier","Dataset Intel"],
                    label_visibility="collapsed")

    st.markdown('<hr style="border:none;border-top:1px solid #1E2535;margin:12px 0 4px;">', unsafe_allow_html=True)
    st.markdown('<div class="sb-section">Models</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="sb-stat"><b>Random Forest</b><br>99.16% aggregate · selected</div>
    <div class="sb-stat"><b>XGBoost</b><br>98.69% aggregate</div>
    <div class="sb-stat"><b>UNSW-NB15 only</b><br>97.28% (real captures)</div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sb-section" style="margin-top:14px;">Datasets</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="sb-stat"><b>UNSW-NB15</b><br>257,673 samples · real</div>
    <div class="sb-stat"><b>CICIDS2017</b><br>2,520,798 samples · lab</div>
    <div class="sb-stat"><b>Merged</b><br>2,778,471 · common schema</div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sb-section" style="margin-top:14px;">Top Features</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="sb-stat"><b>dwin</b> · 11.3%</div>
    <div class="sb-stat"><b>swin</b> · 10.9%</div>
    <div class="sb-stat"><b>djit</b> · 9.9%</div>
    <div class="sb-stat"><b>dmean</b> · 9.3%</div>
    <div class="sb-stat"><b>dbytes</b> · 6.6%</div>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
if page == "Overview":
    st.markdown("""
    <div style="font-family:'Inter',sans-serif;font-size:1.5rem;font-weight:700;color:#F0F4F8;letter-spacing:-0.5px;">
        Threat Intelligence Dashboard</div>
    <div style="font-family:'JetBrains Mono',monospace;font-size:0.72rem;color:#3D5068;margin-top:4px;">
        Cross-Dataset Network Intrusion Detection · UNSW-NB15 + CICIDS2017 · 2,778,471 samples</div>
    """, unsafe_allow_html=True)
    st.markdown('<hr class="div">', unsafe_allow_html=True)

    with st.spinner("Loading evaluations..."):
        y_test, rf_preds, xgb_preds = get_test_split(rf, xgb, df)
        acc_unsw, acc_cicids = get_cross_dataset_acc(rf, df)

    rf_acc = round(accuracy_score(y_test,rf_preds)*100,2)
    xgb_acc= round(accuracy_score(y_test,xgb_preds)*100,2)
    rf_rep = classification_report(y_test,rf_preds, output_dict=True)
    xgb_rep= classification_report(y_test,xgb_preds,output_dict=True)

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("RF Aggregate Accuracy", f"{rf_acc}%")
    c2.metric("XGBoost Accuracy",      f"{xgb_acc}%")
    c3.metric("Test Samples",          "555,694")
    c4.metric("RF Attack Recall",      f"{round(rf_rep['1']['recall']*100,1)}%")

    st.markdown('<hr class="div">', unsafe_allow_html=True)
    st.markdown("### Cross-Dataset Generalization")
    st.markdown("""
    <div class="warn-box">
        <b>Accuracy gap explained:</b> CICIDS2017 uses lab-generated traffic with highly separable 
        flow patterns — RF achieves near-perfect accuracy trivially. UNSW-NB15 uses real network 
        captures with more realistic attack distributions. The 2.63% gap is the cross-dataset 
        generalization penalty and is expected for merged corpus models.
    </div>
    """, unsafe_allow_html=True)

    g1,g2,g3 = st.columns(3)
    g1.metric("RF on UNSW-NB15",   f"{acc_unsw}%",  "Real network captures")
    g2.metric("RF on CICIDS2017",   f"{acc_cicids}%","Lab-generated traffic")
    g3.metric("Generalization Gap", f"{round(acc_cicids-acc_unsw,2)}%","CICIDS advantage")

    st.markdown('<hr class="div">', unsafe_allow_html=True)
    st.markdown("### Confusion Matrices")
    col1,col2 = st.columns(2)
    for col,preds,name,color in [
        (col1,rf_preds, "Random Forest","#60A5FA"),
        (col2,xgb_preds,"XGBoost",      "#A78BFA")
    ]:
        cm=confusion_matrix(y_test,preds)
        fig,ax=plt.subplots(figsize=(5,4))
        sns.heatmap(cm,annot=True,fmt=",d",
                    cmap=sns.light_palette(color,as_cmap=True,n_colors=8),
                    ax=ax,xticklabels=["Benign","Attack"],
                    yticklabels=["Benign","Attack"],
                    linewidths=0.5,linecolor='#0F1117',
                    annot_kws={"size":13,"weight":"bold","color":"#0F1117"})
        ax.set_title(name,fontsize=11,color=color,pad=12,fontweight='bold')
        ax.set_xlabel("Predicted",labelpad=8); ax.set_ylabel("Actual",labelpad=8)
        plt.tight_layout(); col.pyplot(fig,use_container_width=True); plt.close()

    st.markdown('<hr class="div">', unsafe_allow_html=True)
    st.markdown("### Top Feature Importances (Random Forest)")
    feat_imp=pd.Series(rf.feature_importances_,index=FEATURES).sort_values(ascending=False).head(10)
    fig,ax=plt.subplots(figsize=(10,3.5))
    colors=["#EF4444" if i<3 else "#60A5FA" for i in range(len(feat_imp))]
    bars=ax.bar(feat_imp.index,feat_imp.values*100,color=colors,edgecolor='#0F1117',linewidth=0.8)
    for bar in bars:
        ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+0.1,
                f'{bar.get_height():.1f}%',ha='center',va='bottom',
                fontsize=8,color='#7A9AB8',fontweight='bold')
    ax.set_ylabel("Importance %",fontsize=8)
    ax.set_title("Top 10 Features — TCP window sizes and jitter dominate",
                 fontsize=9,color="#7A9AB8",pad=8)
    ax.yaxis.grid(True); ax.set_axisbelow(True)
    plt.tight_layout(); st.pyplot(fig,use_container_width=True); plt.close()

    st.markdown("""
    <div class="info-box">
        <b>Why TCP window (dwin/swin) and jitter dominate:</b> Attack traffic typically has 
        abnormal TCP window sizes (0 or max) and irregular inter-packet timing. These are 
        strong discriminators in both datasets. <b>Known limitation:</b> an adversary aware 
        of these features could mimic normal window behavior to evade detection — this is the 
        core limitation of flow-level ML-based NIDS.
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<hr class="div">', unsafe_allow_html=True)
    st.markdown("### Precision / Recall / F1 Comparison")
    metrics={
        'Metric':['Precision\n(Benign)','Recall\n(Benign)','F1\n(Benign)',
                  'Precision\n(Attack)','Recall\n(Attack)','F1\n(Attack)'],
        'Random Forest':[rf_rep['0']['precision'], rf_rep['0']['recall'], rf_rep['0']['f1-score'],
                         rf_rep['1']['precision'], rf_rep['1']['recall'], rf_rep['1']['f1-score']],
        'XGBoost':      [xgb_rep['0']['precision'],xgb_rep['0']['recall'],xgb_rep['0']['f1-score'],
                         xgb_rep['1']['precision'],xgb_rep['1']['recall'],xgb_rep['1']['f1-score']],
    }
    mdf=pd.DataFrame(metrics)
    fig,ax=plt.subplots(figsize=(11,4))
    x=np.arange(len(mdf['Metric'])); w=0.35
    b1=ax.bar(x-w/2,mdf['Random Forest'],w,color="#60A5FA",alpha=0.85,label='RF',zorder=3)
    b2=ax.bar(x+w/2,mdf['XGBoost'],      w,color="#A78BFA",alpha=0.85,label='XGB',zorder=3)
    ax.set_xticks(x); ax.set_xticklabels(mdf['Metric'],fontsize=8)
    ax.set_ylim(0.88,1.02); ax.yaxis.grid(True,zorder=0); ax.set_axisbelow(True)
    for bar in b1: ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+0.002,f'{bar.get_height():.3f}',ha='center',va='bottom',fontsize=7.5,color="#60A5FA",fontweight='bold')
    for bar in b2: ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+0.002,f'{bar.get_height():.3f}',ha='center',va='bottom',fontsize=7.5,color="#A78BFA",fontweight='bold')
    ax.legend(facecolor='#161B27',edgecolor='#252D3D',labelcolor='#C9D4E0',fontsize=9)
    plt.tight_layout(); st.pyplot(fig,use_container_width=True); plt.close()

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — LIVE STREAM
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Live Stream":
    st.markdown("""
    <div style="font-family:'Inter',sans-serif;font-size:1.5rem;font-weight:700;color:#F0F4F8;">
        Live Traffic Stream</div>
    <div style="font-family:'JetBrains Mono',monospace;font-size:0.72rem;color:#3D5068;margin-top:4px;">
        Simulated real-time flow classification · RF + XGBoost ensemble · sampled from test set</div>
    """, unsafe_allow_html=True)
    st.markdown('<hr class="div">', unsafe_allow_html=True)

    col_a,col_b,col_c = st.columns([1,1,1])
    with col_a:
        speed=st.select_slider("Speed",options=[0.05,0.1,0.2,0.5,1.0],
                               value=0.2,format_func=lambda x:f"{x}s/flow")
    with col_b:
        n_flows=st.selectbox("Flows to simulate",[50,100,200,500],index=1)
    with col_c:
        model_choice=st.selectbox("Model",["Random Forest","XGBoost","Ensemble"])

    start=st.button("▶  Start Stream")

    if start:
        sample=df.sample(n=n_flows,random_state=random.randint(0,9999)).reset_index(drop=True)
        X_s=sample[FEATURES]
        rf_pa=rf.predict(X_s);   rf_proba=rf.predict_proba(X_s)[:,1]
        xgb_pa=xgb.predict(X_s); xgb_proba=xgb.predict_proba(X_s)[:,1]

        if   model_choice=="Random Forest": final_preds=rf_pa;  final_probs=rf_proba
        elif model_choice=="XGBoost":       final_preds=xgb_pa; final_probs=xgb_proba
        else:
            final_probs=(rf_proba+xgb_proba)/2
            final_preds=(final_probs>=0.5).astype(int)

        total_c=0; attack_c=0; benign_c=0
        alert_log=[]; risk_history=[]
        attack_flows=[]

        m1,m2,m3,m4=st.columns(4)
        ph_total=m1.empty(); ph_attack=m2.empty(); ph_benign=m3.empty(); ph_rate=m4.empty()
        st.markdown('<hr class="div">', unsafe_allow_html=True)

        col_feed,col_chart=st.columns([1,1])
        with col_feed:
            st.markdown("**Alert Feed**")
            feed_ph=st.empty()
        with col_chart:
            st.markdown("**Risk Timeline**")
            chart_ph=st.empty()

        st.markdown('<hr class="div">', unsafe_allow_html=True)
        st.markdown("**Current Flow Risk**")
        gauge_ph=st.empty()
        prog_ph=st.empty()

        for i in range(n_flows):
            pred=final_preds[i]; prob=final_probs[i]; risk=round(prob*100,1)
            total_c+=1
            if pred==1:
                attack_c+=1
                attack_flows.append(sample.iloc[i])
            else:
                benign_c+=1
            risk_history.append(risk)

            dot="dot-red" if pred==1 else "dot-green"
            color="#EF4444" if pred==1 else "#22C55E"
            label="ATTACK" if pred==1 else "BENIGN"
            alert_log.insert(0,f"""
            <div class="alert-row">
                <div class="{dot}"></div>
                <span style="color:#3D5068;">#{i+1:04d}</span>
                <span style="color:{color};font-weight:600;">{label}</span>
                <span style="color:#5A6A82;">{risk}%</span>
                <span style="color:#3D5068;font-size:0.65rem;">
                    {int(sample.iloc[i]['sbytes'])}B · {int(sample.iloc[i]['spkts'])}pkts
                </span>
            </div>""")
            if len(alert_log)>12: alert_log=alert_log[:12]

            attack_rate=round(attack_c/total_c*100,1)
            ph_total.metric("Flows Processed",total_c)
            ph_attack.metric("Attacks",attack_c)
            ph_benign.metric("Benign",benign_c)
            ph_rate.metric("Attack Rate",f"{attack_rate}%")

            feed_ph.markdown("".join(alert_log),unsafe_allow_html=True)

            if len(risk_history)>=2:
                fig,ax=plt.subplots(figsize=(6,3))
                ax.fill_between(range(len(risk_history)),risk_history,alpha=0.15,color="#EF4444")
                ax.plot(risk_history,color="#EF4444",linewidth=1.5)
                ax.axhline(y=70,color="#EF4444",linestyle="--",alpha=0.4,linewidth=0.8)
                ax.axhline(y=40,color="#EAB308",linestyle="--",alpha=0.4,linewidth=0.8)
                ax.set_ylim(0,105); ax.set_xlabel("Flow #",fontsize=8); ax.set_ylabel("Risk %",fontsize=8)
                ax.set_title("Attack Probability per Flow",fontsize=9,color="#7A9AB8",pad=8)
                plt.tight_layout(); chart_ph.pyplot(fig,use_container_width=True); plt.close()

            g_color="#EF4444" if risk>=70 else "#EAB308" if risk>=40 else "#22C55E"
            gauge_ph.markdown(f"""
            <div style="background:#161B27;border:1px solid #252D3D;border-radius:8px;padding:14px 18px;">
                <div style="font-family:'JetBrains Mono',monospace;font-size:0.65rem;color:#3D5068;margin-bottom:8px;">
                    Flow #{i+1:04d} · {model_choice}</div>
                <div style="background:#0F1117;border-radius:6px;height:14px;overflow:hidden;">
                    <div style="background:{g_color};width:{risk}%;height:100%;border-radius:6px;"></div>
                </div>
                <div style="font-family:'JetBrains Mono',monospace;font-size:0.8rem;
                            color:{g_color};font-weight:600;margin-top:6px;">{risk}% risk</div>
            </div>""",unsafe_allow_html=True)

            prog_ph.progress((i+1)/n_flows,text=f"Flow {i+1}/{n_flows}")
            time.sleep(speed)

        prog_ph.success(f"Complete · {n_flows} flows · {attack_c} attacks ({attack_rate}%)")

        # ── Stream summary charts ──────────────────────────────────────────
        st.markdown('<hr class="div">', unsafe_allow_html=True)
        st.markdown("### Stream Summary")
        s1,s2,s3=st.columns(3)
        with s1:
            fig,ax=plt.subplots(figsize=(4,4))
            ax.pie([benign_c,attack_c],labels=[f"Benign\n{benign_c}",f"Attack\n{attack_c}"],
                   colors=["#22C55E","#EF4444"],autopct="%1.1f%%",startangle=90,pctdistance=0.78,
                   wedgeprops=dict(width=0.55,edgecolor='#0F1117',linewidth=2))
            ax.set_title("Traffic Breakdown",fontsize=10,color="#7A9AB8",pad=10)
            plt.tight_layout(); st.pyplot(fig,use_container_width=True); plt.close()
        with s2:
            fig,ax=plt.subplots(figsize=(4,4))
            ax.plot(risk_history,color="#EF4444",linewidth=1,alpha=0.7)
            ax.fill_between(range(len(risk_history)),risk_history,alpha=0.15,color="#EF4444")
            ax.axhline(y=70,color="#EF4444",linestyle="--",alpha=0.4,linewidth=0.8)
            ax.set_title("Full Risk Timeline",fontsize=10,color="#7A9AB8",pad=10)
            ax.set_xlabel("Flow #",fontsize=8); ax.set_ylabel("Risk %",fontsize=8)
            plt.tight_layout(); st.pyplot(fig,use_container_width=True); plt.close()
        with s3:
            hist,_=np.histogram(risk_history,bins=[0,20,40,60,70,80,100])
            labels_h=["0-20","20-40","40-60","60-70","70-80","80-100"]
            colors_h=["#22C55E","#22C55E","#EAB308","#EAB308","#EF4444","#EF4444"]
            fig,ax=plt.subplots(figsize=(4,4))
            ax.bar(labels_h,hist,color=colors_h,edgecolor='#0F1117',linewidth=0.8)
            ax.set_title("Risk Distribution",fontsize=10,color="#7A9AB8",pad=10)
            ax.set_xlabel("Risk Band",fontsize=8); ax.set_ylabel("Count",fontsize=8)
            plt.xticks(rotation=30,ha='right',fontsize=7)
            plt.tight_layout(); st.pyplot(fig,use_container_width=True); plt.close()

        # ── Threat analysis + remediation ─────────────────────────────────
        if attack_flows:
            st.markdown('<hr class="div">', unsafe_allow_html=True)
            st.markdown("### Threat Analysis & Remediation")
            st.markdown(f"""
            <div class="info-box">
                Analyzed <b>{len(attack_flows)}</b> detected attack flows. 
                Attack types inferred from flow feature patterns 
                (packet count, byte volume, rate, jitter). 
                Each card below shows likely attack category, detection signals, 
                and actionable remediation steps.
            </div>
            """,unsafe_allow_html=True)

            attack_df=pd.DataFrame(attack_flows)
            attack_types=infer_attack_types(attack_df)

            for attack_type, count, pct in attack_types:
                r=REMEDIATION.get(attack_type, REMEDIATION['Generic Attack'])
                fe=FEATURE_EXPLAIN.get(attack_type, FEATURE_EXPLAIN['Generic Attack'])

                # Feature explanation bars HTML
                feat_bars=""
                for feat,importance in sorted(fe.items(),key=lambda x:-x[1])[:6]:
                    bar_color="#EF4444" if importance>0.8 else "#EAB308" if importance>0.5 else "#60A5FA"
                    feat_bars+=f"""
                    <div class="feat-bar-wrap">
                        <div class="feat-label">{feat} · {int(importance*100)}% influence</div>
                        <div class="feat-bar-bg">
                            <div class="feat-bar-fill" style="width:{int(importance*100)}%;background:{bar_color};"></div>
                        </div>
                    </div>"""

                steps_html="".join([
                    f'<div class="remed-step"><span class="remed-num">{i+1:02d}.</span><span class="remed-body">{step}</span></div>'
                    for i,step in enumerate(r['steps'])
                ])

                with st.expander(f"{attack_type}  ·  {count} flows detected  ·  {pct}% of attacks", expanded=True):
                    left,right=st.columns([1,1])
                    with left:
                        st.markdown(f"""
                        <div class="remed-card {r['css']}">
                            <div class="remed-title">{attack_type}</div>
                            <div class="remed-sub">{count} flows · {pct}% of detected attacks</div>
                            <div class="remed-body" style="margin-bottom:12px;">{r['desc']}</div>
                            <div style="font-family:'JetBrains Mono',monospace;font-size:0.65rem;
                                        color:#3D5068;margin-bottom:6px;text-transform:uppercase;
                                        letter-spacing:1px;">Detection Signals</div>
                            <div class="remed-body" style="font-family:'JetBrains Mono',monospace;
                                        font-size:0.7rem;color:#5A6A82;">{r['signals']}</div>
                        </div>
                        """,unsafe_allow_html=True)

                        st.markdown(f"""
                        <div style="margin-top:12px;">
                            <div style="font-family:'JetBrains Mono',monospace;font-size:0.65rem;
                                        color:#3D5068;text-transform:uppercase;letter-spacing:1px;
                                        margin-bottom:8px;">Key Feature Contributions</div>
                            {feat_bars}
                        </div>
                        """,unsafe_allow_html=True)

                    with right:
                        st.markdown(f"""
                        <div style="font-family:'JetBrains Mono',monospace;font-size:0.65rem;
                                    color:#3D5068;text-transform:uppercase;letter-spacing:1px;
                                    margin-bottom:10px;">Remediation Steps</div>
                        {steps_html}
                        """,unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — MANUAL CLASSIFIER
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Manual Classifier":
    st.markdown("""
    <div style="font-family:'Inter',sans-serif;font-size:1.5rem;font-weight:700;color:#F0F4F8;">
        Manual Flow Classifier</div>
    <div style="font-family:'JetBrains Mono',monospace;font-size:0.72rem;color:#3D5068;margin-top:4px;">
        Input 19 network flow features · RF + XGBoost ensemble verdict</div>
    """,unsafe_allow_html=True)
    st.markdown('<hr class="div">', unsafe_allow_html=True)

    c1,c2,c3=st.columns(3)
    with c1:
        st.markdown("**Flow Metrics**")
        dur=st.number_input("Duration (dur)",value=0.0,format="%.4f")
        spkts=st.number_input("Source Packets",value=0,step=1)
        dpkts=st.number_input("Dest Packets",value=0,step=1)
        sbytes=st.number_input("Source Bytes",value=0,step=1)
        dbytes=st.number_input("Dest Bytes",value=0,step=1)
        rate=st.number_input("Transfer Rate",value=0.0,format="%.4f")
    with c2:
        st.markdown("**Load & Timing**")
        sload=st.number_input("Source Load",value=0.0,format="%.4f")
        dload=st.number_input("Dest Load",value=0.0,format="%.4f")
        sinpkt=st.number_input("Src Inter-Packet (ms)",value=0.0,format="%.4f")
        dinpkt=st.number_input("Dst Inter-Packet (ms)",value=0.0,format="%.4f")
        sjit=st.number_input("Source Jitter",value=0.0,format="%.4f")
        djit=st.number_input("Dest Jitter",value=0.0,format="%.4f")
    with c3:
        st.markdown("**Window & Protocol**")
        swin=st.number_input("Src TCP Window",value=0,step=1)
        dwin=st.number_input("Dst TCP Window",value=0,step=1)
        smean=st.number_input("Src Mean Packet",value=0.0,format="%.4f")
        dmean=st.number_input("Dst Mean Packet",value=0.0,format="%.4f")
        proto=st.number_input("Protocol",value=0,step=1)
        service=st.number_input("Service",value=0,step=1)
        state=st.number_input("State",value=0,step=1)

    if st.button("Classify Flow"):
        inp=pd.DataFrame([[dur,spkts,dpkts,sbytes,dbytes,rate,sload,dload,
                           sinpkt,dinpkt,sjit,djit,swin,dwin,smean,dmean,
                           proto,service,state]],columns=FEATURES)
        rf_pred=rf.predict(inp)[0]; rf_prob=rf.predict_proba(inp)[0]
        xgb_pred=xgb.predict(inp)[0]; xgb_prob=xgb.predict_proba(inp)[0]
        risk=round((rf_prob[1]+xgb_prob[1])/2*100,1)

        st.markdown('<hr class="div">', unsafe_allow_html=True)
        if risk>=70:   cls,lbl="verdict-attack",f"⚠ MALICIOUS — {risk}% Risk"
        elif risk>=40: cls,lbl="verdict-warn",  f"◈ SUSPICIOUS — {risk}% Risk"
        else:          cls,lbl="verdict-benign", f"✓ BENIGN — {risk}% Risk"

        st.markdown(f"""
        <div class="verdict {cls}">
            <div class="verdict-label">Ensemble Verdict</div>
            <div class="verdict-title">{lbl}</div>
        </div>""",unsafe_allow_html=True)

        v1,v2,v3=st.columns(3)
        v1.metric("Risk Score",    f"{risk}%")
        v2.metric("Random Forest", "ATTACK" if rf_pred==1 else "BENIGN", f"{round(max(rf_prob)*100,1)}% conf")
        v3.metric("XGBoost",       "ATTACK" if xgb_pred==1 else "BENIGN",f"{round(max(xgb_prob)*100,1)}% conf")

        # Remediation if attack
        if risk >= 40:
            st.markdown('<hr class="div">', unsafe_allow_html=True)
            st.markdown("### Attack Analysis & Remediation")
            row=inp.iloc[0]
            single_df=pd.DataFrame([row])
            attack_types=infer_attack_types(single_df)
            if attack_types:
                attack_type=attack_types[0][0]
                r=REMEDIATION.get(attack_type,REMEDIATION['Generic Attack'])
                fe=FEATURE_EXPLAIN.get(attack_type,FEATURE_EXPLAIN['Generic Attack'])

                feat_bars=""
                for feat,importance in sorted(fe.items(),key=lambda x:-x[1])[:6]:
                    bar_color="#EF4444" if importance>0.8 else "#EAB308" if importance>0.5 else "#60A5FA"
                    feat_bars+=f"""
                    <div class="feat-bar-wrap">
                        <div class="feat-label">{feat} · {int(importance*100)}% influence</div>
                        <div class="feat-bar-bg">
                            <div class="feat-bar-fill" style="width:{int(importance*100)}%;background:{bar_color};"></div>
                        </div>
                    </div>"""

                steps_html="".join([
                    f'<div class="remed-step"><span class="remed-num">{i+1:02d}.</span><span class="remed-body">{step}</span></div>'
                    for i,step in enumerate(r['steps'])
                ])

                left,right=st.columns([1,1])
                with left:
                    st.markdown(f"""
                    <div class="remed-card {r['css']}">
                        <div class="remed-title">Likely: {attack_type}</div>
                        <div class="remed-sub">{r['signals']}</div>
                        <div class="remed-body">{r['desc']}</div>
                    </div>
                    <div style="margin-top:12px;">{feat_bars}</div>
                    """,unsafe_allow_html=True)
                with right:
                    st.markdown(f"""
                    <div style="font-family:'JetBrains Mono',monospace;font-size:0.65rem;
                                color:#3D5068;text-transform:uppercase;letter-spacing:1px;
                                margin-bottom:10px;">Remediation Steps</div>
                    {steps_html}
                    """,unsafe_allow_html=True)

        fig,ax=plt.subplots(figsize=(7,2.5))
        models=["Random Forest","XGBoost"]; probs=[round(rf_prob[1]*100,1),round(xgb_prob[1]*100,1)]
        bars=ax.barh(models,probs,color=["#60A5FA","#A78BFA"],alpha=0.85,height=0.4)
        ax.set_xlim(0,100)
        ax.axvline(x=70,color="#EF4444",linestyle="--",alpha=0.4,linewidth=0.8)
        ax.axvline(x=40,color="#EAB308",linestyle="--",alpha=0.4,linewidth=0.8)
        for bar,prob in zip(bars,probs):
            ax.text(bar.get_width()+1,bar.get_y()+bar.get_height()/2,
                    f"{prob}%",va='center',fontsize=11,fontweight='bold',color="#C9D4E0")
        ax.set_xlabel("Attack Probability (%)",fontsize=8)
        ax.set_title("Model Confidence",fontsize=9,color="#7A9AB8",pad=8)
        plt.tight_layout(); st.pyplot(fig,use_container_width=True); plt.close()

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — DATASET INTEL
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Dataset Intel":
    st.markdown("""
    <div style="font-family:'Inter',sans-serif;font-size:1.5rem;font-weight:700;color:#F0F4F8;">
        Dataset Intelligence</div>
    <div style="font-family:'JetBrains Mono',monospace;font-size:0.72rem;color:#3D5068;margin-top:4px;">
        Merged corpus · UNSW-NB15 + CICIDS2017 · 2,778,471 samples · Common Feature Schema</div>
    """,unsafe_allow_html=True)
    st.markdown('<hr class="div">', unsafe_allow_html=True)

    total=len(df); benign=int((df["Label"]==0).sum())
    attacks=int((df["Label"]==1).sum())
    unsw_c=int((df["source"]==0).sum()); cicids_c=int((df["source"]==1).sum())

    c1,c2,c3,c4=st.columns(4)
    c1.metric("Total Samples",f"{total:,}")
    c2.metric("Benign Flows", f"{benign:,}")
    c3.metric("Attack Flows", f"{attacks:,}")
    c4.metric("Attack Ratio", f"{round(attacks/total*100,1)}%")

    st.markdown('<hr class="div">', unsafe_allow_html=True)
    col1,col2=st.columns(2)
    with col1:
        fig,ax=plt.subplots(figsize=(5,5))
        ax.pie([benign,attacks],labels=[f"Benign\n{benign:,}",f"Attack\n{attacks:,}"],
               colors=["#22C55E","#EF4444"],autopct="%1.1f%%",startangle=90,pctdistance=0.78,
               wedgeprops=dict(width=0.55,edgecolor='#0F1117',linewidth=2))
        ax.text(0,0,f"{round(attacks/total*100,1)}%\nattacks",ha='center',va='center',
                fontsize=13,fontweight='bold',color="#EF4444",fontfamily='monospace')
        ax.set_title("Traffic Distribution",fontsize=10,color="#7A9AB8",pad=10)
        plt.tight_layout(); st.pyplot(fig,use_container_width=True); plt.close()
    with col2:
        fig,ax=plt.subplots(figsize=(5,5))
        ax.pie([cicids_c,unsw_c],labels=[f"CICIDS2017\n{cicids_c:,}",f"UNSW-NB15\n{unsw_c:,}"],
               colors=["#60A5FA","#A78BFA"],autopct="%1.1f%%",startangle=90,pctdistance=0.78,
               wedgeprops=dict(width=0.55,edgecolor='#0F1117',linewidth=2))
        ax.text(0,0,"2 datasets\nmerged",ha='center',va='center',
                fontsize=10,fontweight='bold',color="#7A9AB8",fontfamily='monospace')
        ax.set_title("Dataset Source Split",fontsize=10,color="#7A9AB8",pad=10)
        plt.tight_layout(); st.pyplot(fig,use_container_width=True); plt.close()

    st.markdown('<hr class="div">', unsafe_allow_html=True)
    st.markdown("### Feature Statistics")
    st.dataframe(df[FEATURES].describe().round(3),use_container_width=True,height=280)

    st.markdown('<hr class="div">', unsafe_allow_html=True)
    st.markdown("### Feature Correlation Heatmap")
    with st.spinner("Computing..."):
        corr=df[FEATURES].sample(n=10000,random_state=42).corr()
    fig,ax=plt.subplots(figsize=(12,9))
    sns.heatmap(corr,ax=ax,cmap="Blues",center=0,linewidths=0.3,
                linecolor='#0F1117',annot=False,square=True)
    ax.set_title("Feature Correlation Matrix (10K sample)",fontsize=10,color="#7A9AB8",pad=10)
    plt.tight_layout(); st.pyplot(fig,use_container_width=True); plt.close()

st.markdown('<hr class="div">', unsafe_allow_html=True)
st.markdown("""
<p style="text-align:center;font-family:'JetBrains Mono',monospace;font-size:0.62rem;color:#2D3A52;">
NIDS Platform · UNSW-NB15 + CICIDS2017 · RF 99.16% aggregate · 97.28% on real captures · Python & Streamlit
</p>""",unsafe_allow_html=True)