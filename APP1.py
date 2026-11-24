import streamlit as st
import pandas as pd
import joblib
import base64
from pathlib import Path
from datetime import datetime

# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="Bankruptcy Prevention System",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------
# Paths & Helper Functions
# ---------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "bankruptcy_model.pkl"
IMAGES_DIR = BASE_DIR / "images"

def get_image_path(image_name: str):
    """Finds image_name with .png, .jpg, or .jpeg extension inside images/ folder."""
    for ext in [".png", ".jpg", ".jpeg", ".webp"]:
        candidate = IMAGES_DIR / f"{image_name}{ext}"
        if candidate.exists():
            return candidate
    return None

@st.cache_data
def get_base64_image(image_path_str: str):
    """Encodes local image to Base64 and caches it so page switches stay instant."""
    with open(image_path_str, "rb") as f:
        return base64.b64encode(f.read()).decode()

def set_page_background(image_name: str, overlay_opacity: float = 0.80):
    """Sets a full-screen background image with frosted white overlay, black sidebar, and high-contrast text."""
    img_path = get_image_path(image_name)
    bg_css = ""
    if img_path:
        b64_str = get_base64_image(str(img_path))
        ext = img_path.suffix.lstrip(".").lower()
        if ext == "jpg":
            ext = "jpeg"
        bg_css = f"""
background-image: linear-gradient(
    rgba(255, 255, 255, {overlay_opacity}),
    rgba(255, 255, 255, {overlay_opacity})
), url("data:image/{ext};base64,{b64_str}");
background-size: cover;
background-position: center;
background-repeat: no-repeat;
background-attachment: fixed;
"""
    else:
        bg_css = "background-color: #f8fafc;"

    st.markdown(f"""
<style>
.stApp {{
    {bg_css}
    color: #0f172a !important;
}}

/* Force dark high-contrast text across headings, labels, markdown, and metrics in the main area */
.stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6 {{
    color: #090d16 !important;
    font-weight: 700 !important;
}}
.stApp p, .stApp span, .stApp label, .stApp li, .stApp div[data-testid="stMarkdownContainer"] {{
    color: #111827 !important;
}}
.stApp small, .stApp .stCaption, [data-testid="stCaptionContainer"] p {{
    color: #334155 !important;
    font-weight: 500 !important;
}}

/* Metric labels & numbers */
[data-testid="stMetricLabel"] p {{
    color: #1e293b !important;
    font-weight: 600 !important;
}}
[data-testid="stMetricValue"] div {{
    color: #090d16 !important;
    font-weight: 800 !important;
}}

/* Frosted white containers & cards in main area */
div[data-testid="stVerticalBlockBorderWrapper"],
div[data-testid="stMetric"] {{
    background-color: rgba(255, 255, 255, 0.93) !important;
    border: 1px solid #cbd5e1 !important;
    border-radius: 12px;
    padding: 14px 18px;
    box-shadow: 0 3px 12px rgba(15, 23, 42, 0.08);
}}

/* Deep BLACK Sidebar with White/Silver Text */
section[data-testid="stSidebar"] {{
    background: linear-gradient(180deg, #090d16 0%, #020205 100%) !important;
    border-right: 1px solid #1e293b;
}}
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3,
section[data-testid="stSidebar"] h4,
section[data-testid="stSidebar"] h5,
section[data-testid="stSidebar"] h6,
section[data-testid="stSidebar"] div[data-testid="stMarkdownContainer"] h2 {{
    color: #ffffff !important;
    text-shadow: 0 2px 4px rgba(0,0,0,0.5);
}}
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] div[data-testid="stMarkdownContainer"] {{
    color: #f1f5f9 !important;
}}
section[data-testid="stSidebar"] .stCaption,
section[data-testid="stSidebar"] [data-testid="stCaptionContainer"] p {{
    color: #94a3b8 !important;
    font-weight: 500 !important;
}}
section[data-testid="stSidebar"] .stDivider {{
    border-color: #334155 !important;
}}

/* Black Sidebar radio buttons/Navigation style */
section[data-testid="stSidebar"] div[data-baseweb="radio"] label {{
    background: rgba(255, 255, 255, 0.05);
    padding: 8px 12px;
    border-radius: 8px;
    margin-bottom: 6px;
    border: 1px solid rgba(255, 255, 255, 0.05);
    transition: all 0.2s ease;
}}
section[data-testid="stSidebar"] div[data-baseweb="radio"] label:hover {{
    background: rgba(255, 255, 255, 0.12) !important;
    border-color: rgba(255, 255, 255, 0.2);
}}
section[data-testid="stSidebar"] div[data-baseweb="radio"] div[data-testid="stMarkdownContainer"] p {{
    color: #ffffff !important;
    font-weight: 500 !important;
}}

/* Input boxes & selectboxes readable light theme in main contents */
div[data-baseweb="select"] > div,
div[data-baseweb="input"] > div {{
    background-color: #ffffff !important;
    color: #0f172a !important;
    border-color: #94a3b8 !important;
}}
div[data-baseweb="select"] span {{
    color: #0f172a !important;
}}

/* Keep Hero Banner text crisp white on dark gradient */
.hero-banner {{
    background: linear-gradient(135deg, rgba(15, 32, 39, 0.95) 0%, rgba(32, 58, 67, 0.95) 50%, rgba(44, 83, 100, 0.95) 100%);
    padding: 28px 32px;
    border-radius: 14px;
    margin-bottom: 18px;
    box-shadow: 0 4px 15px rgba(0,0,0,0.20);
}}
.hero-banner h1 {{
    color: #ffffff !important;
    margin-bottom: 6px;
    font-size: 2.1rem;
}}
.hero-banner p {{
    color: #e2e8f0 !important;
    font-size: 1.05rem;
    margin-bottom: 0;
}}

/* Keep Primary Button text white */
button[kind="primary"] p, button[kind="primary"] span {{
    color: #ffffff !important;
    font-weight: 600 !important;
}}
/* Download button dark text on white */
div[data-testid="stDownloadButton"] button {{
    background-color: #ffffff !important;
    border: 1px solid #64748b !important;
}}
div[data-testid="stDownloadButton"] button p {{
    color: #0f172a !important;
    font-weight: 600 !important;
}}

/* Custom high-impact WHITE TEXT result cards */
.white-alert-container {{
    border-radius: 12px;
    padding: 20px 24px;
    margin-bottom: 18px;
    box-shadow: 0 4px 14px rgba(0,0,0,0.18);
}}
.white-alert-container h3, .white-alert-container p, .white-alert-container span, .white-alert-container strong {{
    color: #ffffff !important;
    text-shadow: 0 1px 3px rgba(0,0,0,0.3) !important;
}}

/* Dark High-Contrast Visual Wrapper for Observations, Upload Guide & Quick Guide */
.dark-card-container {{
    background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%) !important;
    border: 1.5px solid #334155 !important;
    border-radius: 12px !important;
    padding: 20px 24px !important;
    box-shadow: 0 4px 14px rgba(0,0,0,0.25) !important;
    margin-bottom: 18px !important;
}}
.dark-card-container h3,
.dark-card-container h4,
.dark-card-container h5,
.dark-card-container p,
.dark-card-container span,
.dark-card-container strong,
.dark-card-container li,
.dark-card-container ul {{
    color: #ffffff !important;
    text-shadow: 0 1px 2px rgba(0,0,0,0.4) !important;
}}
.dark-card-container code {{
    background-color: rgba(255, 255, 255, 0.15) !important;
    color: #fde047 !important;
    padding: 2px 6px;
    border-radius: 4px;
}}
.dark-card-container .warning-header {{
    color: #fbcfe8 !important;
    font-weight: 700 !important;
}}
.dark-card-container .strength-header {{
    color: #a7f3d0 !important;
    font-weight: 700 !important;
}}
.dark-card-container .info-highlight {{
    color: #e0f2fe !important;
    font-size: 0.95rem;
    line-height: 1.5;
    margin-bottom: 12px !important;
}}

/* Dark card & 100% WHITE text for st.file_uploader */
[data-testid="stFileUploader"] {{
    background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%) !important;
    border: 1.5px solid #334155 !important;
    border-radius: 0 0 12px 12px !important;
    padding: 18px !important;
}}
[data-testid="stFileUploader"] section {{
    background-color: rgba(255, 255, 255, 0.06) !important;
    border: 1.5px dashed #64748b !important;
    border-radius: 10px !important;
}}
[data-testid="stFileUploader"] span,
[data-testid="stFileUploader"] small,
[data-testid="stFileUploader"] p,
[data-testid="stFileUploader"] div,
[data-testid="stFileUploader"] label,
[data-testid="stFileUploader"] svg {{
    color: #ffffff !important;
    fill: #ffffff !important;
}}
[data-testid="stFileUploader"] button {{
    background-color: #2563eb !important;
    color: #ffffff !important;
    border: none !important;
    font-weight: 600 !important;
}}
[data-testid="stFileUploader"] button span,
[data-testid="stFileUploader"] button p {{
    color: #ffffff !important;
}}
</style>
""", unsafe_allow_html=True)

def show_result_banner(title: str, message: str, is_risk: bool):
    """Renders a custom high-contrast alert card with 100% white text on a colorful background."""
    bg_color = "linear-gradient(135deg, #b91c1c 0%, #991b1b 100%)" if is_risk else "linear-gradient(135deg, #15803d 0%, #166534 100%)"
    border_color = "#991b1b" if is_risk else "#166534"
    html = (
        f'<div class="white-alert-container" style="background: {bg_color}; border: 1.5px solid {border_color};">'
        f'<h3 style="margin: 0 0 8px 0; font-size: 1.35rem; font-weight: 700; color: #ffffff !important;">{title}</h3>'
        f'<p style="margin: 0; font-size: 1.05rem; font-weight: 500; line-height: 1.45; color: #ffffff !important;">{message}</p>'
        f'</div>'
    )
    st.markdown(html, unsafe_allow_html=True)

def render_pie_chart(non_bankruptcy_pct: float, bankruptcy_pct: float, title: str = "Risk Proportion Split"):
    """Renders a clean Donut/Pie chart using pure CSS conic-gradient (no plotly/pyplot required)."""
    safe_angle = max(0.0, min(100.0, non_bankruptcy_pct))
    html_chart = (
        f'<div style="background:rgba(255,255,255,0.95); border:1px solid #cbd5e1; border-radius:12px; padding:18px; text-align:center;">'
        f'<div style="font-weight:700; font-size:1.05rem; color:#090d16; margin-bottom:14px;">{title}</div>'
        f'<div style="width:190px; height:190px; border-radius:50%; margin:0 auto 16px auto; background:conic-gradient(#16A34A 0% {safe_angle:.2f}%, #DC2626 {safe_angle:.2f}% 100%); display:flex; align-items:center; justify-content:center;">'
        f'<div style="width:105px; height:105px; background:#ffffff; border-radius:50%; display:flex; flex-direction:column; align-items:center; justify-content:center; box-shadow:inset 0 1px 4px rgba(0,0,0,0.10);">'
        f'<span style="font-size:0.75rem; font-weight:600; color:#334155;">Bankruptcy</span>'
        f'<span style="font-size:1.15rem; font-weight:800; color:#DC2626;">{bankruptcy_pct:.1f}%</span>'
        f'</div></div>'
        f'<div style="display:flex; justify-content:center; gap:20px; font-size:0.9rem; font-weight:600; color:#0f172a;">'
        f'<span><span style="display:inline-block; width:12px; height:12px; background:#16A34A; border-radius:2px; margin-right:6px;"></span>Non-Bankruptcy ({non_bankruptcy_pct:.1f}%)</span>'
        f'<span><span style="display:inline-block; width:12px; height:12px; background:#DC2626; border-radius:2px; margin-right:6px;"></span>Bankruptcy ({bankruptcy_pct:.1f}%)</span>'
        f'</div></div>'
    )
    st.markdown(html_chart, unsafe_allow_html=True)

# ---------------------------------------------------------
# Load Trained Model
# ---------------------------------------------------------
@st.cache_resource
def load_model(model_path: Path):
    return joblib.load(model_path)

try:
    model = load_model(MODEL_PATH)
except FileNotFoundError:
    st.error(f"Model file not found at `{MODEL_PATH}`. Please check your folder path.")
    st.stop()

# ---------------------------------------------------------
# Initialize Session State for History
# ---------------------------------------------------------
if "history" not in st.session_state:
    st.session_state["history"] = []

FEATURE_COLS = [
    "industrial_risk",
    "management_risk",
    "financial_flexibility",
    "credibility",
    "competitiveness",
    "operating_risk"
]

LEVEL_LABELS = {0.0: "0.0 (Low)", 0.5: "0.5 (Medium)", 1.0: "1.0 (High)"}

# ---------------------------------------------------------
# Sidebar Navigation
# ---------------------------------------------------------
with st.sidebar:
    st.markdown("<h2 style='text-align:center; margin-top:4px;'>🏦 FinGuard AI</h2>", unsafe_allow_html=True)
    st.caption("Corporate Solvency & Bankruptcy Risk Intelligence")
    st.divider()

    page = st.radio(
        "Navigation",
        [
            "🏠 Home",
            "🔍 Predict Risk",
            "📂 Batch CSV Upload",
            "🕒 Prediction History"
        ]
    )

    st.divider()
    st.caption("Model Encoding: `0 = Bankruptcy` | `1 = Non-Bankruptcy`")

# ---------------------------------------------------------
# PAGE 1: HOME (Full-Page bank.png Background)
# ---------------------------------------------------------
if page == "🏠 Home":
    set_page_background("bank", overlay_opacity=0.78)

    st.markdown(
        '<div class="hero-banner">'
        '<h1>🏦 Corporate Bankruptcy Prevention System</h1>'
        '<p>Early-warning financial risk assessment powered by Machine Learning across corporate capital, liquidity, governance, and market competitiveness indicators.</p>'
        '</div>',
        unsafe_allow_html=True
    )

    m1, m2, m3 = st.columns(3)
    m1.metric("Risk Indicators", "6 Core Factors")
    m2.metric("Scale Levels", "0.0 | 0.5 | 1.0")
    m3.metric("Assessment Modes", "Single & Batch")

    st.divider()

    col_left, col_right = st.columns([1.3, 1])

    with col_left:
        with st.container(border=True):
            st.subheader("📌 How the System Evaluates Risk")
            st.write(
                "The predictive model analyzes **6 core qualitative financial and operational indicators** "
                "scored at three levels: **0.0 (Low)**, **0.5 (Medium)**, and **1.0 (High)**."
            )
            indicators_df = pd.DataFrame({
                "Indicator": [
                    "Industrial Risk",
                    "Management Risk",
                    "Financial Flexibility",
                    "Credibility",
                    "Competitiveness",
                    "Operating Risk"
                ],
                "Category": [
                    "External Threat",
                    "Governance Threat",
                    "Capital Strength",
                    "Credit Strength",
                    "Market Strength",
                    "Operational Threat"
                ],
                "Impact on Bankruptcy": [
                    "Higher (1.0) increases bankruptcy risk",
                    "Higher (1.0) increases bankruptcy risk",
                    "Lower (0.0) strongly increases bankruptcy risk",
                    "Lower (0.0) strongly increases bankruptcy risk",
                    "Lower (0.0) strongly increases bankruptcy risk",
                    "Higher (1.0) increases bankruptcy risk"
                ]
            })
            st.dataframe(indicators_df, use_container_width=True, hide_index=True)

    with col_right:
        guide_html = (
            '<div class="dark-card-container">'
            '<h3 style="margin:0 0 12px 0; font-size:1.25rem; font-weight:700; color:#ffffff !important;">🚀 Quick Navigation Guide</h3>'
            '<p class="info-highlight"><strong>🔍 Predict Risk:</strong><br>'
            'Test an individual company, inspect Bar & Pie probability charts, review key risk drivers, and download a formal assessment report.</p>'
            '<p class="info-highlight" style="margin-top:12px;"><strong>📂 Batch CSV Upload:</strong><br>'
            'Upload a multi-company CSV file to score an entire portfolio at once and download visual results.</p>'
            '<p class="info-highlight" style="margin-top:12px;"><strong>🕒 Prediction History:</strong><br>'
            'Review, compare, and export all predictions evaluated during your current active session.</p>'
            '</div>'
        )
        st.markdown(guide_html, unsafe_allow_html=True)

# ---------------------------------------------------------
# PAGE 2: SINGLE COMPANY PREDICT (Full-Page cash.png Background)
# ---------------------------------------------------------
elif page == "🔍 Predict Risk":
    set_page_background("cash", overlay_opacity=0.82)

    st.title("🔍 Single Company Bankruptcy Prediction")
    st.write("Evaluate working capital resilience, liquidity flexibility, and operational risk indicators.")

    with st.container(border=True):
        company_name = st.text_input("Company Name / Reference ID", value="Acme Corp Ltd.")

        st.subheader("Company Risk & Strength Indicators")
        col1, col2 = st.columns(2)

        with col1:
            industrial_risk = st.selectbox("Industrial Risk", [0.0, 0.5, 1.0], format_func=LEVEL_LABELS.get)
            management_risk = st.selectbox("Management Risk", [0.0, 0.5, 1.0], format_func=LEVEL_LABELS.get)
            financial_flexibility = st.selectbox("Financial Flexibility", [0.0, 0.5, 1.0], index=2, format_func=LEVEL_LABELS.get)

        with col2:
            credibility = st.selectbox("Credibility", [0.0, 0.5, 1.0], index=2, format_func=LEVEL_LABELS.get)
            competitiveness = st.selectbox("Competitiveness", [0.0, 0.5, 1.0], index=2, format_func=LEVEL_LABELS.get)
            operating_risk = st.selectbox("Operating Risk", [0.0, 0.5, 1.0], format_func=LEVEL_LABELS.get)

    st.write("")

    if st.button("Predict Bankruptcy Risk", type="primary", use_container_width=True):
        input_data = pd.DataFrame([{
            "industrial_risk": industrial_risk,
            "management_risk": management_risk,
            "financial_flexibility": financial_flexibility,
            "credibility": credibility,
            "competitiveness": competitiveness,
            "operating_risk": operating_risk
        }])

        # Model encoded: 0 = Bankruptcy, 1 = Non-Bankruptcy
        prediction = model.predict(input_data)[0]
        probabilities = model.predict_proba(input_data)[0]

        bankruptcy_prob = float(probabilities[0] * 100)
        non_bankruptcy_prob = float(probabilities[1] * 100)
        status_label = "High Bankruptcy Risk" if prediction == 0 else "Low Bankruptcy Risk"

        # Save to session history
        st.session_state["history"].append({
            "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Company": company_name.strip() or "Unnamed Company",
            "Status": status_label,
            "Bankruptcy Prob (%)": round(bankruptcy_prob, 2),
            "Non-Bankruptcy Prob (%)": round(non_bankruptcy_prob, 2),
            "Industrial Risk": industrial_risk,
            "Management Risk": management_risk,
            "Financial Flexibility": financial_flexibility,
            "Credibility": credibility,
            "Competitiveness": competitiveness,
            "Operating Risk": operating_risk
        })

        # Top Assessment Banner
        st.subheader(f"Assessment Result: {company_name}")
        if prediction == 0:
            show_result_banner(
                title="⚠️ FINAL ASSESSMENT: HIGH BANKRUPTCY RISK",
                message="Critical risk identified. Immediate financial restructuring, operational cost containment, and active credit or liquidity renegotiation is strongly recommended.",
                is_risk=True
            )
        else:
            show_result_banner(
                title="✅ FINAL ASSESSMENT: LOW BANKRUPTCY RISK",
                message="Solid solvency status. The company shows robust market competitiveness, high flexibility, and normal creditworthiness, positioning it well against defaults.",
                is_risk=False
            )

        m1, m2, m3 = st.columns(3)
        m1.metric("Bankruptcy Probability", f"{bankruptcy_prob:.2f}%")
        m2.metric("Non-Bankruptcy Probability", f"{non_bankruptcy_prob:.2f}%")
        m3.metric("Risk Classification", "Bankrupt (0)" if prediction == 0 else "Safe (1)")

        st.divider()

        # Side-by-Side Native Bar Chart & Pie (Donut) Chart
        st.subheader("📊 Probability Distribution Charts")
        chart_col1, chart_col2 = st.columns(2)

        with chart_col1:
            with st.container(border=True):
                st.markdown("**Probability Comparison (Bar Chart)**")
                prob_df = pd.DataFrame({
                    "Category": ["Non-Bankruptcy", "Bankruptcy"],
                    "Probability (%)": [non_bankruptcy_prob, bankruptcy_prob]
                })
                st.bar_chart(prob_df.set_index("Category"), height=245)

        with chart_col2:
            render_pie_chart(non_bankruptcy_prob, bankruptcy_prob, title="Risk Proportion Split (Pie Chart)")

        # Automated Key Risk Drivers & Downloadable Report
        st.divider()
        d_col1, d_col2 = st.columns([1.4, 1])

        with d_col1:
            red_flags = []
            strengths = []

            if financial_flexibility == 0.0:
                red_flags.append("Low Financial Flexibility (0.0) limits emergency capital access.")
            elif financial_flexibility == 1.0:
                strengths.append("High Financial Flexibility (1.0) supports liquidity.")

            if credibility == 0.0:
                red_flags.append("Low Credibility (0.0) restricts credit and lender trust.")
            elif credibility == 1.0:
                strengths.append("High Credibility (1.0) strengthens borrowing power.")

            if competitiveness == 0.0:
                red_flags.append("Low Competitiveness (0.0) threatens market share and margins.")
            elif competitiveness == 1.0:
                strengths.append("High Competitiveness (1.0) protects revenue stability.")

            if industrial_risk == 1.0:
                red_flags.append("High Industrial Risk (1.0) exposes firm to sector downturns.")
            if management_risk == 1.0:
                red_flags.append("High Management Risk (1.0) signals governance vulnerability.")
            if operating_risk == 1.0:
                red_flags.append("High Operating Risk (1.0) pressures day-to-day cash flow.")

            obs_parts = [
                '<div class="dark-card-container">',
                '<h3 style="margin: 0 0 14px 0; font-size: 1.25rem; font-weight: 700; color: #ffffff !important;">🔎 Key Indicator Observations</h3>'
            ]

            if red_flags:
                flags_li = "".join(f'<li style="margin-bottom: 6px; color: #ffffff !important;">{item}</li>' for item in red_flags)
                obs_parts.append(
                    f'<p class="warning-header" style="margin: 0 0 6px 0; font-size: 0.98rem; font-weight: 700; color: #fbcfe8 !important;">🚨 Risk Drivers Identified:</p>'
                    f'<ul style="margin: 0 0 14px 20px; padding: 0; font-size: 0.95rem; color: #ffffff !important;">{flags_li}</ul>'
                )

            if strengths:
                strengths_li = "".join(f'<li style="margin-bottom: 6px; color: #ffffff !important;">{item}</li>' for item in strengths)
                obs_parts.append(
                    f'<p class="strength-header" style="margin: 0 0 6px 0; font-size: 0.98rem; font-weight: 700; color: #a7f3d0 !important;">💎 Protective Strengths:</p>'
                    f'<ul style="margin: 0 0 14px 20px; padding: 0; font-size: 0.95rem; color: #ffffff !important;">{strengths_li}</ul>'
                )

            if not red_flags and not strengths:
                obs_parts.append(
                    '<p style="margin: 0; font-size: 0.95rem; color: #ffffff !important;">All monitored risk and financial indicators are currently evaluated at balanced/neutral moderate (0.5) levels.</p>'
                )

            obs_parts.append('</div>')
            st.markdown("".join(obs_parts), unsafe_allow_html=True)

        with d_col2:
            with st.container(border=True):
                st.subheader("📥 Download Assessment Report")
                report_text = f"""BANKRUPTCY PREVENTION SYSTEM - ASSESSMENT REPORT
==================================================
Generated On       : {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
Company Name       : {company_name}
Final Assessment   : {status_label}
Bankruptcy Prob    : {bankruptcy_prob:.2f}%
Non-Bankruptcy Prob: {non_bankruptcy_prob:.2f}%

INPUT INDICATORS (0.0 = Low, 0.5 = Medium, 1.0 = High)
--------------------------------------------------
- Industrial Risk       : {industrial_risk}
- Management Risk       : {management_risk}
- Financial Flexibility : {financial_flexibility}
- Credibility           : {credibility}
- Competitiveness       : {competitiveness}
- Operating Risk        : {operating_risk}

KEY OBSERVATIONS
--------------------------------------------------
Red Flags: {', '.join(red_flags) if red_flags else 'None'}
Strengths: {', '.join(strengths) if strengths else 'None'}
"""
                st.download_button(
                    label="📄 Download Executive Report (.TXT)",
                    data=report_text,
                    file_name=f"{company_name.replace(' ', '_')}_Risk_Report.txt",
                    mime="text/plain",
                    use_container_width=True
                )

                csv_export = input_data.copy()
                csv_export.insert(0, "company_name", company_name)
                csv_export["prediction"] = status_label
                csv_export["bankruptcy_prob_pct"] = round(bankruptcy_prob, 2)
                csv_export["non_bankruptcy_prob_pct"] = round(non_bankruptcy_prob, 2)

                st.download_button(
                    label="📊 Download Data Row (.CSV)",
                    data=csv_export.to_csv(index=False).encode("utf-8"),
                    file_name=f"{company_name.replace(' ', '_')}_Prediction.csv",
                    mime="text/csv",
                    use_container_width=True
                )

# ---------------------------------------------------------
# PAGE 3: BATCH CSV UPLOAD (Full-Page dollar.png Background)
# ---------------------------------------------------------
elif page == "📂 Batch CSV Upload":
    set_page_background("dollar", overlay_opacity=0.82)

    st.title("📂 Batch Portfolio Bankruptcy Prediction")
    st.write("Upload a CSV file containing multiple companies to score corporate portfolio risk in bulk.")

    template_html = (
        '<div class="dark-card-container">'
        '<h3 style="margin: 0 0 8px 0; font-size: 1.22rem; font-weight: 700; color: #ffffff !important;">📥 Template Guidelines</h3>'
        '<p style="margin: 0; font-size: 0.95rem; color: #f8fafc !important; line-height: 1.45;">'
        'To perform batch prediction, upload a <strong>.CSV</strong> file with columns matching our template: '
        '<code>company_name</code>, <code>industrial_risk</code>, <code>management_risk</code>, '
        '<code>financial_flexibility</code>, <code>credibility</code>, <code>competitiveness</code>, '
        'and <code>operating_risk</code>. Use risk level values of <strong>0.0 (Low)</strong>, <strong>0.5 (Medium)</strong>, and <strong>1.0 (High)</strong>.'
        '</p></div>'
    )
    st.markdown(template_html, unsafe_allow_html=True)

    sample_df = pd.DataFrame([
        {"company_name": "Alpha Manufacturing", "industrial_risk": 1.0, "management_risk": 1.0, "financial_flexibility": 0.0, "credibility": 0.0, "competitiveness": 0.0, "operating_risk": 1.0},
        {"company_name": "Beta FinTech Ltd", "industrial_risk": 0.0, "management_risk": 0.0, "financial_flexibility": 1.0, "credibility": 1.0, "competitiveness": 1.0, "operating_risk": 0.0},
        {"company_name": "Gamma Retail Co", "industrial_risk": 0.5, "management_risk": 0.5, "financial_flexibility": 0.5, "credibility": 0.5, "competitiveness": 0.0, "operating_risk": 0.5},
    ])

    st.download_button(
        label="📥 Download Sample CSV Template",
        data=sample_df.to_csv(index=False).encode("utf-8"),
        file_name="sample_bankruptcy_batch.csv",
        mime="text/csv"
    )

    st.divider()

    # Dark card header + 100% white text file uploader
    st.markdown(
        '<div style="background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%); '
        'padding: 12px 18px; border-radius: 12px 12px 0 0; border: 1.5px solid #334155; border-bottom: none;">'
        '<h4 style="margin: 0; color: #ffffff !important; font-weight: 700;">📤 Upload File (.CSV)</h4>'
        '</div>',
        unsafe_allow_html=True
    )
    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"], label_visibility="collapsed")

    if uploaded_file is not None:
        batch_df = pd.read_csv(uploaded_file)
        with st.container(border=True):
            st.subheader("Uploaded Data Preview")
            st.dataframe(batch_df.head(10), use_container_width=True)

        missing_cols = [col for col in FEATURE_COLS if col not in batch_df.columns]

        if missing_cols:
            st.error(f"Missing required columns in CSV: `{', '.join(missing_cols)}`")
        else:
            if st.button("Run Batch Predictions", type="primary"):
                X_batch = batch_df[FEATURE_COLS]
                preds = model.predict(X_batch)
                probs = model.predict_proba(X_batch)

                results_df = batch_df.copy()
                results_df["Prediction"] = ["High Bankruptcy Risk" if p == 0 else "Low Bankruptcy Risk" for p in preds]
                results_df["Bankruptcy Probability (%)"] = (probs[:, 0] * 100).round(2)
                results_df["Non-Bankruptcy Probability (%)"] = (probs[:, 1] * 100).round(2)

                total_count = len(results_df)
                high_risk_count = int((preds == 0).sum())
                low_risk_count = int((preds == 1).sum())

                st.subheader("Assessment Overview")
                if high_risk_count > 0:
                    show_result_banner(
                        title=f"⚠️ {high_risk_count} COMPANIES AT HIGH RISK",
                        message=f"Out of the {total_count} companies evaluated, {high_risk_count} were found to show substantial insolvency and default indicators. Priority auditing recommended.",
                        is_risk=True
                    )
                else:
                    show_result_banner(
                        title="✅ PORTFOLIO ANALYSIS COMPLETED - SAFE STATUS",
                        message="Excellent. No companies inside this uploaded dataset were classified as high risk for impending defaults or insolvency.",
                        is_risk=False
                    )

                k1, k2, k3 = st.columns(3)
                k1.metric("Total Companies Evaluated", total_count)
                k2.metric("High Bankruptcy Risk", high_risk_count)
                k3.metric("Low Bankruptcy Risk", low_risk_count)

                with st.container(border=True):
                    st.subheader("Batch Scoring Results")
                    st.dataframe(results_df, use_container_width=True)

                non_bank_pct = (low_risk_count / total_count * 100) if total_count > 0 else 0.0
                bank_pct = (high_risk_count / total_count * 100) if total_count > 0 else 0.0

                b_chart1, b_chart2 = st.columns(2)
                with b_chart1:
                    with st.container(border=True):
                        st.markdown("**Portfolio Count Comparison**")
                        batch_bar_df = pd.DataFrame({
                            "Status": ["Low Bankruptcy Risk", "High Bankruptcy Risk"],
                            "Companies": [low_risk_count, high_risk_count]
                        })
                        st.bar_chart(batch_bar_df.set_index("Status"), height=245)

                with b_chart2:
                    render_pie_chart(non_bank_pct, bank_pct, title="Portfolio Risk Distribution")

                st.download_button(
                    label="📥 Download Full Batch Results (.CSV)",
                    data=results_df.to_csv(index=False).encode("utf-8"),
                    file_name="batch_bankruptcy_predictions.csv",
                    mime="text/csv",
                    use_container_width=True
                )

# ---------------------------------------------------------
# PAGE 4: PREDICTION HISTORY (Full-Page bank.png Background)
# ---------------------------------------------------------
elif page == "🕒 Prediction History":
    set_page_background("bank", overlay_opacity=0.82)

    st.title("🕒 Session Prediction History")
    st.write("Compare companies evaluated during this session and export your audit log.")

    if not st.session_state["history"]:
        st.info("No predictions recorded yet. Go to **🔍 Predict Risk** to evaluate a company.")
    else:
        history_df = pd.DataFrame(st.session_state["history"])

        with st.container(border=True):
            filter_option = st.selectbox("Filter by Risk Status", ["All", "High Bankruptcy Risk", "Low Bankruptcy Risk"])
            if filter_option != "All":
                filtered_df = history_df[history_df["Status"] == filter_option]
            else:
                filtered_df = history_df

            st.dataframe(filtered_df, use_container_width=True, hide_index=True)

        col_h1, col_h2 = st.columns(2)
        with col_h1:
            st.download_button(
                label="📥 Download History Log (.CSV)",
                data=filtered_df.to_csv(index=False).encode("utf-8"),
                file_name="prediction_history_log.csv",
                mime="text/csv",
                use_container_width=True
            )
        with col_h2:
            if st.button("🗑️ Clear Session History", type="secondary", use_container_width=True):
                st.session_state["history"] = []
                st.rerun()