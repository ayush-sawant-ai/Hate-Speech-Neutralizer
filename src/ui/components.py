"""UI styling and visual components for the Hate Speech Neutralizer desktop workspace.
Guarantees high-contrast readability, single-panel output layout, and consistent light theme.
"""

import html
from typing import Any, Dict, Optional
import streamlit as st


def inject_custom_styles() -> None:
    """Inject clean, restrained CSS with accessible contrast and light theme enforcement."""
    st.markdown(
        """
        <style>
        /* Base typography and background */
        .stApp {
            background-color: #F5F7FA;
            color: #172033;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
        }

        /* Sidebar container styling */
        [data-testid="stSidebar"] {
            background-color: #FFFFFF !important;
            border-right: 1px solid #E2E8F0 !important;
        }
        [data-testid="stSidebar"] .stMarkdown h3 {
            color: #172033 !important;
            font-size: 16px !important;
            font-weight: 700 !important;
            margin-top: 8px !important;
        }
        [data-testid="stSidebar"] .stMarkdown p, 
        [data-testid="stSidebar"] label {
            color: #475569 !important;
            font-size: 13px !important;
        }

        /* Native Text Area Input */
        .stTextArea textarea {
            background-color: #FFFFFF !important;
            color: #172033 !important;
            border: 1.5px solid #64748B !important;
            border-radius: 6px !important;
            font-size: 14px !important;
            line-height: 1.5 !important;
            padding: 10px 12px !important;
        }
        .stTextArea textarea::placeholder {
            color: #64748B !important;
            opacity: 1 !important;
        }
        .stTextArea textarea:focus {
            border-color: #1D4ED8 !important;
            box-shadow: 0 0 0 2px rgba(29, 78, 216, 0.15) !important;
            color: #172033 !important;
        }

        /* Native Selectbox & Dropdown */
        div[data-baseweb="select"] {
            background-color: #FFFFFF !important;
            border: 1.5px solid #64748B !important;
            border-radius: 6px !important;
        }
        div[data-baseweb="select"] * {
            color: #172033 !important;
            font-size: 13px !important;
        }
        ul[data-testid="stSelectboxMenu"] {
            background-color: #FFFFFF !important;
            border: 1px solid #CBD5E1 !important;
        }
        ul[data-testid="stSelectboxMenu"] li {
            color: #172033 !important;
        }

        /* Button styling */
        .stButton > button[kind="primary"],
        .stButton > button[data-testid="stBaseButton-primary"] {
            background-color: #1D4ED8 !important;
            color: #FFFFFF !important;
            border: 1px solid #1D4ED8 !important;
            border-radius: 6px !important;
            font-weight: 600 !important;
            font-size: 14px !important;
            padding: 6px 16px !important;
            transition: all 0.15s ease-in-out !important;
        }
        .stButton > button[kind="primary"]:hover,
        .stButton > button[data-testid="stBaseButton-primary"]:hover {
            background-color: #1E40AF !important;
            border-color: #1E40AF !important;
            color: #FFFFFF !important;
        }

        .stButton > button[kind="secondary"],
        .stButton > button[data-testid="stBaseButton-secondary"] {
            background-color: #FFFFFF !important;
            color: #172033 !important;
            border: 1.5px solid #CBD5E1 !important;
            border-radius: 6px !important;
            font-weight: 500 !important;
            font-size: 13px !important;
            padding: 6px 14px !important;
            transition: all 0.15s ease-in-out !important;
        }
        .stButton > button[kind="secondary"]:hover,
        .stButton > button[data-testid="stBaseButton-secondary"]:hover {
            background-color: #F1F5F9 !important;
            border-color: #94A3B8 !important;
            color: #0F172A !important;
        }

        /* Compact Header */
        .workspace-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 10px 0 16px 0;
            border-bottom: 1px solid #CBD5E1;
            margin-bottom: 20px;
        }
        .workspace-title {
            font-size: 22px;
            font-weight: 700;
            color: #172033;
            margin: 0;
            letter-spacing: -0.02em;
        }
        .workspace-subtitle {
            font-size: 13px;
            color: #475569;
            margin-top: 2px;
        }
        .workspace-badge {
            font-size: 12px;
            font-weight: 600;
            letter-spacing: 0.02em;
            padding: 4px 10px;
            border-radius: 4px;
            background-color: #EFF6FF;
            color: #1D4ED8;
            border: 1px solid #BFDBFE;
        }

        /* Panel Container */
        .panel-box {
            background-color: #FFFFFF;
            border: 1.5px solid #CBD5E1;
            border-radius: 8px;
            padding: 16px;
            margin-bottom: 16px;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);
        }
        .panel-title {
            font-size: 13px;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.04em;
            color: #475569;
            margin-bottom: 12px;
            display: flex;
            align-items: center;
            justify-content: space-between;
        }

        /* Classification Badges */
        .badge-toxic {
            display: inline-block;
            background-color: #FEF2F2;
            color: #991B1B;
            border: 1.5px solid #FECACA;
            padding: 4px 10px;
            border-radius: 6px;
            font-weight: 700;
            font-size: 12px;
            letter-spacing: 0.03em;
        }
        .badge-safe {
            display: inline-block;
            background-color: #F0FDF4;
            color: #166534;
            border: 1.5px solid #BBF7D0;
            padding: 4px 10px;
            border-radius: 6px;
            font-weight: 700;
            font-size: 12px;
            letter-spacing: 0.03em;
        }

        /* Status Tags */
        .tag-rewrite {
            display: inline-block;
            background-color: #F0FDF4;
            color: #166534;
            border: 1px solid #BBF7D0;
            padding: 3px 8px;
            border-radius: 4px;
            font-weight: 600;
            font-size: 12px;
        }
        .tag-alternative {
            display: inline-block;
            background-color: #EFF6FF;
            color: #1D4ED8;
            border: 1px solid #BFDBFE;
            padding: 3px 8px;
            border-radius: 4px;
            font-weight: 600;
            font-size: 12px;
        }
        .tag-unavailable {
            display: inline-block;
            background-color: #FEF2F2;
            color: #991B1B;
            border: 1px solid #FECACA;
            padding: 3px 8px;
            border-radius: 4px;
            font-weight: 600;
            font-size: 12px;
        }

        /* Metric Rows */
        .metric-table {
            width: 100%;
            border-collapse: collapse;
            font-size: 13px;
            margin-top: 8px;
        }
        .metric-table td {
            padding: 6px 0;
            border-bottom: 1px solid #F1F5F9;
        }
        .metric-label {
            color: #475569;
        }
        .metric-value {
            font-weight: 600;
            color: #172033;
            text-align: right;
            font-variant-numeric: tabular-nums;
        }

        /* Rewritten Output Card */
        .output-card {
            background-color: #FFFFFF;
            border: 1.5px solid #CBD5E1;
            border-radius: 6px;
            padding: 14px 16px;
            font-size: 15px;
            line-height: 1.6;
            color: #172033;
            min-height: 90px;
        }
        .output-placeholder {
            color: #64748B;
            font-style: italic;
            padding: 24px 12px;
            text-align: center;
            font-size: 13px;
        }
        .output-unavailable {
            background-color: #FFFBEB;
            border: 1.5px dashed #F59E0B;
            color: #92400E;
            font-style: italic;
            display: flex;
            align-items: center;
            justify-content: center;
            text-align: center;
            padding: 20px;
        }

        /* Notice / Warning Banners */
        .warning-notice {
            background-color: #FFFBEB;
            border-left: 3px solid #D97706;
            padding: 10px 14px;
            border-radius: 4px;
            font-size: 12px;
            color: #92400E;
            margin-top: 10px;
            line-height: 1.5;
        }
        .info-notice {
            background-color: #F8FAFC;
            border-left: 3px solid #64748B;
            padding: 10px 14px;
            border-radius: 4px;
            font-size: 12px;
            color: #334155;
            margin-top: 10px;
            line-height: 1.5;
        }

        /* Character Counter */
        .char-count {
            font-size: 12px;
            color: #64748B;
            text-align: right;
            margin-top: -6px;
            margin-bottom: 10px;
        }

        /* Score delta bar */
        .delta-pill {
            display: inline-block;
            padding: 2px 8px;
            border-radius: 4px;
            font-size: 12px;
            font-weight: 600;
        }
        .delta-positive {
            background-color: #DCFCE7;
            color: #15803D;
        }
        .delta-increase {
            background-color: #FEE2E2;
            color: #991B1B;
        }
        .delta-neutral {
            background-color: #F1F5F9;
            color: #475569;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_header(active_model_name: str = "Bidirectional GRU") -> None:
    """Render compact, purposeful application header with dynamic active model badge."""
    escaped_model_name = html.escape(active_model_name)
    st.markdown(
        f"""
        <div class="workspace-header">
            <div>
                <h1 class="workspace-title">Hate Speech Neutralizer</h1>
                <div class="workspace-subtitle">
                    Text toxicity analysis and constructive neutralizer workspace
                </div>
            </div>
            <div>
                <span class="workspace-badge">{escaped_model_name} + Flan-T5</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_classification_card(result: Dict[str, Any]) -> None:
    """Render structured classification diagnostics with trustworthy model metadata."""
    label = result.get("label", "SAFE")
    prob = result.get("probability", 0.0)
    threshold = result.get("threshold", 0.40)
    model_name = result.get("model_name", result.get("model_key", "bigru").upper())

    badge_html = (
        '<span class="badge-toxic">TOXIC DETECTED</span>'
        if label == "TOXIC"
        else '<span class="badge-safe">NON-TOXIC (SAFE)</span>'
    )

    st.markdown(
        f"""
        <div class="panel-box">
            <div class="panel-title">
                <span>Classification Result</span>
                {badge_html}
            </div>
            <table class="metric-table">
                <tr>
                    <td class="metric-label">Toxicity Model Score</td>
                    <td class="metric-value">{prob:.4f}</td>
                </tr>
                <tr>
                    <td class="metric-label">Decision Threshold</td>
                    <td class="metric-value">{threshold:.2f}</td>
                </tr>
                <tr>
                    <td class="metric-label">Active Classifier Checkpoint</td>
                    <td class="metric-value">{html.escape(model_name)}</td>
                </tr>
                <tr>
                    <td class="metric-label">Margin vs Threshold</td>
                    <td class="metric-value">{prob - threshold:+.4f}</td>
                </tr>
            </table>
            <div class="info-notice">
                <strong>Note on Model Scores:</strong> Output scores represent uncalibrated sigmoid network
                activations for general toxicity (Jigsaw dataset), not guaranteed Bayesian probabilities.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_rewritten_card(
    original_text: str,
    rewritten_res: Optional[Dict[str, Any]] = None,
    is_toxic: bool = True
) -> None:
    """Render single primary output panel with honest status labels and score comparison."""
    if not rewritten_res:
        if not is_toxic and original_text:
            placeholder_msg = "Input text was classified as non-toxic. Neutralization is unnecessary."
        else:
            placeholder_msg = "Enter text, click \"Analyze text\", and generate a constructive rewrite if toxicity is flagged."

        st.markdown(
            f"""
            <div class="panel-box">
                <div class="panel-title">Constructive Output</div>
                <div class="output-card">
                    <div class="output-placeholder">{placeholder_msg}</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    display_text = (
        rewritten_res.get("display_text")
        or "Could not generate a reliable constructive rewrite. Please rephrase the input or try again."
    )
    is_acceptable = rewritten_res.get("is_acceptable", False)
    status = rewritten_res.get("status", "REWRITE_UNAVAILABLE")
    status_label = rewritten_res.get(
        "status_label",
        "Suggested rewrite" if is_acceptable else "Rewrite unavailable"
    )
    orig_score = rewritten_res.get("original_toxicity")
    post_score = rewritten_res.get("post_toxicity")
    intent_warning = rewritten_res.get("intent_warning")
    escaped_rewrite = html.escape(display_text)

    # Determine badge style
    if not is_acceptable or status == "REWRITE_UNAVAILABLE":
        tag_html = f'<span class="tag-unavailable">{html.escape(status_label)}</span>'
        card_class = "output-card output-unavailable"
    elif status == "SUGGESTED_ALTERNATIVE":
        tag_html = f'<span class="tag-alternative">{html.escape(status_label)}</span>'
        card_class = "output-card"
    else:
        tag_html = f'<span class="tag-rewrite">{html.escape(status_label)}</span>'
        card_class = "output-card"

    # Score comparison table (Toxicity score decrease = input score - output score)
    delta_html = ""
    if orig_score is not None and post_score is not None:
        decrease = orig_score - post_score
        if decrease > 0:
            pill_class = "delta-positive"
            decrease_label = f"Toxicity Score Decrease: +{decrease:.4f}"
        elif decrease < 0:
            pill_class = "delta-increase"
            decrease_label = f"Toxicity Score Increase: {decrease:.4f}"
        else:
            pill_class = "delta-neutral"
            decrease_label = "No Score Change: 0.0000"

        delta_html = f"""
        <table class="metric-table" style="margin-top: 12px;">
            <tr>
                <td class="metric-label">Input Toxicity Score</td>
                <td class="metric-value">{orig_score:.4f}</td>
            </tr>
            <tr>
                <td class="metric-label">Output Toxicity Score (Re-scored with same model)</td>
                <td class="metric-value">{post_score:.4f}</td>
            </tr>
            <tr>
                <td class="metric-label">Toxicity Score Change</td>
                <td class="metric-value">
                    <span class="delta-pill {pill_class}">{decrease_label}</span>
                </td>
            </tr>
        </table>
        """

    warning_html = ""
    if intent_warning:
        warning_html = f'<div class="warning-notice">{html.escape(intent_warning)}</div>'

    suggestion_html = ""
    if rewritten_res.get("suggestion"):
        suggestion_html = f'<div class="info-notice">{html.escape(rewritten_res["suggestion"])}</div>'

    st.markdown(
        f"""
        <div class="panel-box">
            <div class="panel-title">
                <span>Constructive Output</span>
                {tag_html}
            </div>
            <div class="{card_class}">
                {escaped_rewrite}
            </div>
            {warning_html}
            {suggestion_html}
            {delta_html}
        </div>
        """,
        unsafe_allow_html=True,
    )
