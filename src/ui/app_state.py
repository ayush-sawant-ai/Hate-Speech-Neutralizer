"""Session state management for the Streamlit UI.
Guarantees that editing the input invalidates stale analysis and rewrite results.
"""

from typing import Any, Dict, Optional
import streamlit as st


def init_session_state() -> None:
    """Initialize all required session state variables."""
    if "input_text" not in st.session_state:
        st.session_state.input_text = ""
    if "analyzed_text" not in st.session_state:
        st.session_state.analyzed_text = ""
    if "classification_result" not in st.session_state:
        st.session_state.classification_result = None
    if "rewritten_result" not in st.session_state:
        st.session_state.rewritten_result = None
    if "is_stale" not in st.session_state:
        st.session_state.is_stale = False
    if "active_model_key" not in st.session_state:
        st.session_state.active_model_key = "bigru"
    if "detection_threshold" not in st.session_state:
        st.session_state.detection_threshold = 0.40
    if "analysis_record" not in st.session_state:
        st.session_state.analysis_record = None

    # Optional query parameter demo initialization for visual inspection
    try:
        demo_param = st.query_params.get("demo")
        if demo_param == "analyzed" and st.session_state.classification_result is None:
            st.session_state.input_text = "You are a complete idiot and nobody cares what you think."
            st.session_state.analyzed_text = st.session_state.input_text
            st.session_state.classification_result = {
                "label": "TOXIC",
                "probability": 0.9774,
                "is_toxic": True,
                "threshold": 0.40,
                "model_key": "bigru",
                "model_name": "Bidirectional GRU",
                "cleaned_text": "you are a complete idiot and nobody cares what you think",
                "is_valid": True,
                "note": "Binary toxicity target derived from Jigsaw dataset."
            }
        elif demo_param == "rewritten" and st.session_state.rewritten_result is None:
            st.session_state.input_text = "You are a complete idiot and nobody cares what you think."
            st.session_state.analyzed_text = st.session_state.input_text
            st.session_state.classification_result = {
                "label": "TOXIC",
                "probability": 0.9774,
                "is_toxic": True,
                "threshold": 0.40,
                "model_key": "bigru",
                "model_name": "Bidirectional GRU",
                "cleaned_text": "you are a complete idiot and nobody cares what you think",
                "is_valid": True,
                "note": "Binary toxicity target derived from Jigsaw dataset."
            }
            st.session_state.rewritten_result = {
                "original_text": st.session_state.input_text,
                "rewritten_text": "I disagree with your opinion. Let's discuss the issue respectfully.",
                "display_text": "I disagree with your opinion. Let's discuss the issue respectfully.",
                "status": "SUGGESTED_ALTERNATIVE",
                "status_label": "Suggested constructive alternative",
                "is_acceptable": True,
                "message": "Personal attack reframed into a respectful statement of disagreement or dissatisfaction.",
                "intent_warning": "",
                "suggestion": "",
                "original_toxicity": 0.9774,
                "post_toxicity": 0.0033,
                "score_decrease": 0.9741,
                "toxicity_reduction": 0.9741,
            }
        elif demo_param == "failure" and st.session_state.rewritten_result is None:
            st.session_state.input_text = "I will find you and hurt you."
            st.session_state.analyzed_text = st.session_state.input_text
            st.session_state.classification_result = {
                "label": "TOXIC",
                "probability": 0.8850,
                "is_toxic": True,
                "threshold": 0.40,
                "model_key": "bigru",
                "model_name": "Bidirectional GRU",
                "cleaned_text": "i will find you and hurt you",
                "is_valid": True,
                "note": "Binary toxicity target derived from Jigsaw dataset."
            }
            st.session_state.rewritten_result = {
                "original_text": st.session_state.input_text,
                "rewritten_text": None,
                "display_text": "Could not generate a reliable constructive rewrite. Please rephrase the input or try again.",
                "status": "REWRITE_UNAVAILABLE",
                "status_label": "Rewrite unavailable",
                "is_acceptable": False,
                "message": "Explicit violent threats cannot be safely paraphrased without removing the threat.",
                "intent_warning": "Note: The original input contained threats or severe hostility. Safe alternatives cannot preserve harmful intent.",
                "suggestion": "Violent threats violate safety guidelines and cannot be neutralized.",
                "original_toxicity": 0.8850,
                "post_toxicity": None,
                "score_decrease": None,
                "toxicity_reduction": None,
            }
    except Exception:
        pass


def on_text_input_change() -> None:
    """Callback fired when the user types or edits in the input text area.
    Invalidates stale results so previous output does not linger for new text.
    """
    current_text = st.session_state.get("input_text", "")
    analyzed_text = st.session_state.get("analyzed_text", "")

    if current_text != analyzed_text:
        st.session_state.is_stale = True
        st.session_state.classification_result = None
        st.session_state.rewritten_result = None
        st.session_state.analysis_record = None


def on_model_or_threshold_change() -> None:
    """Callback fired when the user changes classifier architecture or threshold.
    Invalidates dependent results so outputs never reflect mismatched settings.
    """
    st.session_state.is_stale = True
    st.session_state.classification_result = None
    st.session_state.rewritten_result = None
    st.session_state.analysis_record = None


def set_example_input(text: str) -> None:
    """Set example input and clear stale results."""
    st.session_state.input_text = text
    st.session_state.analyzed_text = ""
    st.session_state.classification_result = None
    st.session_state.rewritten_result = None
    st.session_state.analysis_record = None
    st.session_state.is_stale = False


def reset_workspace() -> None:
    """Clear all inputs, state, and outputs together."""
    st.session_state.input_text = ""
    st.session_state.analyzed_text = ""
    st.session_state.classification_result = None
    st.session_state.rewritten_result = None
    st.session_state.analysis_record = None
    st.session_state.is_stale = False

