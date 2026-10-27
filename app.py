"""Hate Speech Neutralizer - Desktop Workspace Application.

A production-grade text analysis and constructive neutralization tool combining
recurrent neural classification (BiGRU, GRU, BiLSTM, LSTM) with fine-tuned
T5 text-to-text rewriting.
"""

from pathlib import Path
import streamlit as st

from src.config import (
    CLASSIFIER_CHECKPOINTS,
    DEFAULT_CLASSIFIER_KEY,
    DEFAULT_THRESHOLD,
    SAFE_THRESHOLD_MAX,
    SAFE_THRESHOLD_MIN,
)
from src.classification.detector import ToxicityDetector
from src.rewriting.neutralizer import TextNeutralizer
from src.ui.app_state import (
    init_session_state,
    on_model_or_threshold_change,
    on_text_input_change,
    reset_workspace,
    set_example_input,
)
from src.ui.components import (
    inject_custom_styles,
    render_classification_card,
    render_header,
    render_rewritten_card,
)

# Page configuration
st.set_page_config(
    page_title="Hate Speech Neutralizer",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="auto"
)

# Apply custom restrained CSS styling
inject_custom_styles()

# Initialize session state variables
init_session_state()


# Cached Model Loaders
@st.cache_resource(show_spinner="Loading toxicity classification model...")
def get_cached_detector(model_key: str) -> ToxicityDetector:
    """Load and cache the neural toxicity classifier for the given model_key."""
    detector = ToxicityDetector(model_key=model_key, threshold=st.session_state.detection_threshold)
    detector.load()
    return detector


@st.cache_resource(show_spinner="Loading T5 neutralization transformer...")
def get_cached_neutralizer() -> TextNeutralizer:
    """Load and cache the T5 sequence-to-sequence model."""
    neutralizer = TextNeutralizer()
    neutralizer.load()
    return neutralizer


# Sidebar Configuration and Model Controls
with st.sidebar:
    st.markdown("### Model Configuration")

    model_options = list(CLASSIFIER_CHECKPOINTS.keys())
    model_labels = [CLASSIFIER_CHECKPOINTS[k]["name"] for k in model_options]

    current_index = model_options.index(st.session_state.active_model_key) if st.session_state.active_model_key in model_options else 0

    def handle_sidebar_model_change():
        new_label = st.session_state.sidebar_model_select
        new_key = model_options[model_labels.index(new_label)]
        if new_key != st.session_state.active_model_key:
            st.session_state.active_model_key = new_key
            on_model_or_threshold_change()

    selected_label = st.selectbox(
        "Classifier Architecture",
        options=model_labels,
        index=current_index,
        key="sidebar_model_select",
        help="Select which recurrent neural network checkpoint to load for toxicity detection.",
        on_change=handle_sidebar_model_change
    )
    selected_model_key = model_options[model_labels.index(selected_label)]
    st.session_state.active_model_key = selected_model_key

    st.markdown(
        f"<div style='font-size:12px; color:#475569; margin-top:-8px; margin-bottom:16px; line-height:1.4;'>"
        f"{CLASSIFIER_CHECKPOINTS[selected_model_key]['description']}"
        f"</div>",
        unsafe_allow_html=True
    )

    def handle_sidebar_threshold_change():
        on_model_or_threshold_change()

    threshold = st.slider(
        "Decision Threshold",
        min_value=SAFE_THRESHOLD_MIN,
        max_value=SAFE_THRESHOLD_MAX,
        value=float(st.session_state.detection_threshold),
        step=0.05,
        key="threshold_slider",
        help=(
            "Threshold determining when a model score is flagged as TOXIC. "
            "Lower values flag more content but increase false alarms; "
            "higher values reduce false alarms but may miss subtle hostility."
        ),
        on_change=handle_sidebar_threshold_change
    )
    st.session_state.detection_threshold = threshold

    st.markdown(
        f"<div style='font-size:12px; color:#475569; margin-top:-6px;'>"
        f"Active decision threshold: <strong>{threshold:.2f}</strong>"
        f"</div>",
        unsafe_allow_html=True
    )

    st.divider()

    st.markdown("### Sample Test Cases")
    sample_cases = [
        ("Direct Insult", "You are a complete idiot and nobody cares what you think."),
        ("Abusive Request", "You are such a shit, improve your work."),
        ("Code Review Attack", "This code is trash, fix it."),
        ("Hostile Demand", "Shut up and listen to me."),
        ("Flawed Explanation", "Your explanation is stupid and makes no sense."),
        ("Legitimate Disagreement (Safe)", "I disagree with your proposal because latency is too high."),
        ("Extreme Threat (Unsafe)", "I will find you and hurt you."),
    ]

    for label, sample_text in sample_cases:
        st.button(
            label,
            use_container_width=True,
            on_click=set_example_input,
            args=(sample_text,)
        )

    st.divider()
    st.button(
        "Clear",
        key="sidebar_clear_btn",
        use_container_width=True,
        on_click=reset_workspace
    )


# Render Header with dynamic model name
active_name = CLASSIFIER_CHECKPOINTS[st.session_state.active_model_key]["name"]
render_header(active_model_name=active_name)

# Two-Column Workspace Layout
col_input, col_output = st.columns([1, 1], gap="large")

with col_input:
    st.markdown(
        '<div style="font-weight:600; font-size:14px; margin-bottom:6px; color:#172033;">Input Text</div>',
        unsafe_allow_html=True
    )

    text_input = st.text_area(
        label="Input text for toxicity screening",
        key="input_text",
        height=130,
        max_chars=1000,
        placeholder="Type or paste text to evaluate for toxicity...",
        label_visibility="collapsed",
        on_change=on_text_input_change
    )

    char_count = len(text_input) if text_input else 0
    st.markdown(
        f'<div class="char-count">{char_count} / 1000 characters</div>',
        unsafe_allow_html=True
    )

    action_col1, action_col2 = st.columns([2, 1])
    with action_col1:
        analyze_btn = st.button("Analyze text", type="primary", use_container_width=True)
    with action_col2:
        st.button("Clear", key="main_clear_btn", use_container_width=True, on_click=reset_workspace)

    # Trigger Analysis
    if analyze_btn:
        if not text_input.strip():
            st.warning("Please enter non-empty text to analyze.")
        else:
            try:
                detector = get_cached_detector(model_key=st.session_state.active_model_key)
                with st.spinner("Analyzing text sequence..."):
                    result = detector.predict(text_input, threshold=st.session_state.detection_threshold)

                st.session_state.classification_result = result
                st.session_state.analyzed_text = text_input
                st.session_state.rewritten_result = None
                st.session_state.is_stale = False

            except Exception as e:
                st.error("Error during toxicity classification.")
                st.exception(e)

    # Render Classification Diagnostics
    if st.session_state.classification_result is not None:
        render_classification_card(st.session_state.classification_result)

        # Show Generate Constructive Rewrite action if toxic
        is_toxic = st.session_state.classification_result.get("is_toxic", False)
        if is_toxic:
            generate_btn = st.button(
                "Generate constructive rewrite",
                type="primary",
                use_container_width=True,
                disabled=st.session_state.is_stale
            )
            if generate_btn:
                try:
                    neutralizer = get_cached_neutralizer()
                    detector = get_cached_detector(model_key=st.session_state.active_model_key)
                    with st.spinner("Rewriting using constructive T5 model..."):
                        rewrite_res = neutralizer.rewrite(
                            st.session_state.analyzed_text,
                            detector_instance=detector,
                            original_prob=st.session_state.classification_result["probability"],
                            threshold=st.session_state.detection_threshold
                        )
                    st.session_state.rewritten_result = rewrite_res
                    st.rerun()

                except Exception as e:
                    st.error("Error during neutral text generation.")
                    st.exception(e)
        else:
            st.markdown(
                "<div style='font-size:13px; color:#166534; margin-top:8px;'>"
                "✓ Screened as non-toxic. Constructive rewriting is not required for legitimate criticism or safe text."
                "</div>",
                unsafe_allow_html=True
            )


with col_output:
    st.markdown(
        '<div style="font-weight:600; font-size:14px; margin-bottom:6px; color:#172033;">Constructive Output</div>',
        unsafe_allow_html=True
    )

    is_currently_toxic = (
        st.session_state.classification_result.get("is_toxic", False)
        if st.session_state.classification_result is not None
        else True
    )

    render_rewritten_card(
        original_text=st.session_state.analyzed_text,
        rewritten_res=st.session_state.rewritten_result,
        is_toxic=is_currently_toxic
    )

    # Working visible Copy control (only displayed when acceptable rewrite exists)
    if (
        st.session_state.rewritten_result
        and st.session_state.rewritten_result.get("is_acceptable")
        and st.session_state.rewritten_result.get("rewritten_text")
    ):
        copy_text = st.session_state.rewritten_result["rewritten_text"]
        st.markdown(
            '<div style="font-size:12px; font-weight:600; color:#475569; margin-top:12px; margin-bottom:4px;">'
            'Copy Rewrite'
            '</div>',
            unsafe_allow_html=True
        )
        st.code(copy_text, language=None)


# Technical & Limitations Collapsible Section
st.divider()

with st.expander("Model Details, Architecture & Dataset Limitations"):
    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            """
            #### Detection Architecture
            - **Models Available:** Bidirectional GRU, Standard GRU, Bidirectional LSTM, Standard LSTM.
            - **Input Preprocessing:** Unicode NFKC normalization, lowercasing, URL & mention stripping, alpha-numeric retaining. Padded/truncated to 200 tokens.
            - **Dataset:** Trained on Jigsaw Toxic Comments dataset.
            - **Important Target Definition:** Binary classification represents general toxicity. It is not an explicit demographic hate-speech category classifier; toxic speech includes general insults, profanity, and harassment.
            - **Model Output Calibration:** Sigmoid activations reflect uncalibrated network scores rather than Bayesian posterior probabilities.
            """
        )

    with col2:
        st.markdown(
            """
            #### Neutralization Architecture
            - **Model:** Fine-tuned `google/flan-t5-base` sequence-to-sequence transformer.
            - **Instruction Prompt:** Constructive instruction directing the model to remove insults, preserve feedback or requests, and output complete grammatical sentences.
            - **Decoding Strategy:** 4-beam search with 3-gram repetition penalty and bounded alternative generation retry.
            - **Validation Verification:** Re-scoring using the identical classifier function, broken syntax detection, message loss checks, and violent threat safety filters.
            - **Threat Intent Transformation:** Safe alternatives cannot preserve harmful violent intent; explicit violent threats cannot be safely paraphrased and are suppressed.
            """
        )
