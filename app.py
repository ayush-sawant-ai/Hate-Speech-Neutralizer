import html
import os
import pickle
import re

import numpy as np
import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Hate Speech Neutralizer",
    page_icon="🛡️",
    layout="wide"
)


# ============================================================
# MODEL PATHS
# ============================================================

# Classifier (unchanged)
CLASSIFIER_MODEL_PATH = "models/gru_hate_speech_model.h5"
CLASSIFIER_TOKENIZER_PATH = "data/tokenizer.pkl"

# Neutralizer: fine-tuned T5 folder (contains model.safetensors, config.json, tokenizer files)
REWRITER_MODEL_DIR = "models/t5_neutralizer"
REWRITER_PREFIX = "Rewrite neutrally: "   # must match the prefix used in training


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 34px;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .subtitle {
        color: #666666;
        font-size: 16px;
        margin-bottom: 25px;
    }

    .result-box {
        padding: 18px;
        border-radius: 10px;
        border: 1px solid #dddddd;
        margin-top: 15px;
        margin-bottom: 15px;
        color: #222222;
    }

    .toxic-box {
        background-color: #fff4f4;
        border-left: 5px solid #d9534f;
    }

    .safe-box {
        background-color: #f3fff5;
        border-left: 5px solid #28a745;
    }

    .neutral-box {
        background-color: #f4f8ff;
        border-left: 5px solid #4285f4;
    }

    .result-title {
        font-size: 20px;
        font-weight: 650;
        margin-bottom: 8px;
    }

    .confidence {
        font-size: 15px;
        color: #555555;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# TEXT CLEANING (classifier only)
# ============================================================

def clean_text(text):

    text = str(text).lower()

    # Remove URLs
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)

    # Remove mentions
    text = re.sub(r"@\w+", "", text)

    # Keep alphabetic characters and spaces
    text = re.sub(r"[^a-zA-Z\s]", " ", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


# ============================================================
# LOAD CLASSIFIER
# ============================================================

@st.cache_resource
def load_classifier():

    model = tf.keras.models.load_model(CLASSIFIER_MODEL_PATH)

    with open(CLASSIFIER_TOKENIZER_PATH, "rb") as f:
        tokenizer = pickle.load(f)

    # Get model input length dynamically
    input_shape = model.input_shape

    if isinstance(input_shape, list):
        input_shape = input_shape[0]

    classifier_maxlen = input_shape[1]

    return model, tokenizer, classifier_maxlen


def classify_text(text):

    model, tokenizer, max_length = load_classifier()

    cleaned_text = clean_text(text)

    sequence = tokenizer.texts_to_sequences([cleaned_text])

    padded = pad_sequences(
        sequence,
        maxlen=max_length,
        padding="post",
        truncating="post"
    ).astype(np.int32)

    prediction = model.predict(padded, verbose=0)

    # Single sigmoid output
    probability = float(prediction[0][0])

    label = "TOXIC" if probability >= 0.5 else "SAFE"

    return {
        "label": label,
        "probability": probability
    }


# ============================================================
# LOAD REWRITER (fine-tuned T5)
# ============================================================

@st.cache_resource
def load_rewriter():

    # Imported here so the classifier still works if torch/transformers
    # have a problem, and so TensorFlow loads first.
    import torch
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

    if not os.path.isdir(REWRITER_MODEL_DIR):
        raise FileNotFoundError(
            f"Neutralizer folder not found: '{REWRITER_MODEL_DIR}'. "
            "Copy the trained t5_neutralizer folder into your models/ directory."
        )

    tokenizer = AutoTokenizer.from_pretrained(REWRITER_MODEL_DIR)
    model = AutoModelForSeq2SeqLM.from_pretrained(REWRITER_MODEL_DIR)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    return model, tokenizer, device


def rewrite_text(text):

    import torch

    model, tokenizer, device = load_rewriter()

    # T5 handles raw text well: do NOT lowercase or strip punctuation here.
    source = REWRITER_PREFIX + str(text).strip()

    inputs = tokenizer(
        source,
        return_tensors="pt",
        truncation=True,
        max_length=64
    ).to(device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            num_beams=5,
            no_repeat_ngram_size=3,
            max_new_tokens=64,
            early_stopping=True
        )

    result = tokenizer.decode(
        output_ids[0],
        skip_special_tokens=True
    ).strip()

    return result if result else None


# ============================================================
# SESSION STATE
# ============================================================

if "classification_result" not in st.session_state:
    st.session_state.classification_result = None

if "rewritten_text" not in st.session_state:
    st.session_state.rewritten_text = None

if "last_input_text" not in st.session_state:
    st.session_state.last_input_text = ""

if "input_text" not in st.session_state:
    st.session_state.input_text = ""


def set_example(example_text):
    st.session_state.input_text = example_text
    st.session_state.classification_result = None
    st.session_state.rewritten_text = None


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">Hate Speech Neutralizer</div>',
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="subtitle">
    Detect potentially toxic speech and generate a neutralized version
    using a GRU classifier and a fine-tuned T5 rewriting model.
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("Model Information")

    st.markdown(
        """
        **Detection Model**

        GRU-based binary classifier

        **Rewriting Model**

        Fine-tuned T5 (text-to-text transformer)

        **Detection Threshold**

        0.50
        """
    )

    st.divider()

    st.caption("The classifier produces a single sigmoid probability.")
    st.caption("The rewriter uses beam search decoding.")


# ============================================================
# EXAMPLE INPUTS
# ============================================================

st.subheader("Test Input")

example_columns = st.columns(3)

examples = [
    ("Toxic example", "I hate you and your stupid behavior."),
    ("Another example", "People like you should leave this place."),
    ("Safe example", "I really enjoyed the community event this weekend."),
]

for column, (button_text, example_text) in zip(example_columns, examples):

    with column:

        st.button(
            button_text,
            use_container_width=True,
            on_click=set_example,
            args=(example_text,)
        )


# ============================================================
# TEXT INPUT
# ============================================================

input_text = st.text_area(
    "Enter a sentence",
    key="input_text",
    height=130,
    max_chars=500,
    placeholder="Type a sentence here..."
)


# ============================================================
# DETECT BUTTON
# ============================================================

detect_button = st.button(
    "Detect Speech",
    type="primary",
    use_container_width=True
)


# ============================================================
# DETECTION
# ============================================================

if detect_button:

    if not input_text.strip():

        st.warning("Please enter some text first.")

    else:

        try:

            with st.spinner("Running hate speech classifier..."):
                result = classify_text(input_text)

            # Store so results survive Streamlit reruns
            st.session_state.classification_result = result
            st.session_state.last_input_text = input_text

            # Clear previous rewrite
            st.session_state.rewritten_text = None

        except Exception as e:

            st.error("Unable to run the classifier.")
            st.exception(e)


# ============================================================
# DISPLAY CLASSIFICATION RESULT
# ============================================================

result = st.session_state.classification_result

if result is not None:

    probability = result["probability"]
    label = result["label"]

    st.divider()

    st.subheader("Classification Result")

    if label == "TOXIC":

        st.markdown(
            f"""
            <div class="result-box toxic-box">

            <div class="result-title">
            ⚠️ Toxic / Hate Speech Detected
            </div>

            <div class="confidence">
            Toxicity probability:
            <strong>{probability * 100:.2f}%</strong>
            </div>

            </div>
            """,
            unsafe_allow_html=True
        )

        generate_button = st.button(
            "Generate Neutral Version",
            type="primary",
            use_container_width=True
        )

        if generate_button:

            try:

                with st.spinner("Generating neutral version..."):
                    rewritten = rewrite_text(
                        st.session_state.last_input_text
                    )

                st.session_state.rewritten_text = rewritten

            except Exception as e:

                st.session_state.rewritten_text = None

                st.error("Unable to generate the neutral version.")
                st.exception(e)

    else:

        st.markdown(
            f"""
            <div class="result-box safe-box">

            <div class="result-title">
            ✓ Safe / Non-Toxic
            </div>

            <div class="confidence">
            Safe probability:
            <strong>{(1 - probability) * 100:.2f}%</strong>
            </div>

            </div>
            """,
            unsafe_allow_html=True
        )

        st.info("The input does not require neutralization.")


# ============================================================
# DISPLAY NEUTRALIZED VERSION
# ============================================================

rewritten_text = st.session_state.rewritten_text

if rewritten_text:

    st.divider()

    st.subheader("Neutralized Version")

    st.markdown(
        f"""
        <div class="result-box neutral-box">

        <div class="result-title">
        Neutralized Text
        </div>

        <div style="font-size:17px;">
        {html.escape(rewritten_text)}
        </div>

        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# MODEL DETAILS
# ============================================================

st.divider()

with st.expander("Model Details"):

    col1, col2 = st.columns(2)

    with col1:

        st.markdown(
            """
            ### Hate Speech Detector

            **Architecture:** GRU

            **Task:** Binary classification

            **Output:** Sigmoid probability

            **Threshold:** 0.50

            **Input:** Cleaned text

            **Tokenizer:** Keras Tokenizer

            **Vocabulary:** 20,000
            """
        )

    with col2:

        st.markdown(
            """
            ### Neutralization Model

            **Architecture:** Fine-tuned T5 (encoder-decoder transformer)

            **Task:** Toxic-to-neutral text rewriting

            **Training data:** Parallel toxic/neutral sentence pairs

            **Decoding:** Beam search (5 beams)

            **Input:** Raw text with a task prefix
            """
        )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption("Hate Speech Neutralizer — Deep Learning Project")