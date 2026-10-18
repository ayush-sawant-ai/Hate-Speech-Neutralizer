import streamlit as st
import tensorflow as tf
import numpy as np
import pickle
import re

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

CLASSIFIER_MODEL_PATH = "models/gru_hate_speech_model.h5"
CLASSIFIER_TOKENIZER_PATH = "data/tokenizer.pkl"

REWRITER_MODEL_PATH = "models/rewriting_seq2seq_gru.keras"
SOURCE_TOKENIZER_PATH = "data/source_tokenizer.pkl"
TARGET_TOKENIZER_PATH = "data/target_tokenizer.pkl"
SEQUENCE_INFO_PATH = "data/rewriting_sequence_info.pkl"


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

    .model-card {
        background-color: #f7f7f7;
        padding: 15px;
        border-radius: 8px;
        margin-bottom: 10px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# TEXT CLEANING
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

    model = tf.keras.models.load_model(
        CLASSIFIER_MODEL_PATH
    )

    with open(CLASSIFIER_TOKENIZER_PATH, "rb") as f:
        tokenizer = pickle.load(f)

    # Get model input length dynamically
    input_shape = model.input_shape

    if isinstance(input_shape, list):
        input_shape = input_shape[0]

    classifier_maxlen = input_shape[1]

    return model, tokenizer, classifier_maxlen


# ============================================================
# CLASSIFIER
# ============================================================

def classify_text(text):

    model, tokenizer, max_length = load_classifier()

    cleaned_text = clean_text(text)

    sequence = tokenizer.texts_to_sequences(
        [cleaned_text]
    )

    padded = pad_sequences(
        sequence,
        maxlen=max_length,
        padding="post",
        truncating="post"
    ).astype(np.int32)

    prediction = model.predict(
        padded,
        verbose=0
    )

    # Your model has a single sigmoid output
    probability = float(prediction[0][0])

    if probability >= 0.5:
        label = "TOXIC"
    else:
        label = "SAFE"

    return {
        "label": label,
        "probability": probability
    }


# ============================================================
# LOAD REWRITER
# ============================================================

@st.cache_resource
def load_rewriter():

    # Load trained full Seq2Seq model
    model = tf.keras.models.load_model(
        REWRITER_MODEL_PATH
    )

    # Load tokenizers
    with open(SOURCE_TOKENIZER_PATH, "rb") as f:
        source_tokenizer = pickle.load(f)

    with open(TARGET_TOKENIZER_PATH, "rb") as f:
        target_tokenizer = pickle.load(f)

    # Load sequence information
    with open(SEQUENCE_INFO_PATH, "rb") as f:
        sequence_info = pickle.load(f)

    return (
        model,
        source_tokenizer,
        target_tokenizer,
        sequence_info
    )


# ============================================================
# BUILD INFERENCE MODELS
# ============================================================

@st.cache_resource
def build_rewriter():

    (
        model,
        source_tokenizer,
        target_tokenizer,
        sequence_info
    ) = load_rewriter()

    # --------------------------------------------------------
    # Sequence information
    # --------------------------------------------------------

    source_vocab_size = sequence_info["source_vocab_size"]
    target_vocab_size = sequence_info["target_vocab_size"]

    max_source_length = sequence_info["max_source_length"]
    max_target_length = sequence_info["max_target_length"]

    # --------------------------------------------------------
    # Get trained layers
    # --------------------------------------------------------

    encoder_embedding = model.get_layer(
        "encoder_embedding"
    )

    encoder_gru = model.get_layer(
        "encoder_gru"
    )

    decoder_embedding = model.get_layer(
        "decoder_embedding"
    )

    decoder_gru = model.get_layer(
        "decoder_gru"
    )

    decoder_output_layer = model.get_layer(
        "decoder_output"
    )

    latent_dim = int(
        encoder_gru.units
    )

    # --------------------------------------------------------
    # Encoder inference model
    # --------------------------------------------------------

    encoder_input = tf.keras.Input(
        shape=(max_source_length,),
        dtype=tf.int32,
        name="inference_encoder_input"
    )

    encoder_embedded = encoder_embedding(
        encoder_input
    )

    _, encoder_state = encoder_gru(
        encoder_embedded
    )

    encoder_model = tf.keras.Model(
        encoder_input,
        encoder_state
    )

    # --------------------------------------------------------
    # Decoder inference model
    # --------------------------------------------------------

    decoder_token_input = tf.keras.Input(
        shape=(1,),
        dtype=tf.int32,
        name="inference_decoder_token"
    )

    decoder_state_input = tf.keras.Input(
        shape=(latent_dim,),
        dtype=tf.float32,
        name="inference_decoder_state"
    )

    decoder_embedded = decoder_embedding(
        decoder_token_input
    )

    decoder_output_sequence, decoder_state = decoder_gru(
        decoder_embedded,
        initial_state=decoder_state_input
    )

    decoder_prediction = decoder_output_layer(
        decoder_output_sequence
    )

    decoder_model = tf.keras.Model(
        [
            decoder_token_input,
            decoder_state_input
        ],
        [
            decoder_prediction,
            decoder_state
        ]
    )

    return (
        encoder_model,
        decoder_model,
        source_tokenizer,
        target_tokenizer,
        max_source_length,
        max_target_length
    )


# ============================================================
# REWRITE TEXT
# ============================================================

def rewrite_text(text):

    (
        encoder_model,
        decoder_model,
        source_tokenizer,
        target_tokenizer,
        max_source_length,
        max_target_length
    ) = build_rewriter()

    # --------------------------------------------------------
    # Clean input
    # --------------------------------------------------------

    cleaned_text = clean_text(text)

    # --------------------------------------------------------
    # Tokenize source sentence
    # --------------------------------------------------------

    source_sequence = source_tokenizer.texts_to_sequences(
        [cleaned_text]
    )

    encoder_input = pad_sequences(
        source_sequence,
        maxlen=max_source_length,
        padding="post",
        truncating="post"
    ).astype(np.int32)

    # --------------------------------------------------------
    # Encode
    # --------------------------------------------------------

    state = encoder_model.predict(
        encoder_input,
        verbose=0
    ).astype(np.float32)

    # --------------------------------------------------------
    # START / END tokens
    # --------------------------------------------------------

    start_id = target_tokenizer.word_index.get(
        "<start>"
    )

    end_id = target_tokenizer.word_index.get(
        "<end>"
    )

    if start_id is None:
        raise ValueError(
            "<start> token was not found in target tokenizer."
        )

    if end_id is None:
        raise ValueError(
            "<end> token was not found in target tokenizer."
        )

    # --------------------------------------------------------
    # Initial decoder token
    # --------------------------------------------------------

    current_token = np.array(
        [[start_id]],
        dtype=np.int32
    )

    generated_words = []

    # --------------------------------------------------------
    # Greedy decoding
    # --------------------------------------------------------

    for _ in range(max_target_length - 1):

        prediction, state = decoder_model.predict(
            [
                current_token,
                state
            ],
            verbose=0
        )

        predicted_id = int(
            np.argmax(
                prediction[0, -1, :]
            )
        )

        # Stop conditions
        if predicted_id == end_id:
            break

        if predicted_id == 0:
            break

        # Convert token ID to word
        word = target_tokenizer.index_word.get(
            predicted_id,
            ""
        )

        if word:

            if word not in [
                "<start>",
                "<end>",
                "<OOV>"
            ]:

                generated_words.append(
                    word
                )

        # Feed predicted token back to decoder
        current_token = np.array(
            [[predicted_id]],
            dtype=np.int32
        )

    # --------------------------------------------------------
    # Final sentence
    # --------------------------------------------------------

    result = " ".join(
        generated_words
    ).strip()

    if not result:
        return None

    return result


# ============================================================
# SESSION STATE
# ============================================================

if "classification_result" not in st.session_state:
    st.session_state.classification_result = None

if "rewritten_text" not in st.session_state:
    st.session_state.rewritten_text = None

if "last_input_text" not in st.session_state:
    st.session_state.last_input_text = ""


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
    using a trained GRU-based NLP pipeline.
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

        Seq2Seq GRU encoder-decoder

        **Detection Threshold**

        0.50
        """
    )

    st.divider()

    st.caption(
        "The classifier produces a single sigmoid probability."
    )

    st.caption(
        "The rewriting model uses <start> and <end> tokens."
    )


# ============================================================
# EXAMPLE INPUTS
# ============================================================

st.subheader("Test Input")

example_columns = st.columns(3)

examples = [
    (
        "Toxic example",
        "I hate you and your stupid behavior."
    ),
    (
        "Another example",
        "People like you should leave this place."
    ),
    (
        "Safe example",
        "I really enjoyed the community event this weekend."
    )
]

for column, (button_text, example_text) in zip(
    example_columns,
    examples
):

    with column:

        if st.button(
            button_text,
            use_container_width=True
        ):

            st.session_state.input_text = example_text

            # Clear old results when changing example
            st.session_state.classification_result = None
            st.session_state.rewritten_text = None


# ============================================================
# TEXT INPUT
# ============================================================

if "input_text" not in st.session_state:

    st.session_state.input_text = ""


input_text = st.text_area(
    "Enter a sentence",
    value=st.session_state.input_text,
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

        st.warning(
            "Please enter some text first."
        )

    else:

        try:

            with st.spinner(
                "Running hate speech classifier..."
            ):

                result = classify_text(
                    input_text
                )

            # IMPORTANT:
            # Store result so it survives Streamlit reruns
            st.session_state.classification_result = result

            st.session_state.last_input_text = input_text

            # Clear previous rewrite
            st.session_state.rewritten_text = None

        except Exception as e:

            st.error(
                "Unable to run the classifier."
            )

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

        # ----------------------------------------------------
        # GENERATE NEUTRAL VERSION
        # ----------------------------------------------------

        generate_button = st.button(
            "Generate Neutral Version",
            type="primary",
            use_container_width=True
        )

        if generate_button:

            try:

                with st.spinner(
                    "Generating neutral version..."
                ):

                    rewritten = rewrite_text(
                        st.session_state.last_input_text
                    )

                # IMPORTANT:
                # Store generated text in session state.
                # This prevents it disappearing after rerun.
                st.session_state.rewritten_text = rewritten

            except Exception as e:

                st.session_state.rewritten_text = None

                st.error(
                    "Unable to generate the neutral version."
                )

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

        st.info(
            "The input does not require neutralization."
        )


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
        {rewritten_text}
        </div>

        </div>
        """,
        unsafe_allow_html=True
    )

    # Also provide normal Streamlit output
    st.success(
        rewritten_text
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

            **Architecture:** Seq2Seq GRU

            **Encoder:** GRU (256 units)

            **Decoder:** GRU (256 units)

            **Embedding:** 128 dimensions

            **Decoding:** Greedy decoding

            **Special tokens:** `<start>`, `<end>`
            """
        )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Hate Speech Neutralizer — Deep Learning Project"
)