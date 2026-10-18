import pickle
import numpy as np
import tensorflow as tf


# ============================================================
# LOAD MODEL
# ============================================================

model = tf.keras.models.load_model(
    "models/rewriting_seq2seq_gru.keras"
)

print("Model loaded successfully.")


# ============================================================
# LOAD TOKENIZERS
# ============================================================

with open("data/source_tokenizer.pkl", "rb") as f:
    source_tokenizer = pickle.load(f)

with open("data/target_tokenizer.pkl", "rb") as f:
    target_tokenizer = pickle.load(f)


# ============================================================
# SPECIAL TOKENS
# ============================================================

word_index = target_tokenizer.word_index

start_token_id = word_index.get("<start>")
end_token_id = word_index.get("<end>")

print("START TOKEN:", start_token_id)
print("END TOKEN:", end_token_id)

if start_token_id is None:
    raise ValueError(
        "<start> token is missing from target tokenizer."
    )

if end_token_id is None:
    raise ValueError(
        "<end> token is missing from target tokenizer."
    )


reverse_word_index = {
    value: key
    for key, value in word_index.items()
}


# ============================================================
# MODEL LAYERS
# ============================================================

encoder_inputs = model.get_layer(
    "encoder_inputs"
).input

encoder_embedding = model.get_layer(
    "encoder_embedding"
)

encoder_gru = model.get_layer(
    "encoder_gru"
)

encoder_embedded = encoder_embedding(
    encoder_inputs
)

_, encoder_state = encoder_gru(
    encoder_embedded
)

encoder_model = tf.keras.Model(
    encoder_inputs,
    encoder_state
)


# ============================================================
# DECODER
# ============================================================

decoder_token_input = tf.keras.Input(
    shape=(1,),
    dtype="int32",
    name="decoder_token_input"
)

decoder_state_input = tf.keras.Input(
    shape=(256,),
    dtype="float32",
    name="decoder_state_input"
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


decoder_embedded = decoder_embedding(
    decoder_token_input
)

decoder_output, decoder_state = decoder_gru(
    decoder_embedded,
    initial_state=decoder_state_input
)

decoder_output = decoder_output_layer(
    decoder_output
)


decoder_model = tf.keras.Model(
    [
        decoder_token_input,
        decoder_state_input
    ],
    [
        decoder_output,
        decoder_state
    ]
)


# ============================================================
# REWRITE FUNCTION
# ============================================================

def rewrite(text):

    # ----------------------------------------
    # Source tokenization
    # ----------------------------------------

    sequence = source_tokenizer.texts_to_sequences(
        [text.lower()]
    )

    encoder_input = tf.keras.preprocessing.sequence.pad_sequences(
        sequence,
        maxlen=17,
        padding="post",
        truncating="post"
    )

    encoder_input = np.asarray(
        encoder_input,
        dtype=np.int32
    )


    # ----------------------------------------
    # Encoder
    # ----------------------------------------

    state = encoder_model.predict(
        encoder_input,
        verbose=0
    )


    # ----------------------------------------
    # Start decoder
    # ----------------------------------------

    current_token = np.array(
        [[start_token_id]],
        dtype=np.int32
    )

    generated_words = []


    # ----------------------------------------
    # Generate
    # ----------------------------------------

    for _ in range(16):

        output, state = decoder_model.predict(
            [
                current_token,
                state
            ],
            verbose=0
        )

        token_id = int(
            np.argmax(
                output[0, 0, :]
            )
        )


        # END token
        if token_id == end_token_id:
            break


        # Padding
        if token_id == 0:
            break


        word = reverse_word_index.get(
            token_id,
            ""
        )


        if word not in [
            "",
            "<start>",
            "<end>"
        ]:
            generated_words.append(
                word
            )


        current_token = np.array(
            [[token_id]],
            dtype=np.int32
        )


    return " ".join(
        generated_words
    )


# ============================================================
# TEST
# ============================================================

tests = [
    "I hate you and your stupid behavior.",
    "You are a disgusting idiot.",
    "People like you should leave this place."
]


for text in tests:

    print("\n" + "=" * 60)

    print("INPUT:")
    print(text)

    result = rewrite(text)

    print("\nOUTPUT:")
    print(result)