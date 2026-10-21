import tensorflow as tf
import pickle
import numpy as np

MODEL_PATH = "models/gru_hate_speech_model.h5"
TOKENIZER_PATH = "data/tokenizer.pkl"

# Load model
model = tf.keras.models.load_model(
    MODEL_PATH,
    compile=False
)

# Load tokenizer
with open(TOKENIZER_PATH, "rb") as f:
    tokenizer = pickle.load(f)

print("\n========== MODEL ==========")
model.summary()

print("\nInput shape:")
print(model.input_shape)

print("\nOutput shape:")
print(model.output_shape)

print("\n========== TOKENIZER ==========")
print("Vocabulary size:", len(tokenizer.word_index))

print("\n========== TEST ==========")

texts = [
    "You are a disgusting idiot and nobody wants you here.",
    "I hate you and your stupid behavior.",
    "People like you should leave this place.",
    "I really enjoyed the community event this weekend.",
    "The weather is beautiful today."
]

sequences = tokenizer.texts_to_sequences(texts)

print("\nSequences:")
for text, seq in zip(texts, sequences):
    print("\nTEXT:", text)
    print("SEQ :", seq[:30])

# IMPORTANT:
# Your original classifier preprocessing used max length 200.
MAX_LEN = 200

padded = tf.keras.preprocessing.sequence.pad_sequences(
    sequences,
    maxlen=MAX_LEN,
    padding="post",
    truncating="post"
)

predictions = model.predict(
    padded,
    verbose=0
)

print("\n========== PREDICTIONS ==========")

for text, prediction in zip(texts, predictions):

    print("\nTEXT:")
    print(text)

    print("RAW OUTPUT:")
    print(prediction)

    print("OUTPUT SHAPE:")
    print(np.asarray(prediction).shape)

print("\n=================================")