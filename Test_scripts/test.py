# import pickle
# import numpy as np
# import matplotlib.pyplot as plt

# from tensorflow.keras.models import load_model
# from sklearn.metrics import (
#     classification_report,
#     confusion_matrix,
#     ConfusionMatrixDisplay
# )

# # Load test data
# with open("data/test_data.pkl", "rb") as f:
#     X_test, y_test = pickle.load(f)

# # Load trained model
# model = load_model("models/hate_speech_model.h5")

# # Predictions
# y_prob = model.predict(X_test)
# y_pred = (y_prob >= 0.5).astype(int).flatten()

# # Classification report
# print("\nClassification Report:")
# print(
#     classification_report(
#         y_test,
#         y_pred,
#         target_names=["Non-Toxic", "Toxic"]
#     )
# )

# # Confusion matrix
# cm = confusion_matrix(y_test, y_pred)

# print("\nConfusion Matrix:")
# print(cm)

# # Display confusion matrix
# disp = ConfusionMatrixDisplay(
#     confusion_matrix=cm,
#     display_labels=["Non-Toxic", "Toxic"]
# )

# disp.plot()
# plt.title("LSTM Confusion Matrix")
# plt.show()


import pickle
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load test data
with open(r"E:\SEM5\DipLerning\DL_Project\HateSpeechDetection\data\test_data.pkl", "rb") as f:
    X_test, y_test = pickle.load(f)


def evaluate_model(model_path, model_name):
    model = load_model(model_path)

    # Predictions
    probabilities = model.predict(X_test, verbose=0)
    predictions = (probabilities >= 0.5).astype(int).flatten()

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)

    print(f"\n{model_name}")
    print("-" * 30)
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")

    return precision, recall, f1

from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

# Load final GRU
model = load_model(r"E:\SEM5\DipLerning\DL_Project\HateSpeechDetection\models\bigru_hate_speech_model.h5")

# Predictions
probabilities = model.predict(X_test, verbose=0)
predictions = (probabilities >= 0.5).astype(int).flatten()

# Classification report
print("\n========== CLASSIFICATION REPORT ==========")

print(
    classification_report(
        y_test,
        predictions,
        target_names=["Non-Toxic", "Toxic"]
    )
)

# Confusion matrix
print("\n========== CONFUSION MATRIX ==========")

cm = confusion_matrix(y_test, predictions)

print(cm)

print("\nTN:", cm[0][0])
print("FP:", cm[0][1])
print("FN:", cm[1][0])
print("TP:", cm[1][1])


# # Evaluate LSTM
# gru_scores = evaluate_model(
#     "models/gru_hate_speech_model.h5",
#     "GRU"
# )

# # Evaluate GRU
# bigru_scores = evaluate_model(
#     "models/bigru_hate_speech_model.h5",
#     "biGRU"
# )

# # Comparison
# print("\n========== COMPARISON ==========")

# print(f"BiGRU F1 Score: {bigru_scores[2]:.4f}")
# print(f"GRU  F1 Score: {gru_scores[2]:.4f}")