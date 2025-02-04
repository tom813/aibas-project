import os
import sys
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers

# ------------------------------------------------------------------------------
# Paths INSIDE the container:
# We expect the activation data is copied to /tmp/activationBase
# We expect the model is copied to /tmp/knowledgeBase
# ------------------------------------------------------------------------------
MODEL_PATH = "/tmp/knowledgeBase/sarcasm_model.h5"
ACTIVATION_PATH = "/tmp/activationBase/activation_data.csv"

# ------------------------------------------------------------------------------
# Hyperparameters for vectorization (must match what you used in training)
# ------------------------------------------------------------------------------
MAX_TOKENS = 1000
SEQUENCE_LENGTH = 50

def load_model(model_path: str):
    print(f"Loading model from: {model_path}")
    # Load without compiling to avoid deserializing the metric functions
    return tf.keras.models.load_model(model_path, compile=False)

def load_activation_data(csv_path: str):
    print(f"Loading activation data from: {csv_path}")
    df = pd.read_csv(csv_path)
    return df

def predict_inference(model, df):
    """
    1) Adapt TextVectorization on the single activation file itself.
    2) Vectorize those tweets to integer sequences.
    3) Model expects integer sequences (from its Embedding layer).
    4) Print predictions.
    """
    tweets = df["tweet"].astype(str).values

    # Build a vectorizer that matches your training parameters
    vectorizer = layers.TextVectorization(
        max_tokens=MAX_TOKENS,
        output_mode='int',
        output_sequence_length=SEQUENCE_LENGTH
    )

    # WARNING: Adapting on activation_data.csv alone means the vocabulary
    #          won't match full training. This is purely a demonstration.
    vectorizer.adapt(tweets)

    # Convert raw text to integer sequences
    tweets_vec = vectorizer(tweets)

    # Model expects integer sequences
    predictions = model.predict(tweets_vec)

    # Print predictions
    for i, pred in enumerate(predictions):
        label = "Sarcastic" if pred[0] >= 0.5 else "Not Sarcastic"
        print(f"Tweet: {tweets[i]}")
        print(f"Prediction: {label} (score: {pred[0]:.4f})\n")

if __name__ == "__main__":
    model = load_model(MODEL_PATH)
    df_act = load_activation_data(ACTIVATION_PATH)
    predict_inference(model, df_act)
