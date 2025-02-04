import os
import sys
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import statsmodels.api as sm

# ------------------------------------------------------------------------------
# Paths INSIDE the container:
# We expect the activation data is copied to /tmp/activationBase
# We expect the OLS model is copied to /tmp/knowledgeBase
# ------------------------------------------------------------------------------
MODEL_PATH = "/tmp/knowledgeBase/currentOlsSoluGon.pickle"
ACTIVATION_PATH = "/tmp/activationBase/activation_data.csv"

# ------------------------------------------------------------------------------
# Hyperparameters for vectorization (must match what you used in training)
# ------------------------------------------------------------------------------
MAX_TOKENS = 1000
SEQUENCE_LENGTH = 50

def load_model(model_path: str):
    print(f"Loading OLS model from: {model_path}")
    # Use statsmodels' load function (this assumes your model was saved via results.save(...))
    return sm.load(model_path)

def load_activation_data(csv_path: str):
    print(f"Loading activation data from: {csv_path}")
    df = pd.read_csv(csv_path)
    return df

def predict_inference(ols_results, df):
    """
    For each tweet in the activation data:
      1. Adapt a TextVectorization layer on the tweets (for demonstration).
      2. Convert the raw tweet strings to integer sequences.
      3. Add a constant column (to match the training data).
      4. Use the loaded OLS model to predict.
      5. Print the predictions.
    """
    tweets = df["tweet"].astype(str).values

    # Build a TextVectorization layer (this should match the training configuration)
    vectorizer = layers.TextVectorization(
        max_tokens=MAX_TOKENS,
        output_mode='int',
        output_sequence_length=SEQUENCE_LENGTH
    )
    
    # WARNING: Adapting on the activation data alone means the vocabulary
    # will likely differ from training. This is acceptable here for demonstration.
    vectorizer.adapt(tweets)
    
    # Convert the raw text tweets into integer sequences
    tweets_tensor = vectorizer(tweets)  # This returns a tf.Tensor
    tweets_int = tweets_tensor.numpy()   # Convert to a NumPy array

    # In training, you added a constant column to the integer sequences.
    # Do the same here so the input matches what the OLS model expects.
    X_activation = sm.add_constant(tweets_int, has_constant="add")
    
    # Predict using the loaded OLS model
    predictions = ols_results.predict(X_activation)
    
    # Print predictions
    for i, pred in enumerate(predictions):
        # For demonstration, we threshold at 0.5
        label = "Sarcastic" if pred >= 0.5 else "Not Sarcastic"
        print(f"Tweet: {tweets[i]}")
        print(f"Prediction: {label} (score: {pred:.4f})\n")

if __name__ == "__main__":
    ols_results = load_model(MODEL_PATH)
    df_act = load_activation_data(ACTIVATION_PATH)
    predict_inference(ols_results, df_act)
