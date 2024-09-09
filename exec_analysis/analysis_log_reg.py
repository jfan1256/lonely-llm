import warnings
warnings.filterwarnings('ignore')

import brotli
import pickle
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, recall_score
from sklearn.linear_model import LogisticRegression

from utils.system import get_data

# Load Brotli files
def load_brotli(filename):
    with open(filename, 'rb') as f:
        compressed_data = f.read()
        data = pickle.loads(brotli.decompress(compressed_data))
        return data

if __name__ == '__main__':
    # Load the CSV file into a DataFrame
    train_file_path = get_data() / "loneliness" / "OurLabeledData" / "SamplingData-1" / "loneliness-train-1.csv"
    test_file_path = get_data() / "loneliness" / "OurLabeledData" / "SamplingData-1" / "loneliness-test-1.csv"
    val_file_path = get_data() / "loneliness" / "OurLabeledData" / "SamplingData-1" / "loneliness-val-1.csv"
    # Load the data
    train_df = pd.read_csv(train_file_path)
    test_df = pd.read_csv(test_file_path)
    eval_df = pd.read_csv(val_file_path)

    # Load features from Brotli-compressed files
    print("Loading Train Features from Brotli file...")
    train_features = load_brotli('features/train_features.brotli')

    print("Loading Test Features from Brotli file...")
    test_features = load_brotli('features/test_features.brotli')

    print("Loading Eval Features from Brotli file...")
    eval_features = load_brotli('features/eval_features.brotli')

    # Extract labels
    y_train = train_df['label'].values
    y_test = test_df['label'].values
    y_eval = eval_df['label'].values

    # Convert the features to NumPy arrays
    X_train = np.array(train_features)
    X_test = np.array(test_features)
    X_eval = np.array(eval_features)

    print(X_train.shape)
    print(X_test.shape)
    print(X_eval.shape)

    # Standardize the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    X_eval = scaler.transform(X_eval)

    # Train Logistic Regression Model
    log_reg = LogisticRegression(max_iter=1000)
    print("Start Training Logistic Regression model...")
    log_reg.fit(X_train, y_train)

    # Predictions
    y_pred_test = log_reg.predict_proba(X_test)[:, 1]
    y_pred_eval = log_reg.predict_proba(X_eval)[:, 1]

    # You can apply a threshold to convert probabilities to binary predictions if needed
    y_pred_test_binary = (y_pred_test > 0.5).astype(int)
    y_pred_eval_binary = (y_pred_eval > 0.5).astype(int)

    # Calculate F1 score for the test set
    f1_test = f1_score(y_test, y_pred_test_binary)
    # Calculate recall score for the test set
    recall_test = recall_score(y_test, y_pred_test_binary)

    # Calculate F1 score for the eval set
    f1_eval = f1_score(y_eval, y_pred_eval_binary)
    # Calculate recall score for the evaluation set
    recall_eval = recall_score(y_eval, y_pred_eval_binary)
    print(f"F1 Score on Test Set: {f1_test}")
    print(f"F1 Score on Eval Set: {f1_eval}")
    print(f"Recall Score on Test Set: {recall_test}")
    print(f"Recall Score on Eval Set: {recall_eval}")