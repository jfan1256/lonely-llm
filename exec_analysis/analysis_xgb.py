import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import xgboost as xgb
from utils.system import get_data
from sklearn.metrics import f1_score, recall_score
import brotli
import pickle

###Load data file
# Load the CSV file into a DataFrame
train_file_path = get_data() / "loneliness" / "OurLabeledData" / "SamplingData-1" / "loneliness-train-1.csv"
test_file_path = get_data() / "loneliness" / "OurLabeledData" / "SamplingData-1" / "loneliness-test-1.csv"
val_file_path = get_data() / "loneliness" / "OurLabeledData" / "SamplingData-1" / "loneliness-val-1.csv"
# Load the data
train_df = pd.read_csv(train_file_path)
test_df = pd.read_csv(test_file_path)
eval_df = pd.read_csv(val_file_path)

def load_brotli(filename):
    with open(filename, 'rb') as f:
        compressed_data = f.read()
        data = pickle.loads(brotli.decompress(compressed_data))
        return data

###Load feature file
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


# Convert to DMatrix
d_train = xgb.DMatrix(X_train, label=y_train)
d_test = xgb.DMatrix(X_test, label=y_test)
d_eval = xgb.DMatrix(X_eval, label=y_eval)

###Train XGBoost Model
params = {
    'objective': 'binary:logistic',  # or 'multi:softmax' for multi-class
    'max_depth': 6,
    'eta': 0.1,
    'eval_metric': 'logloss'  # or 'mlogloss' for multi-class
}

# List of datasets to evaluate during training
eval_list = [(d_train, 'train'), (d_test, 'test'), (d_eval, 'eval')]


# Train the model
print("Start Train XGBoost model...")
num_round = 100
model = xgb.train(params, d_train, num_boost_round=num_round, evals=eval_list, early_stopping_rounds=25, verbose_eval=1 )

# Predictions
y_pred_test = model.predict(d_test)
y_pred_eval = model.predict(d_eval)

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
