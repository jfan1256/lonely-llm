import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from utils.system import get_data
from sklearn.metrics import f1_score, recall_score
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import brotli
import pickle

### Load data file
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

### Load feature file
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

### Train Linear SVM Model
# Create a pipeline with StandardScaler and SVM
svm_model = Pipeline([
    ('scaler', StandardScaler()),
    ('svm', SVC(kernel='linear', probability=True))
])

# Train the model
print("Start Train Linear SVM model...")
svm_model.fit(X_train, y_train)

# Predictions
y_pred_test = svm_model.predict_proba(X_test)[:, 1]
y_pred_eval = svm_model.predict_proba(X_eval)[:, 1]

# Convert probabilities to binary predictions
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