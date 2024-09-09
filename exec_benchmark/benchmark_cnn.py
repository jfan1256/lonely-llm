import warnings
warnings.filterwarnings('ignore')

import torch
import brotli
import pickle
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as DataLoader

from sklearn.metrics import f1_score, recall_score

from utils.system import get_data

# Load Brotli files
def load_brotli(filename):
    with open(filename, 'rb') as f:
        compressed_data = f.read()
        data = pickle.loads(brotli.decompress(compressed_data))
        return data

# CNN model
class CNNModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=128, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=128, out_channels=64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)

        # Calculate the size after conv and pool layers
        def calc_conv_output_size(input_size, kernel_size, stride=1, padding=0):
            return (input_size - kernel_size + 2 * padding) // stride + 1

        conv1_output_size = calc_conv_output_size(input_dim, 3, padding=1)
        pool1_output_size = conv1_output_size // 2
        conv2_output_size = calc_conv_output_size(pool1_output_size, 3, padding=1)
        pool2_output_size = conv2_output_size // 2

        flattened_size = 64 * pool2_output_size

        self.fc1 = nn.Linear(flattened_size, 32)
        self.fc2 = nn.Linear(32, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return torch.sigmoid(x)

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

    # Convert to PyTorch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32)
    X_eval_tensor = torch.tensor(X_eval, dtype=torch.float32)
    y_eval_tensor = torch.tensor(y_eval, dtype=torch.float32)

    # Create PyTorch datasets
    train_dataset = DataLoader.TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = DataLoader.TensorDataset(X_test_tensor, y_test_tensor)
    eval_dataset = DataLoader.TensorDataset(X_eval_tensor, y_eval_tensor)

    # Create DataLoader
    batch_size = 32
    train_loader = DataLoader.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    eval_loader = DataLoader.DataLoader(eval_dataset, batch_size=batch_size, shuffle=False)

    # Initialize the model, loss function, and optimizer
    input_dim = X_train.shape[1]
    output_dim = 1
    model = CNNModel(input_dim=input_dim, output_dim=output_dim)

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()
        for i, (inputs, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), labels)
            loss.backward()
            optimizer.step()

        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item()}")

    # Evaluation on Test and Eval sets
    model.eval()
    with torch.no_grad():
        y_pred_test = []
        y_pred_eval = []

        for inputs, _ in test_loader:
            outputs = model(inputs)
            y_pred_test.append(outputs.squeeze().numpy())

        for inputs, _ in eval_loader:
            outputs = model(inputs)
            y_pred_eval.append(outputs.squeeze().numpy())

        y_pred_test = np.concatenate(y_pred_test)
        y_pred_eval = np.concatenate(y_pred_eval)

    # Apply threshold to convert probabilities to binary predictions
    y_pred_test_binary = (y_pred_test > 0.5).astype(int)
    y_pred_eval_binary = (y_pred_eval > 0.5).astype(int)

    # Calculate F1 and recall scores
    f1_test = f1_score(y_test, y_pred_test_binary)
    recall_test = recall_score(y_test, y_pred_test_binary)
    # Calculate recall score for the evaluation set
    f1_eval = f1_score(y_eval, y_pred_eval_binary)
    recall_eval = recall_score(y_eval, y_pred_eval_binary)
    print(f"F1 Score on Test Set: {f1_test}")
    print(f"F1 Score on Eval Set: {f1_eval}")
    print(f"Recall Score on Test Set: {recall_test}")
    print(f"Recall Score on Eval Set: {recall_eval}")