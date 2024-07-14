import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np

# Specify the file path
file_path = r"/data/loneliness/OurLabeledData\\"
file_name = "human_annotations_all_8000_overall.csv"
#new_file_name = "processed_loneliness-test.csv"

# Concatenate file_path and file_name to get the complete file path
full_file_path = file_path + file_name

print("Full file path:", full_file_path)

# Load the CSV file into a DataFrame
df = pd.read_csv(full_file_path)

print(df.head())

# Separate into positive and negative labels
positive_df = df[df['overall_label'] == 1]
negative_df = df[df['overall_label'] == 0]

# Randomly sample 80% for training and 20% for testing from positive_df
positive_train = positive_df.sample(frac=0.8, random_state=42)
positive_test = positive_df.drop(positive_train.index)

# Randomly sample 80% for training and 20% for testing from negative_df
negative_train = negative_df.sample(frac=0.8, random_state=42)
negative_test = negative_df.drop(negative_train.index)

# Concatenate the sampled dataframes into final train and test dataframes
train_df = pd.concat([positive_train, negative_train])
test_df = pd.concat([positive_test, negative_test])

# Displaying the sizes of train and test dataframes
print("Train dataframe size:", len(train_df))
print("Test dataframe size:", len(test_df))

# Separate into positive and negative labels
positive_df = train_df[train_df['overall_label'] == 1]
negative_df = train_df[train_df['overall_label'] == 0]

# Randomly sample 80% for training and 20% for testing from positive_df
positive_train = positive_df.sample(frac=0.8, random_state=42)
positive_val = positive_df.drop(positive_train.index)

# Randomly sample 80% for training and 20% for testing from negative_df
negative_train = negative_df.sample(frac=0.8, random_state=42)
negative_val = negative_df.drop(negative_train.index)

# Concatenate the sampled dataframes into final train and test dataframes
train_df = pd.concat([positive_train, negative_train])
val_df = pd.concat([positive_val, negative_val])

# Displaying the sizes of train and test dataframes
print("Train dataframe size:", len(train_df))
print("Test dataframe size:", len(val_df))

train_file_name = "loneliness-train-2.csv"
test_file_name = "loneliness-test-2.csv"
val_file_name = "loneliness-val-2.csv"

train_file_path = file_path + train_file_name
test_file_path = file_path + test_file_name
val_file_path = file_path + val_file_name

# Optionally, you can save the train and test dataframes to CSV files
train_df.to_csv(train_file_path, index=False)
test_df.to_csv(test_file_path, index=False)
val_df.to_csv(val_file_path, index=False)