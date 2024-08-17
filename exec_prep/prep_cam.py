import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np

from utils.system import get_data

# Concatenate file_path and file_name to get the complete file path
full_file_path = get_data() / "CAMS" / "CAMS.csv"

print("Full file path:", full_file_path)

# Load the CSV file into a DataFrame
df = pd.read_csv(full_file_path)

print(df.head())

# Separate into different category labels
category_0_df = df[df['label'] == 0]
category_1_df = df[df['label'] == 1]
category_2_df = df[df['label'] == 2]
category_3_df = df[df['label'] == 3]
category_4_df = df[df['label'] == 4]
category_5_df = df[df['label'] == 5]

# Randomly sample 80% for training and 20% for testing from each category
category_0_train = category_0_df.sample(frac=0.8, random_state=42)
category_0_test = category_0_df.drop(category_0_train.index)

category_1_train = category_1_df.sample(frac=0.8, random_state=42)
category_1_test = category_1_df.drop(category_1_train.index)

category_2_train = category_2_df.sample(frac=0.8, random_state=42)
category_2_test = category_2_df.drop(category_2_train.index)

category_3_train = category_3_df.sample(frac=0.8, random_state=42)
category_3_test = category_3_df.drop(category_3_train.index)

category_4_train = category_4_df.sample(frac=0.8, random_state=42)
category_4_test = category_4_df.drop(category_4_train.index)

category_5_train = category_5_df.sample(frac=0.8, random_state=42)
category_5_test = category_5_df.drop(category_5_train.index)

# Concatenate the sampled dataframes into final train and test dataframes
train_df = pd.concat([category_0_train, category_1_train, category_2_train, category_3_train, category_4_train, category_5_train])
test_df = pd.concat([category_0_test, category_1_test, category_2_test, category_3_test, category_4_test, category_5_test])

# Displaying the sizes of train and test dataframes
print("Train dataframe size:", len(train_df))
print("Test dataframe size:", len(test_df))

# Separate into different category labels
category_0_df = train_df[train_df['label'] == 0]
category_1_df = train_df[train_df['label'] == 1]
category_2_df = train_df[train_df['label'] == 2]
category_3_df = train_df[train_df['label'] == 3]
category_4_df = train_df[train_df['label'] == 4]
category_5_df = train_df[train_df['label'] == 5]

# Randomly sample 80% for training and 20% for testing from each category
category_0_train = category_0_df.sample(frac=0.8, random_state=42)
category_0_val = category_0_df.drop(category_0_train.index)

category_1_train = category_1_df.sample(frac=0.8, random_state=42)
category_1_val = category_1_df.drop(category_1_train.index)

category_2_train = category_2_df.sample(frac=0.8, random_state=42)
category_2_val = category_2_df.drop(category_2_train.index)

category_3_train = category_3_df.sample(frac=0.8, random_state=42)
category_3_val = category_3_df.drop(category_3_train.index)

category_4_train = category_4_df.sample(frac=0.8, random_state=42)
category_4_val = category_4_df.drop(category_4_train.index)

category_5_train = category_5_df.sample(frac=0.8, random_state=42)
category_5_val = category_5_df.drop(category_5_train.index)

# Concatenate the sampled dataframes into final train and test dataframes
train_df = pd.concat([category_0_train, category_1_train, category_2_train, category_3_train, category_4_train, category_5_train])
val_df = pd.concat([category_0_val, category_1_val, category_2_val, category_3_val, category_4_val, category_5_val])

# Displaying the sizes of train and test dataframes
print("Train dataframe size:", len(train_df))
print("Test dataframe size:", len(val_df))

train_file_name = "cams-train.csv"
test_file_name = "cams-test.csv"
val_file_name = "cams-val.csv"

file_path = get_data() / "CAMS" / 'SamplingData-2'

train_file_path = file_path / train_file_name
test_file_path = file_path / test_file_name
val_file_path = file_path / val_file_name

# Optionally, you can save the train and test dataframes to CSV files
train_df.to_csv(train_file_path, index=False)
test_df.to_csv(test_file_path, index=False)
val_df.to_csv(val_file_path, index=False)