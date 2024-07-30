#Import Functions
import os
import re
import pandas as pd

from utils.system import get_data

# Extract filenames from file path
def extract_filename(file_path):
    return os.path.basename(file_path)

def preprocess_text(text):
    # Convert text to lowercase
    text = text.lower()
    # Remove URLs
    text = re.sub(r'http[s]?://\S+', '', text)
    # Remove special characters except basic punctuation
    text = re.sub(r'[^a-z0-9\s,.?!-]', '', text)
    return text

def filter_content_column(file_paths, output_file_paths):
    # Get base paths for input and output
    base_path = get_data()
    full_path = base_path / file_paths
    full_output_path = base_path / output_file_paths

    # Ensure output directory exists
    if not full_output_path.exists():
        full_output_path.mkdir(parents=True, exist_ok=True)

    # List CSV files in the directory
    file_paths = list(full_path.glob('*.csv'))
    print(file_paths)

    # Iterate through file paths
    for file_path in file_paths:
        # Read in dataframe
        dataframe = pd.read_csv(file_path)

        # Keep only the 'content' column
        filtered_df = dataframe[['content']]

        # Remove rows where 'content' is empty, contains '[deleted]', '[removed]', or just a period
        conditions = ~filtered_df['content'].isin(['[deleted]', '[removed]']) & (filtered_df['content'].str.strip() != '.') & (filtered_df['content'].str.strip() != '')
        filtered_df = filtered_df[conditions]

        # Remove rows with NaN values
        filtered_df = filtered_df.dropna()

        # Preprocess the 'content' column after initial cleaning
        filtered_df['content'] = filtered_df['content'].apply(preprocess_text)

        # Get output file path
        file_name = extract_filename(file_path)
        output_file_path = full_output_path / file_name

        print(filtered_df)
        print(output_file_path)

        # Write the filtered DataFrame to a new CSV file
        filtered_df.to_csv(output_file_path, index=False)

# Example usage
if __name__ == "__main__":
    filter_content_column('OriginalData/MentaRedditRawData-2008-2023', 'OriginalData/ProcessedRedditRawData')