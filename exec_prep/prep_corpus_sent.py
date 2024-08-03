import warnings
warnings.filterwarnings('ignore')

import os
import re
import pandas as pd
import nltk
from nltk.tokenize import sent_tokenize

from utils.system import get_data

# Ensure you have the necessary tokenizer models downloaded
nltk.download('punkt')

# Extract filenames from file path
def extract_filename(file_path):
    return os.path.basename(file_path)

def tokenize_sentences(text):
    return sent_tokenize(text)

def preprocess_text(text):
    # Convert to str
    text = str(text)

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

        # Remove rows where 'content' is entirely float values
        filtered_df = filtered_df[~filtered_df['content'].apply(lambda x: isinstance(x, float))]

        # Convert all content to strings
        filtered_df['content'] = filtered_df['content'].apply(str)

        # Preprocess the 'content' column after initial cleaning
        filtered_df['content'] = filtered_df['content'].apply(preprocess_text)

        # Apply the function to the relevant column
        filtered_df['sentences'] = filtered_df['content'].apply(tokenize_sentences)

        # Optionally, expand the list of sentences into separate rows# Expand the list of sentences into separate rows with a unique ID for each row
        rows = []
        for index, row in filtered_df.iterrows():
            for sentence in row['sentences']:
                # Create a unique ID for each sentence, using the original row index and the sentence's position
                sentence_id = f"{index}_{row['sentences'].index(sentence)}"
                rows.append({'id': sentence_id, 'sentence': sentence})

        new_df = pd.DataFrame(rows)

        # Get output file path
        file_name = extract_filename(file_path)
        output_file_path = full_output_path / file_name

        print(new_df)
        print(output_file_path)

        # Write the filtered DataFrame to a new CSV file
        new_df.to_csv(output_file_path, index=False)

# Example usage
if __name__ == "__main__":
    filter_content_column('corpus/MentaRedditRawData-2008-2023', 'corpus/ProcessedRedditRawDataInSentence')