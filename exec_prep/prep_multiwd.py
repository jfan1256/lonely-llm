import os
import pandas as pd

from utils.system import get_data

# Extract filenames from file path
def extract_filename(file_path):
    return os.path.basename(file_path)

# Split post column in test.csv
def split_post_question(row):
    post_part, question_part = row.split('Question:', 1)
    return pd.Series([post_part.strip(), question_part.strip()])

# Determine category
def determine_category(question):
    if "spiritual" in question:
        return 0
    elif "physical" in question:
        return 1
    elif "intellectual" in question:
        return 2
    elif "social" in question:
        return 3
    elif "vocational" in question:
        return 4
    elif "emotional" in question:
        return 5
    else:
        return -1

# Process dataframe for training
def process_dataframe(file_paths, output_file_paths):
    # Get file paths
    base_path = get_data()
    full_path = base_path / file_paths
    full_output_path = base_path / output_file_paths
    file_paths = list(full_path.glob('*.csv'))

    # Iterate through file paths
    for file_path in file_paths:
        # Read in dataframe
        file_name = extract_filename(file_path)
        output_file_path = full_output_path / file_name
        dataframe = pd.read_csv(file_path)

        # Handle test.csv
        if (file_name == "test.csv"):
            dataframe[['post', 'question']] = dataframe['post'].apply(split_post_question)

        # Rename columns
        dataframe.rename(columns={'post': 'narrative', 'response': 'reason'}, inplace=True)
        dataframe['narrative'] = dataframe['narrative'].str.replace('^Post:', '', regex=True).str.strip()

        # Determine category
        dataframe['category'] = dataframe['question'].apply(determine_category)
        dataframe = dataframe.drop('question', axis=1)

        # Format label and reason
        dataframe['label'] = dataframe['reason'].apply(lambda x: 0 if pd.notna(x) and x.split('.')[0] == "no" else 1)
        dataframe['reason'] = dataframe['reason'].str.replace(r'^(\s*no\s*\.|\s*yes\s*\.)\s*Reasoning:\s*', '', regex=True).str.strip()

        # Export
        dataframe.to_csv(output_file_path, index=False)

# Execute main
if __name__ == "__main__":
    process_dataframe("multiwd/raw/", "multiwd/process/")