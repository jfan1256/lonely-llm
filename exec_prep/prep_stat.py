import os
import pandas as pd
from utils.system import get_data

# Get file paths
base_path = get_data()
# folder_path = base_path / "corpus" / "ProcessedRedditRawData"
# new_path = base_path / "corpus" / "StatsOfFiles"
folder_path = base_path / "loneliness" / "ProcessedRedditRawData"
new_path = base_path / "loneliness" / "StatsOfFiles"

# Initialize variables to store results
all_word_counts = []
total_posts_all_csvs = 0
results = []

# Iterate through all CSV files in the folder
for filename in os.listdir(folder_path):
    if filename.endswith('.csv'):
        # Load the CSV file
        file_path = os.path.join(folder_path, filename)
        df = pd.read_csv(file_path)

        # Calculate word count for each post
        df['word_count'] = df['content'].apply(lambda x: len(str(x).split()))

        # Calculate mean and std of word count
        mean_word_count = df['word_count'].mean()
        std_word_count = df['word_count'].std()

        # Total number of posts in the current CSV
        total_posts_csv = len(df)

        # Store results
        all_word_counts.extend(df['word_count'].tolist())
        total_posts_all_csvs += total_posts_csv

        # Append results to the list
        results.append({
            'File': filename,
            'Mean Word Count': mean_word_count,
            'Standard Deviation of Word Count': std_word_count,
            'Total Posts': total_posts_csv
        })

        # Print results for the current CSV
        print(f'File: {filename}')
        print(f'Mean word count: {mean_word_count:.2f}')
        print(f'Standard deviation of word count: {std_word_count:.2f}')
        print(f'Total posts in {filename}: {total_posts_csv}\n')

# Calculate overall statistics across all CSVs
mean_word_count_all = pd.Series(all_word_counts).mean()
std_word_count_all = pd.Series(all_word_counts).std()

# Append overall results to the list
results.append({
    'File': 'Overall',
    'Mean Word Count': mean_word_count_all,
    'Standard Deviation of Word Count': std_word_count_all,
    'Total Posts': total_posts_all_csvs
})

# Create a DataFrame from the results
results_df = pd.DataFrame(results)

# Ensure the output directory exists
os.makedirs(new_path, exist_ok=True)

# Save the results to a new CSV file
output_file = new_path / 'word_count_analysis.csv'
results_df.to_csv(output_file, index=False)

print(f'Results saved to {output_file}')