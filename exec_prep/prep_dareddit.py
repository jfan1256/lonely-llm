import warnings
warnings.filterwarnings('ignore')

import pandas as pd

from utils.system import get_data

# Specify the file path
file_path = get_data() / "dreaddit" / "MentalLLaMA"
# file_name = "dreaddit-train.csv"
# file_name = "dreaddit-val.csv"
file_name = "dreaddit-test.csv"
full_file_path = file_path / file_name
print("Full file path:", full_file_path)

new_file_path = get_data() / "dreaddit" / "Processed"
new_file_name = f"processed_{file_name}"
full_new_file_path = new_file_path / new_file_name

# Load the CSV file into a DataFrame
df = pd.read_csv(full_file_path)

# Initialize empty lists to store the cleaned narratives, labels, and reasoning
narratives = []
labels = []
reasons = []

for index, row in df.iterrows():
    post_text = row['post']
    response_text = row['response']

    # Remove the leading "Post:" and any text following "Question:" if present
    post_start_index = post_text.find("Post: ") + len("Post: ") if "Post: " in post_text else 0
    question_start_index = post_text.find(" Question:")
    narrative = post_text[post_start_index:question_start_index if question_start_index != -1 else None].strip()

    # Extract label and reasoning from response_text
    end_index = response_text.find(". Reasoning:")
    if end_index != -1:
        response = response_text[:end_index].strip()
        label = 1 if response == "yes" else 0
        reasoning_start_index = response_text.find(". Reasoning:") + len(". Reasoning:")
        reasoning = response_text[reasoning_start_index:].strip()
    else:
        label = 0
        reasoning = ""

    # Append data to lists
    narratives.append(narrative)
    labels.append(label)
    reasons.append(reasoning)

# Create a new dataframe from the lists
new_df = pd.DataFrame({
    'narrative': narratives,
    'label': labels,
    'reason': reasons
})

# Save the new dataframe to a CSV file
new_df.to_csv(full_new_file_path, index=False)