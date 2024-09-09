import warnings
warnings.filterwarnings("ignore")

import pandas as pd

from sklearn.metrics import jaccard_score
from sklearn.preprocessing import MultiLabelBinarizer

# Compute jaccard score
def compute_jaccard_score(df, col1, col2):
    # Convert text to set of words
    df['set1'] = df[col1].apply(lambda x: set(str(x).split()))
    df['set2'] = df[col2].apply(lambda x: set(str(x).split()))

    # Create binarized sets for Jaccard calculation using MultiLabelBinarizer
    mlb = MultiLabelBinarizer()
    all_sets = pd.concat([df['set1'], df['set2']]).tolist()
    mlb.fit(all_sets)

    # Calculate Jaccard score for each row
    df['jaccard_score'] = df.apply(lambda x: jaccard_score(mlb.transform([x['set1']])[0], mlb.transform([x['set2']])[0]), axis=1)

    # Drop the temporary columns
    df.drop(columns=['set1', 'set2'], inplace=True)
    return df

if __name__ == "__main__":
    # Compute Jaccard Score
    file_path = '../paper_experiment/loneliness/best_classifier_decoder_only_all/eval.csv'
    data = pd.read_csv(file_path)
    df = pd.DataFrame(data)
    result_df = compute_jaccard_score(df, 'reason', 'pred_reason')
    print("Loneliness:")
    print('mean jaccard score: {:2f}'.format(result_df['jaccard_score'].mean()))
    print('std jaccard score: {:2f}'.format(result_df['jaccard_score'].std()))
    print('max jaccard score: {:2f}'.format(result_df['jaccard_score'].max()))
    print('min jaccard score: {:2f}'.format(result_df['jaccard_score'].min()))

    file_path = '../paper_experiment/dreaddit/best_classifier_decoder_only_all/eval.csv'
    data = pd.read_csv(file_path)
    df = pd.DataFrame(data)
    result_df = compute_jaccard_score(df, 'reason', 'pred_reason')
    print("Dreaddit:")
    print('mean jaccard score: {:2f}'.format(result_df['jaccard_score'].mean()))
    print('std jaccard score: {:2f}'.format(result_df['jaccard_score'].std()))
    print('max jaccard score: {:2f}'.format(result_df['jaccard_score'].max()))
    print('min jaccard score: {:2f}'.format(result_df['jaccard_score'].min()))
