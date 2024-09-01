import warnings
warnings.filterwarnings('ignore')
import pandas as pd
from sentence_transformers import SentenceTransformer
from bertopic import BERTopic
from umap import UMAP
from utils.system import get_data

# Load the CSV file into a DataFrame
full_file_path = get_data() / "loneliness" / "OurLabeledData" / "human_annotations_all_8000_overall.csv"
df = pd.read_csv(full_file_path)

# Extract the relevant text column into a list of documents
docs = df['narrative'].tolist()

topic_model = BERTopic()
topics, probs = topic_model.fit_transform(docs)

#search for topics that are similar to an input search_term "loneliness"
similar_topics, similarity = topic_model.find_topics("loneliness", top_n=20)
print(topic_model.get_topic(similar_topics[0]))

print("***********************************************************************************")

similar_topics, similarity = topic_model.find_topics("lonely", top_n=20)
print(topic_model.get_topic(similar_topics[0]))

#Intertopic Distance Map
distance_map = topic_model.visualize_topics()
distance_map.show()

#Visualize Topic Similarity
heatmap = topic_model.visualize_heatmap()
heatmap.show()

#topic distributions for documents:
topic_distr, _ = topic_model.approximate_distribution(docs)
distr_map = topic_model.visualize_distribution(topic_distr[1])
distr_map.show()

# Calculate the topic distributions on a token-level
#topic_distr, topic_token_distr = topic_model.approximate_distribution(docs, use_embedding_model=True)
# Visualize the token-level distributions
#distr = topic_model.visualize_approximate_distribution(docs[1], topic_token_distr[1])
#distr.show()


#hierchal topics
sentence_model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = sentence_model.encode(docs, show_progress_bar=False)

# Train BERTopic with default settings or with UMAP reduction
topic_model = BERTopic().fit(docs, embeddings)

# Extract hierarchical topics
hierarchical_topics = topic_model.hierarchical_topics(docs)
#print(hierarchical_topics)
print("Number of hierarchical levels:", len(hierarchical_topics))
# Run the visualization with the original embeddings
topic_model.visualize_hierarchical_documents(docs, hierarchical_topics, embeddings=embeddings, nr_levels=len(hierarchical_topics)-1)
print("Success - 1")
# Reduce dimensionality of embeddings, this step is optional but much faster to perform iteratively:
reduced_embeddings = UMAP(n_neighbors=10, n_components=2, min_dist=0.0, metric='cosine').fit_transform(embeddings)
print("Success - 2")
hierarchical_map = topic_model.visualize_hierarchical_documents(docs, hierarchical_topics, reduced_embeddings=reduced_embeddings, nr_levels=len(hierarchical_topics)-1)
hierarchical_map.show()