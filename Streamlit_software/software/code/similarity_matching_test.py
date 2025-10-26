import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import pairwise_distances


# Load the sample dataset with 50 questions
sample_file = './dataset/sample_qa_50.xlsx'
df = pd.read_excel(sample_file)


# Function to preprocess text (similar to your actual implementation)
def text_preprocessing(text):
    # Simple preprocessing for simulation
    return str(text).lower()


# Function to determine if a match would be found based on actual TF-IDF similarity
def simulate_real_matching(query, questions, threshold):
    """
    Uses actual TF-IDF and cosine similarity to determine if a match would be found
    This more accurately simulates your QA.py implementation
    """
    # Process questions
    processed_questions = [text_preprocessing(q) for q in questions]

    # TF-IDF Vectorization
    tfidf_vec = TfidfVectorizer(analyzer='word')
    X_tfidf = tfidf_vec.fit_transform(processed_questions).toarray()

    # Process query
    processed_query = text_preprocessing(query)
    input_tfidf = tfidf_vec.transform([processed_query]).toarray()

    # Calculate cosine similarity
    cos = 1 - pairwise_distances(X_tfidf, input_tfidf, metric='cosine')

    # Check if max similarity exceeds threshold
    return cos.max() >= threshold, cos.max()


# Run actual TF-IDF simulations for different thresholds
thresholds = np.arange(0.0, 1.05, 0.05)
results = []

# Get all questions for the similarity calculation
all_questions = df['Question'].tolist()

for thresh in thresholds:
    correct = 0
    total_similarity = 0

    for idx, row in df.iterrows():
        query = row['Question']
        # Temporarily remove the current question from the list to avoid perfect matching
        temp_questions = all_questions.copy()
        temp_questions.pop(idx)

        found_match, similarity = simulate_real_matching(query, temp_questions, thresh)
        if found_match:
            correct += 1
        total_similarity += similarity

    accuracy = correct / len(df)
    avg_similarity = total_similarity / len(df)

    results.append({
        "Threshold": round(thresh, 2),
        "Accuracy": round(accuracy, 3),
        "Avg_Similarity": round(avg_similarity, 3)
    })

# Create a DataFrame for results
results_df = pd.DataFrame(results)

print(results_df)

# Plot
plt.figure(figsize=(10, 6))
plt.plot(results_df['Threshold'], results_df['Accuracy'], marker='o')
plt.title("Chatbot Accuracy vs. Threshold")
plt.xlabel("Threshold")
plt.ylabel("Accuracy")
plt.grid(True)
plt.xticks(thresholds)
plt.ylim(0, 1)
plt.show()
# Display results to the user
#import ace_tools as tools




#tools.display_dataframe_to_user(name="Threshold Sensitivity Analysis", dataframe=results_df)