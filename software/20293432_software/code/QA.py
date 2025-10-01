import pandas as pd
import numpy as np
import sklearn
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import pairwise_distances
from spell_check import text_preprocessing

# Dataset path
data_path = './dataset/question_answering.xlsx'  # Ensure this is correct

# Global variables to track last matched question and answer
last_matched_question = ""
last_matched_answer = ""


def answer_Q(query, threshold, user_financial_literacy):
    """
    Retrieve an answer from the dataset based on the user's query and financial literacy level.
    Also tracks the matched question and answer for feedback purposes.

    :param query: The user's question.
    :param threshold: Similarity threshold for finding a matching question.
    :param user_financial_literacy: The literacy level ("Beginner", "Intermediate", "Advanced").
    :return: The most relevant answer based on the user's literacy level.
    """
    global last_matched_question, last_matched_answer

    try:
        # Handle literacy level for file selection and column mapping
        valid_literacy_levels = ["Beginner", "Intermediate", "Advanced"]
        literacy_level = user_financial_literacy if user_financial_literacy in valid_literacy_levels else "Beginner"

        # Select the appropriate dataset based on literacy level
        actual_data_path = data_path
        if literacy_level == "Advanced" and user_financial_literacy == "Advanced":
            # If you have a separate file for advanced users, use it
            advanced_path = './dataset/question_answering_advanced.xlsx'
            try:
                # Check if the advanced file exists
                open(advanced_path)
                actual_data_path = advanced_path
            except:
                # If it doesn't exist, use the default path
                actual_data_path = data_path

        df = pd.read_excel(actual_data_path, engine='openpyxl')

        # Create a copy of the DataFrame to avoid the chained assignment warning
        df = df.copy()

        # Fix NaN values properly
        df['Question'] = df['Question'].fillna("")

        # Ensure the answer columns also have no NaN values
        for col in ['Beginner', 'Intermediate', 'Advanced']:
            if col in df.columns:
                df[col] = df[col].fillna("No answer available.")

        # Apply text preprocessing
        df['processed_Q'] = df['Question'].apply(lambda x: text_preprocessing(str(x), 'stemming'))

        # TF-IDF Vectorization
        tfidf_vec = TfidfVectorizer(analyzer='word')
        X_tfidf = tfidf_vec.fit_transform(df['processed_Q']).toarray()
        df_tfidf = pd.DataFrame(X_tfidf, columns=tfidf_vec.get_feature_names_out())

        # Process query
        processed_query = text_preprocessing(query, 'stemming')
        input_tfidf = tfidf_vec.transform([processed_query]).toarray()
        cos = 1 - pairwise_distances(df_tfidf, input_tfidf, metric='cosine')

        if cos.max() >= threshold:
            id_argmax = np.where(cos == np.max(cos, axis=0))
            id = np.random.choice(id_argmax[0])

            # Store the matched question for feedback purposes
            last_matched_question = df['Question'].iloc[id]

            # Select the answer column dynamically based on literacy level
            if literacy_level in df.columns:
                answer = df[literacy_level].iloc[id]
                last_matched_answer = answer
                return answer
            else:
                # Default to Beginner if the specific level isn't available
                default_answer = df['Beginner'].iloc[id] if 'Beginner' in df.columns else "No answer available."
                last_matched_answer = default_answer
                return default_answer
        else:
            # Clear the tracked question and answer if no match found
            last_matched_question = ""
            last_matched_answer = ""
            return "NOT FOUND"

    except Exception as e:
        # Clear the tracked question and answer in case of error
        last_matched_question = ""
        last_matched_answer = ""
        return f"Error processing question: {str(e)}"