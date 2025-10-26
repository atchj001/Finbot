import pandas as pd
import numpy as np 
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import pairwise_distances
import datetime as dt
from spell_check import text_preprocessing


def time_response(str):
    date = dt.datetime.now()
    if str == 'time':
        hour = date.strftime("%H")
        minute = date.strftime("%M") 
        second = date.strftime("%S")
        print("- Skynet: It's %s:%s:%s now. " %(hour,minute,second))



# path of small talk dataset
data_path = './dataset/small_talk.csv'

def talk_response(query, threshold):

    df = pd.read_csv(data_path)

    # TF-IDF
    tfidf_vec = TfidfVectorizer(analyzer='word')
    X_tfidf = tfidf_vec.fit_transform(df['Question']).toarray()
    df_tfidf = pd.DataFrame(X_tfidf, columns = tfidf_vec.get_feature_names_out())

    # process query 
    input_tfidf = tfidf_vec.transform([query.lower()]).toarray()

    # cosine similarity
    cos = 1 - pairwise_distances(df_tfidf, input_tfidf, metric = 'cosine')
    
    if cos.max() >= threshold:
        id_argmax = np.where(cos == np.max(cos, axis=0))
        id = np.random.choice(id_argmax[0]) 
        return df['Answer'].loc[id]
    else:
        return 'NOT FOUND'
    
if __name__ == "__main__":
    print(talk_response("hey how are you", 0.1))

