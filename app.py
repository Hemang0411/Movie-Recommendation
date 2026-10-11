import streamlit as st
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.title("Movie Recommendation System")
st.write("Find movies similar to your favorites!")

@st.cache_data
def load_and_prep_data():
    df = pd.read_csv("Trending Movies.csv")
    df = df.head(2500).reset_index(drop=True)
    df["clean_text"] = df["overview"].fillna("").str.lower().apply(lambda x: re.sub(r'[^\w\s]', '', x))
    
    vectorizer = TfidfVectorizer(max_features=5000, stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(df["clean_text"])
    sim_matrix = cosine_similarity(tfidf_matrix)
    
    return df, sim_matrix

df, similarity_matrix = load_and_prep_data()

movie_list = df["title"].tolist()
selected_movie = st.selectbox("Select a movie from the dropdown", movie_list)

if st.button("Generate Recommendations"):
    idx = df[df["title"] == selected_movie].index[0]
    scores = list(enumerate(similarity_matrix[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)
    top_scores = scores[1:6]
    
    movie_indices = [i[0] for i in top_scores]
    recommendations = df["title"].iloc[movie_indices].tolist()
    
    st.write("### Top 5 Recommendations:")
    for i, movie in enumerate(recommendations, 1):
        st.write(f"{i}. {movie}")