# Task 6

import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.title("Movie Recommendation System")
st.write("Select a movie to get content-based recommendations!")

df = pd.read_csv("Trending Movies.csv").head(2500)
df['overview'] = df['overview'].fillna("")

vec = TfidfVectorizer(max_features=5000, stop_words='english')
matrix = vec.fit_transform(df['overview'])
sim_matrix = cosine_similarity(matrix)

selected_movie = st.selectbox("Choose a movie from the list:", df['title'].values)

if st.button("Generate Recommendations"):
    idx = df[df['title'] == selected_movie].index[0]
    scores = list(enumerate(sim_matrix[idx]))
    sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)
    
    st.write("### You should also watch:")
    for i in sorted_scores[1:6]:
        st.write("- " + df.iloc[i[0]]['title'])