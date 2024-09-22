import streamlit as st
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer, util

# Load the model and dataset
@st.cache_resource
def load_model():
    return SentenceTransformer('paraphrase-MiniLM-L6-v2')

@st.cache_resource
def load_data():
    return pd.read_csv("lyrics.csv")

# Function to recommend songs
def recommend_songs_with_percentage(user_input, top_n=3):
    model = load_model()
    songs_df = load_data()

    lyrics_embeddings = model.encode(songs_df['Lyrics'].tolist(), convert_to_tensor=True)

    # Encode the user input into SBERT embedding
    user_input_embedding = model.encode(user_input, convert_to_tensor=True)

    # Compute cosine similarity between user input and all song lyrics embeddings
    cosine_similarities = util.pytorch_cos_sim(user_input_embedding, lyrics_embeddings).squeeze(0)

    # Normalize the cosine similarities to percentages
    match_percentages = cosine_similarities / cosine_similarities.max() * 100

    # Get the top N most similar songs
    top_n_indices = torch.topk(cosine_similarities, k=top_n).indices

    # Return the most similar songs along with their match percentages
    return songs_df.iloc[top_n_indices.cpu().numpy()][['Song Name', 'Album', 'Lyrics']], match_percentages[top_n_indices].cpu().numpy()

# Streamlit app
st.title('Taylor Swift Song Recommendation System')
st.write('Enter a sentence to get Taylor Swift song recommendations!')

# Get user input
user_input = st.text_input("Enter a sentence or describe a mood:")

# If user input is provided, recommend songs
if user_input:
    st.write(f"Songs matching your input '{user_input}':")

    # Get recommendations and match percentages
    recommended_songs, match_percentages = recommend_songs_with_percentage(user_input)

    # Display the top 3 recommendations
    for idx, (index, row) in enumerate(recommended_songs.iterrows()):
        st.subheader(f"Top {idx + 1}: {row['Song Name']} (Album: {row['Album']})")

        # Show the match percentage as a progress bar for the top match
        if idx == 0:
            st.write("Match Percentage:")
            st.progress(int(match_percentages[idx]))  # Display progress bar for the top song

        # Display match percentage as text for other matches
        st.write(f"Match Percentage: {match_percentages[idx]:.2f}%")

        # Optionally, show a snippet of the lyrics
        st.write(row['Lyrics'][:500] + "...")  # Show a preview of the lyrics
