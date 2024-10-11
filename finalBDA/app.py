from flask import Flask, render_template, request, jsonify
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

app = Flask(__name__)

# Global variables to store the DataFrame and TF-IDF matrix
df = None
tfidf_matrix = None

# Load the dataset and prepare TF-IDF matrix
def load_dataset():
    global df, tfidf_matrix
    df = pd.read_csv('spotify_dataset.csv')
    
    # Combine relevant features into a single string for each song
    df['content'] = df['genre'] + ' ' + df['artist'] + ' ' + df['track_name']
    
    # Create TF-IDF vectorizer
    tfidf = TfidfVectorizer(stop_words='english')
    
    # Generate TF-IDF matrix
    tfidf_matrix = tfidf.fit_transform(df['content'])

# Load the dataset when the app starts
load_dataset()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    input_song = request.form['song']
    input_artist = request.form['artist']
    
    if df is None or tfidf_matrix is None:
        return jsonify({"error": "Dataset not loaded"})
    
    # Find the index of the input song
    input_idx = df[(df['track_name'].str.contains(input_song, case=False, na=False)) &
                   (df['artist'].str.contains(input_artist, case=False, na=False))].index
    
    if len(input_idx) == 0:
        return jsonify({"error": "Song not found. Try a different input."})
    
    # Get the TF-IDF vector for the input song
    input_vec = tfidf_matrix[input_idx[0]]
    
    # Calculate cosine similarity between input and all songs
    cosine_similarities = cosine_similarity(input_vec, tfidf_matrix).flatten()
    
    # Get indices of top 5 similar songs (excluding the input song)
    similar_indices = cosine_similarities.argsort()[:-6:-1]
    similar_indices = similar_indices[similar_indices != input_idx[0]]
    
    # Get the top 5 recommendations
    recommendations = df.iloc[similar_indices]
    
    results = []
    for _, row in recommendations.iterrows():
        results.append({
            "artist": row['artist'],
            "track_name": row['track_name'],
            "genre": row['genre']
        })
    
    return jsonify(results)

if __name__ == '__main__':
    app.run(debug=True)