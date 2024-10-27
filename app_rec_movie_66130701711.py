
import streamlit as st
import pickle
import pandas as pd
from surprise import SVD, Dataset, Reader

# Load the data for user similarity-based recommendations
with open('66130701711_recommendation_movie_svd.pkl', 'rb') as file:
    user_similarity_df, user_movie_ratings = pickle.load(file)

# Load the data for SVD-based recommendations
with open('66130701711_recommendation_movie_svd.pkl', 'rb') as file:
    svd_model, movie_ratings, movies = pickle.load(file)

# Streamlit app title
st.title("Movie Recommendation System")

# User input to select recommendation type and user ID
st.subheader("Choose a Recommendation Method")
method = st.radio("Recommendation Method:", ("User Similarity", "SVD"))

user_id = st.number_input("Enter User ID:", min_value=1, step=1)

if st.button("Get Recommendations"):
    if method == "User Similarity":
        # Get recommendations using user similarity
        recommendations = get_movie_recommendations(user_id, user_similarity_df, user_movie_ratings, 10)
        st.write(f"Top 10 movie recommendations for User {user_id} based on user similarity:")
        for movie_title in recommendations:
            st.write(f"• {movie_title}")

    elif method == "SVD":
        # Get recommendations using SVD model
        rated_user_movies = movie_ratings[movie_ratings['userId'] == user_id]['movieId'].values
        unrated_movies = movies[~movies['movieId'].isin(rated_user_movies)]['movieId']
        
        # Predict ratings for unrated movies
        pred_rating = [svd_model.predict(user_id, movie_id) for movie_id in unrated_movies]
        # Sort predictions by estimated rating in descending order
        sorted_predictions = sorted(pred_rating, key=lambda x: x.est, reverse=True)
        
        # Get top 10 movie recommendations
        top_recommendations = sorted_predictions[:10]
        
        st.write(f"Top 10 movie recommendations for User {user_id} based on SVD:")
        for recommendation in top_recommendations:
            movie_title = movies[movies['movieId'] == recommendation.iid]['title'].values[0]
            st.write(f"{movie_title} (Estimated Rating: {recommendation.est:.2f})")
