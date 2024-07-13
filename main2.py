import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# Load the user-video interaction data
interactions = pd.read_csv('user_interactions.csv')

# Create a user-video interaction matrix
user_video_matrix = interactions.pivot_table(index='user_id', columns='video_id', values='interaction_type', fill_value=0)

# Calculate item-item similarity matrix using cosine similarity
item_similarity_matrix = 1 - user_video_matrix.T.corr(method='pearson')
item_similarity_matrix = pd.DataFrame(cosine_similarity(user_video_matrix.T), index=user_video_matrix.columns, columns=user_video_matrix.columns)

# Define a function to get the top N recommendations for a given user
def get_top_recommendations(user_id, user_video_matrix, item_similarity_matrix, N=10):
    # Check if the user has any interactions
    if user_id not in user_video_matrix.index:
        return []

    # Get the user's interacted videos
    user_interactions = user_video_matrix.loc[user_id]
    
    # Calculate a weighted average of similarities for each video
    video_similarities = item_similarity_matrix.mul(user_interactions, axis=1)
    video_similarities = video_similarities.sum(axis=0) / user_interactions.sum()
    
    # Sort the videos by similarity and get the top N recommendations
    recommendations = video_similarities.sort_values(ascending=False)
    return recommendations.head(N).index.tolist()

# Example usage
user_id = 10
top_recommendations = get_top_recommendations(user_id, user_video_matrix, item_similarity_matrix)
print(f"Top recommendations for user {user_id}: {top_recommendations}")