import pandas as pd
import numpy as np
from pymilvus import MilvusClient
from sentence_transformers import SentenceTransformer
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')

# Load data
try:
    videos = pd.read_csv('videos.csv')
    user_interactions = pd.read_csv('user_interactions.csv')
except FileNotFoundError:
    logging.error("Error: 'videos.csv' or 'user_interactions.csv' file not found.")
    exit(1)

embed_model = SentenceTransformer('intfloat/e5-base-v2')
video_embeddings = embed_model.encode(videos['title'] + ' : ' + videos['description'])

# Initialize Milvus vector search
client = MilvusClient("milvus_demo.db")
collection_name = 'video_embeddings'

if not client.has_collection(collection_name=collection_name):
    logging.info("Creating Milvus collection: 'video_embeddings'")
    client.create_collection(
        collection_name=collection_name,
        dimension=768,
    )
    data = [{"id": video['video_id'], "vector": video_embeddings[i], "video_id": video['video_id']} for i, video in enumerate(videos.to_dict('records'))]
    res = client.upsert(collection_name=collection_name, data=data)
    logging.info(f"Inserted {len(res)} entities into the collection.")
else:
    logging.info("Milvus collection 'video_embeddings' already exists.")

def get_recommendations(user_id, top_n=10):
    try:
        # Content-based filtering
        user_interactions_sorted = user_interactions[user_interactions['user_id'] == user_id].sort_values(by='interaction_type', ascending=False).head(2)
        user_watched_videos = videos.loc[videos['video_id'].isin(user_interactions_sorted['video_id'].tolist())]
        user_watched_videos = pd.merge(user_interactions_sorted[['video_id', 'interaction_type']], user_watched_videos, on='video_id', how='left')

        video_content = user_watched_videos['title'] + ' : ' + user_watched_videos['description']
        user_watched_embeddings = embed_model.encode(video_content.to_list())

        search_results = client.search(
            collection_name=collection_name,
            data=user_watched_embeddings,
            limit=top_n,
            output_fields=["id"],
        )
        video_ids = [result["id"] for result in search_results[0]]
        video_count = 0
        recommendations = []
        for id in video_ids:
            if id in videos['video_id'].values:
                recommendations.append(id)
                video_count += 1
            if video_count > top_n:
                break
        return recommendations
    except Exception as e:
        logging.error(f"Error in get_recommendations: {e}")
        return []

# Example usage
user_id = 10
try:
    recommendations = get_recommendations(user_id, top_n=10)
    print(recommendations)
except Exception as e:
    logging.error(f"Error in example usage: {e}")