from pymilvus import MilvusClient
from sentence_transformers import SentenceTransformer


client = MilvusClient("milvus_demo.db")
model = SentenceTransformer('intfloat/e5-base-v2')


if not client.has_collection(collection_name="demo_collection"):
    client.create_collection(
        collection_name="demo_collection",
        dimension=768,      )
    
    docs = [
        "Artificial intelligence was founded as an academic discipline in 1956.",
        "Alan Turing was the first person to conduct substantial research in AI.",
        "Born in Maida Vale, London, Turing was raised in southern England.",
    ]

    embedding = model.encode(docs)
    data = [{"id": i, "vector": embedding[i], "text": docs[i], "subject": "history"}  for i in range(len(embedding))]

    res = client.insert(collection_name="demo_collection", data=data)
    
    print(res)


query_vectors = model.encode(["when Artificial intelligence founded?"])

res = client.search(
    collection_name="demo_collection",  # target collection
    data=query_vectors,  # query vectors
    limit=2,  # number of returned entities
    output_fields=["text", "subject"],  # specifies fields to be returned
)

print(res)