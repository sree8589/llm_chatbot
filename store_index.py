
from src.helper import load_pdf, text_split, download_hugging_face_embeddings
from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import ServerlessSpec
import pinecone
import os
from sentence_transformers import SentenceTransformer
PINECONE_API_KEY="60a9a852-6ad2-4041-9ddf-b4b2d2d06731"
PINECONE_API_ENV="us-east-1" 

extracted_data=load_pdf("data/")
text_chunks = text_split(extracted_data)
embeddings = download_hugging_face_embeddings()
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index('medical-bot')

vectors = []
for chunk in text_chunks:
    text = chunk.page_content  # Extract text from Document
    embedding = embedding_model.encode(text).tolist()  # Get embedding and convert to list
    vector = {
        "id": f"chunk-{chunk.metadata['page']}",  # Use page number or another unique identifier
        "values": embedding,
        "metadata": {"text": text}  # Include the text data in metadata
    }
    vectors.append(vector)
from uuid import uuid4
def generate_unique_id(base_id):
    return f"{base_id}-{uuid4().hex}"


def chunk_data(data, chunk_size):
    """Yield successive chunks of data of a specified size."""
    for i in range(0, len(data), chunk_size):
        yield data[i:i + chunk_size]

total_inserted=0
batch_count = 0
batch_size = 1000
for batch in chunk_data(vectors, batch_size):
    batch_count += 1
    try:
            # Prepare the items to be inserted
        vect = [{'id': generate_unique_id(item['id']), 'values': item['values'], 'metadata': item['metadata']} for item in batch]
            
            # Upsert the batch into Pinecone
        index.upsert(vectors=vect)
        total_inserted += len(batch)
        print(f"Inserted batch-{batch_count} of size {len(batch)}")

    except Exception as e:
            # Handle exceptions that occur during the upsert operation
        print(f"An error occurred while inserting batch-{batch_count}: {str(e)}")

print(f"Data insertion completed. Total inserted: {total_inserted}, Expected: {len(text_chunks)}")