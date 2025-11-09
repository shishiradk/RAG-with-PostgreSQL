from datetime import datetime
import uuid

import pandas as pd
from database.vector_store import VectorStore

# Initialize VectorStore
vec = VectorStore()

# Read the CSV file
df = pd.read_csv("../data/faq_dataset.csv", sep=";")


# Prepare data for insertion
def prepare_record(row):
    """Prepare a record for insertion into the vector store.
    
    This function creates a record with a UUID as the ID.
    """
    content = f"Question: {row['question']}\nAnswer: {row['answer']}"
    embedding = vec.get_embedding(content)
    return pd.Series(
        {
            "id": str(uuid.uuid4()),  # Use standard UUID instead of timescale_vector's uuid_from_time
            "metadata": {
                "category": row["category"],
                "created_at": datetime.now().isoformat(),
            },
            "content": content,  # Changed from "contents" to "content" to match your VectorStore
            "embedding": embedding,
        }
    )


records_df = df.apply(prepare_record, axis=1)

# Create tables and insert data
vec.create_tables()
vec.create_index()  # HNSW index with pgvector
vec.upsert(records_df)

print("Data ingestion completed successfully")
print(f"Inserted {len(records_df)} records into the vector store")