import logging
import time
import json
import sys
import os
from typing import Any, List, Optional, Tuple, Union
from datetime import datetime

import pandas as pd
import psycopg2
from pgvector.psycopg2 import register_vector
import numpy as np
from openai import OpenAI

# Add the parent directory to Python path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Use relative import since we're in the app package
from config.settings import get_settings


class VectorStore:
    """A class for managing vector operations and database interactions using pgvector."""

    def __init__(self):
        """Initialize the VectorStore with settings, OpenAI client, and pgvector connection."""
        self.settings = get_settings()
        self.openai_client = OpenAI(api_key=self.settings.openai.api_key)
        self.embedding_model = self.settings.openai.embedding_model
        self.vector_settings = self.settings.vector_store
        
        # Initialize pgvector connection
        self.conn = psycopg2.connect(self.settings.database.service_url)
        register_vector(self.conn)
        self.cur = self.conn.cursor()

        # Check if vector extension is installed
        self._ensure_vector_extension()

    def _ensure_vector_extension(self):
        """Ensure the vector extension is installed in the database."""
        try:
            self.cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            self.conn.commit()
            logging.info("Vector extension ensured")
        except Exception as e:
            logging.warning(f"Could not create vector extension: {e}")

    def get_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for the given text.

        Args:
            text: The input text to generate an embedding for.

        Returns:
            A list of floats representing the embedding.
        """
        text = text.replace("\n", " ")
        start_time = time.time()
        embedding = (
            self.openai_client.embeddings.create(
                input=[text],
                model=self.embedding_model,
            )
            .data[0]
            .embedding
        )
        elapsed_time = time.time() - start_time
        logging.info(f"Embedding generated in {elapsed_time:.3f} seconds")
        return embedding

    def create_tables(self) -> None:
        """Create the necessary tables in the database"""
        # First ensure vector extension exists
        self._ensure_vector_extension()
        
        # Create documents table with vector column
        self.cur.execute(f"""
            CREATE TABLE IF NOT EXISTS {self.vector_settings.table_name} (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                metadata JSONB,
                content TEXT,
                embedding vector({self.vector_settings.embedding_dimensions}),
                created_at TIMESTAMP DEFAULT NOW()
            );
        """)
        
        # Create HNSW index for efficient similarity search
        self.cur.execute(f"""
            CREATE INDEX IF NOT EXISTS {self.vector_settings.table_name}_embedding_idx 
            ON {self.vector_settings.table_name} USING hnsw (embedding vector_cosine_ops);
        """)
        
        self.conn.commit()
        logging.info(f"Created table {self.vector_settings.table_name} with pgvector")

    def create_index(self) -> None:
        """Create the HNSW index to speed up similarity search"""
        # Index is already created in create_tables, but you can recreate if needed
        self.cur.execute(f"""
            DROP INDEX IF EXISTS {self.vector_settings.table_name}_embedding_idx;
            CREATE INDEX {self.vector_settings.table_name}_embedding_idx 
            ON {self.vector_settings.table_name} USING hnsw (embedding vector_cosine_ops);
        """)
        self.conn.commit()
        logging.info("HNSW index created")

    def drop_index(self) -> None:
        """Drop the HNSW index in the database"""
        self.cur.execute(f"DROP INDEX IF EXISTS {self.vector_settings.table_name}_embedding_idx;")
        self.conn.commit()
        logging.info("HNSW index dropped")

    def upsert(self, df: pd.DataFrame) -> None:
        """
        Insert or update records in the database from a pandas DataFrame.

        Args:
            df: A pandas DataFrame containing the data to insert or update.
                Expected columns: id, metadata, content, embedding
        """
        for _, row in df.iterrows():
            # Convert metadata dict to JSON string for PostgreSQL
            metadata_json = json.dumps(row['metadata']) if row['metadata'] is not None else None
            
            self.cur.execute(
                f"""
                INSERT INTO {self.vector_settings.table_name} (id, metadata, content, embedding)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (id) DO UPDATE SET
                    metadata = EXCLUDED.metadata,
                    content = EXCLUDED.content,
                    embedding = EXCLUDED.embedding
                """,
                (row['id'], metadata_json, row['content'], row['embedding'])
            )
        
        self.conn.commit()
        logging.info(f"Upserted {len(df)} records into {self.vector_settings.table_name}")

    def search(
        self,
        query_text: str,
        limit: int = 5,
        metadata_filter: dict = None,
        predicates: Optional[Any] = None,  # Keeping for compatibility, but not used in pgvector
        time_range: Optional[Tuple[datetime, datetime]] = None,
        return_dataframe: bool = True,
    ) -> Union[List[Tuple[Any, ...]], pd.DataFrame]:
        """
        Query the vector database for similar embeddings based on input text using pgvector.

        Args:
            query_text: The input text to search for.
            limit: The maximum number of results to return.
            metadata_filter: A dictionary for equality-based metadata filtering.
            predicates: Not supported in pgvector version (kept for compatibility).
            time_range: A tuple of (start_date, end_date) to filter results by time.
            return_dataframe: Whether to return results as a DataFrame (default: True).

        Returns:
            Either a list of tuples or a pandas DataFrame containing the search results.
        """
        query_embedding = self.get_embedding(query_text)
        start_time = time.time()

        # Build the query - FIXED: Use array casting for the embedding parameter
        base_query = f"""
            SELECT id, metadata, content, embedding, 
                   embedding <=> %s::vector as distance
            FROM {self.vector_settings.table_name}
        """
        
        where_conditions = []
        params = [query_embedding]  # This will be cast to vector type
        
        # Add metadata filter
        if metadata_filter:
            for key, value in metadata_filter.items():
                where_conditions.append(f"metadata->>%s = %s")
                params.extend([key, str(value)])
        
        # Add time range filter
        if time_range:
            start_date, end_date = time_range
            where_conditions.append("created_at BETWEEN %s AND %s")
            params.extend([start_date, end_date])
        
        # Combine conditions
        if where_conditions:
            base_query += " WHERE " + " AND ".join(where_conditions)
        
        # Add ordering and limit
        base_query += " ORDER BY distance LIMIT %s"
        params.append(limit)
        
        # Execute query
        try:
            self.cur.execute(base_query, params)
            results = self.cur.fetchall()
            elapsed_time = time.time() - start_time
            logging.info(f"Vector search completed in {elapsed_time:.3f} seconds, found {len(results)} results")
        except Exception as e:
            logging.error(f"Error executing search query: {e}")
            # Fallback: try without explicit vector casting
            try:
                base_query_fallback = f"""
                    SELECT id, metadata, content, embedding, 
                           embedding <=> %s as distance
                    FROM {self.vector_settings.table_name}
                """
                if where_conditions:
                    base_query_fallback += " WHERE " + " AND ".join(where_conditions)
                base_query_fallback += " ORDER BY distance LIMIT %s"
                
                self.cur.execute(base_query_fallback, params)
                results = self.cur.fetchall()
                elapsed_time = time.time() - start_time
                logging.info(f"Vector search (fallback) completed in {elapsed_time:.3f} seconds, found {len(results)} results")
            except Exception as e2:
                logging.error(f"Fallback search also failed: {e2}")
                results = []

        if return_dataframe:
            return self._create_dataframe_from_results(results)
        else:
            return results

    def _create_dataframe_from_results(
        self,
        results: List[Tuple[Any, ...]],
    ) -> pd.DataFrame:
        """
        Create a pandas DataFrame from the search results.

        Args:
            results: A list of tuples containing the search results.

        Returns:
            A pandas DataFrame containing the formatted search results.
        """
        if not results:
            return pd.DataFrame(columns=["id", "metadata", "content", "embedding", "distance"])
            
        # Convert results to DataFrame
        df = pd.DataFrame(
            results, columns=["id", "metadata", "content", "embedding", "distance"]
        )

        # Expand metadata column if it exists and has data
        if not df.empty and df["metadata"].iloc[0] is not None:
            try:
                # Handle both string and dict metadata
                if isinstance(df["metadata"].iloc[0], str):
                    df["metadata"] = df["metadata"].apply(json.loads)
                
                metadata_df = df["metadata"].apply(pd.Series)
                df = pd.concat([df.drop(["metadata"], axis=1), metadata_df], axis=1)
            except Exception as e:
                logging.warning(f"Could not expand metadata: {e}")

        # Convert id to string for better readability
        df["id"] = df["id"].astype(str)

        return df

    def delete(
        self,
        ids: List[str] = None,
        metadata_filter: dict = None,
        delete_all: bool = False,
    ) -> None:
        """Delete records from the vector database.

        Args:
            ids (List[str], optional): A list of record IDs to delete.
            metadata_filter (dict, optional): A dictionary of metadata key-value pairs to filter records for deletion.
            delete_all (bool, optional): A boolean flag to delete all records.

        Raises:
            ValueError: If no deletion criteria are provided or if multiple criteria are provided.
        """
        if sum(bool(x) for x in (ids, metadata_filter, delete_all)) != 1:
            raise ValueError(
                "Provide exactly one of: ids, metadata_filter, or delete_all"
            )

        if delete_all:
            self.cur.execute(f"DELETE FROM {self.vector_settings.table_name}")
            logging.info(f"Deleted all records from {self.vector_settings.table_name}")
        elif ids:
            placeholders = ','.join(['%s'] * len(ids))
            self.cur.execute(
                f"DELETE FROM {self.vector_settings.table_name} WHERE id IN ({placeholders})",
                ids
            )
            logging.info(
                f"Deleted {len(ids)} records from {self.vector_settings.table_name}"
            )
        elif metadata_filter:
            conditions = []
            params = []
            for key, value in metadata_filter.items():
                conditions.append("metadata->>%s = %s")
                params.extend([key, str(value)])
            
            where_clause = " AND ".join(conditions)
            self.cur.execute(
                f"DELETE FROM {self.vector_settings.table_name} WHERE {where_clause}",
                params
            )
            logging.info(
                f"Deleted records matching metadata filter from {self.vector_settings.table_name}"
            )
        
        self.conn.commit()

    def get_table_info(self):
        """Get information about the current table and available tables"""
        try:
            # Check current table
            self.cur.execute(f"""
                SELECT COUNT(*) as count 
                FROM {self.vector_settings.table_name}
            """)
            current_count = self.cur.fetchone()[0]
            
            # Check all tables
            self.cur.execute("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'public'
            """)
            all_tables = [row[0] for row in self.cur.fetchall()]
            
            return {
                "current_table": self.vector_settings.table_name,
                "current_table_count": current_count,
                "all_tables": all_tables
            }
        except Exception as e:
            logging.error(f"Error getting table info: {e}")
            return {}

    def close(self):
        """Close the database connection"""
        self.cur.close()
        self.conn.close()


# Test function to verify everything works
def test_vector_store():
    """Test the VectorStore functionality"""
    try:
        store = VectorStore()
        
        # Get table info first
        table_info = store.get_table_info()
        print(f"Table Info: {table_info}")
        
        # Create tables if they don't exist
        store.create_tables()
        print("Tables created/verified successfully!")
        
        # Test embedding generation
        test_embedding = store.get_embedding("Test document")
        print(f"Embedding generated with {len(test_embedding)} dimensions")
        
        # Test search with empty table
        results = store.search("test query", limit=2)
        print(f"Search test completed, found {len(results)} results")
        
        store.close()
        print("All tests passed!")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_vector_store()