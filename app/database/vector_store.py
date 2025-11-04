import logging
import time
from typing import List, Optional, Tuple, Union, Any
from datetime import datetime

import pandas as pd
from config.settings import get_settings
from openai import OpenAI
from timescale.client import Client
from timescale import client


class VectorStore:
    "A class for managing vector operations and database interactions."
    
    def __init__(self):
        """Initialize the vectorstore with settings, OpenAI client and timescale vector client"""
        self.settings = get_settings()
        self.openai_client = OpenAI(api_key=self.settings.openai.api_key)
        self.embedding_model = self.settings.openai.embedding_model
        self.vector_settings = self.settings.vector_store

        self.vec_client = Client.Sync(
            self.settings.database.service_url,
            self.vector_settings.table_name,
            self.vector_settings.embedding_dimensions,
            time_partition_interval=self.vector_settings.time_partition_interval,
        )
        
    def get_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for the given text.
        """
        
        text = text.replace("\n", " ")
        start_time = time.time()

        embedding = (
            self.openai_client.embeddings.create(
                input=text,
                model=self.embedding_model,
            )
            .data[0]
            .embedding
        )

        elapsed_time = time.time() - start_time
        logging.info(f"Generated embedding in {elapsed_time:.3f} seconds.")
        return embedding

    def create_table(self) -> None:
        """Create the necessary tables in the database"""
        self.vec_client.create_table()
        
    def create_index(self) -> None:
        """Create the StreamingDiskANN index to speed up similarity search"""
        self.vec_client.create_embedding_index(client.DiskAnnIndex())
        
    def drop_index(self) -> None:
        """Drop the StreamingDiskANN index in the database"""
        self.vec_client.drop_embedding_index()
        
    def upsert(self, df: pd.DataFrame) -> None:
        """
        Insert or update records in the database from a pandas DataFrame.
        """
        records = df.to_records(index=False)
        self.vec_client.upsert(list(records))
        logging.info(
            f"Inserted {len(df)} records into {self.vector_settings.table_name}"
        )  
        
    def search(
        self,
        query_text: str,
        limit: int = 5,
        metadata_filter: Union[dict, None] = None,
        predicates: Optional[client.Predicates] = None,
        time_range: Optional[Tuple[datetime, datetime]] = None,
        return_dataframe: bool = True,
    ) -> Union[List[Tuple[Any, ...]], pd.DataFrame]:
        """
        Query the vector database for similar embeddings based on input text.
        """
        query_embedding = self.get_embedding(query_text)
        
        start_time = time.time()
        
        search_args = {"limit": limit}
        
        if metadata_filter:
            search_args["filter"] = metadata_filter
            
        if predicates:
            search_args["predicates"] = predicates
            
        if time_range:
            start_date, end_date = time_range
            search_args["uuid_time_range"] = client.UUIDTimeRange(start_date, end_date)
            
        results = self.vec_client.search(query_embedding, **search_args)
        elapsed_time = time.time() - start_time
        
        logging.info(f"Search completed in {elapsed_time:.3f} seconds.")
        
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
        """
        df = pd.DataFrame(results, columns=["id", "metadata", "contents", "embedding", "distance"])
        
        df = pd.concat(
            [df.drop(["metadata"], axis=1), df["metadata"].apply(pd.Series)],
            axis=1,
        )
        
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

        Examples:
            Delete by IDs:
                vector_store.delete(ids=["8ab544ae-766a-11ef-81cb-decf757b836d"])

            Delete by metadata filter:
                vector_store.delete(metadata_filter={"category": "Shipping"})

            Delete all records:
                vector_store.delete(delete_all=True)
        """
        
        if sum(bool(x) for x in (ids, metadata_filter, delete_all)) != 1:
            raise  ValueError(
                "Provide exactly one of : ids, metadata_filter, or delete_all"
            )
            
        if delete_all:
            self.vec_client.delete_all()
            logging.info(f"Deleted all records from {self.vector_settings.table_name}")
        elif ids:
            self.vec_client.delete_by_ids(ids)
            logging.info(f"Deleted {len(ids)} records by IDs from {self.vector_settings.table_name}")
        
        elif metadata_filter:
            self.vec_client.delete_by_filter(metadata_filter)
            logging.info(f"Deleted records by metadata filter from {self.vector_settings.table_name}")