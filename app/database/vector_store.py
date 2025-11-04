import logging
import time
from typing import List, Optional, Tuple, Union
from datetime import datetime

import pandas as pd
from config.settings import get_settings
from openai import OpenAI
from timescale.client import Client 

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
