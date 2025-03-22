import pandas as pd
import numpy as np
from pathlib import Path
import chromadb
from chromadb.utils import embedding_functions
import logging
from typing import Optional, List, Dict, Any, Tuple
from data_preprocessing import DataPreprocessor

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class VectorStore:
    
    def __init__(
        self,
        df: Optional[pd.DataFrame] = None,
        data_path: Optional[str] = None,
        collection_name: str = "hotel_bookings",
        persist_directory: str = "./chroma_db",
        embedding_model: str = "all-MiniLM-L6-v2"
    ):
        if df is not None:
            self.df = df
        elif data_path is not None:
            self.df = pd.read_csv(data_path, parse_dates=['arrival_date', 'reservation_status_date'])
            logger.info(f"Data loaded from {data_path}. Shape: {self.df.shape}")
        else:
            raise ValueError("Either df or data_path must be provided")
        
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(exist_ok=True)
        
        self.client = chromadb.PersistentClient(path=str(self.persist_directory))
        self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=embedding_model)
        
        self.collection = self.client.get_or_create_collection(
            name=collection_name, embedding_function=self.embedding_function)
        self.is_data_loaded = self.collection.count() > 0
        logger.info(f"Collection '{collection_name}' initialized with {self.collection.count()} documents.")
        
    def _prepare_document_chunks(self) -> Tuple[List[str], List[Dict[str, Any]], List[str]]:
        documents: List[str] = []
        metadatas: List[Dict[str, Any]] = []
        ids: List[str] = []
        
        yearly_summaries = self.df.groupby('arrival_date_year').agg({
            'booking_id': 'count', 'is_canceled': 'mean', 'adr': 'mean', 'total_revenue': 'sum'
        }).reset_index()
        for _, row in yearly_summaries.iterrows():
            year = int(row['arrival_date_year'])
            doc = (f"Hotel bookings summary for year {year}. Total bookings: {int(row['booking_id'])}. "
                   f"Cancellation rate: {row['is_canceled'] * 100:.2f}%. Average daily rate: {row['adr']:.2f}. "
                   f"Total revenue: {row['total_revenue']:.2f}.")
            documents.append(doc)
            metadatas.append({
                "type": "year_summary", "year": year, "total_bookings": int(row['booking_id']),
                "cancellation_rate": float(row['is_canceled'] * 100), "avg_adr": float(row['adr']),
                "total_revenue": float(row['total_revenue'])})
            ids.append(f"year_{year}")

        monthly_summaries = self.df.groupby('year_month').agg({
            'booking_id': 'count', 'is_canceled': 'mean', 'adr': 'mean', 'total_revenue': 'sum'
        }).reset_index()
        for _, row in monthly_summaries.iterrows():
            year_month = row['year_month']
            doc = (f"Hotel bookings summary for {year_month}. Total bookings: {int(row['booking_id'])}. "
                   f"Cancellation rate: {row['is_canceled'] * 100:.2f}%. Average daily rate: {row['adr']:.2f}. "
                   f"Total revenue: {row['total_revenue']:.2f}.")
            documents.append(doc)
            metadatas.append({
                "type": "month_summary", "year_month": year_month, "total_bookings": int(row['booking_id']),
                "cancellation_rate": float(row['is_canceled'] * 100), "avg_adr": float(row['adr']),
                "total_revenue": float(row['total_revenue'])})
            ids.append(f"month_{year_month}")

        for chunk in np.array_split(self.df, max(1, len(self.df) // 1000)):
            for _, row in chunk.iterrows():
                doc = (f"Booking ID {row['booking_id']}: Hotel: {row['hotel']}, Country: {row['country']}, "
                       f"Arrival Date: {row['arrival_date'].strftime('%Y-%m-%d')}, "
                       f"Canceled: {'Yes' if row['is_canceled'] else 'No'}, Total Guests: {row['total_guests']}, "
                       f"Total Stays: {row['total_stays']}, ADR: {row['adr']:.2f}, "
                       f"Total Revenue: {row['total_revenue']:.2f}.")
                documents.append(doc)
                metadatas.append({
                    "type": "booking_detail", "booking_id": int(row['booking_id']), "hotel": row['hotel'],
                    "country": row['country'], "arrival_date": row['arrival_date'].strftime('%Y-%m-%d'),
                    "is_canceled": bool(row['is_canceled']), "total_guests": int(row['total_guests']),
                    "total_stays": int(row['total_stays']), "adr": float(row['adr']),
                    "total_revenue": float(row['total_revenue'])})
                ids.append(f"booking_{row['booking_id']}")
        
        return documents, metadatas, ids

    def load_data(self, batch_size: int = 5000) -> None:
        if not self.is_data_loaded:
            documents, metadatas, ids = self._prepare_document_chunks()
            total_docs = len(documents)
            logger.info(f"Total documents to load: {total_docs}")
            
            for i in range(0, total_docs, batch_size):
                batch_docs = documents[i:i + batch_size]
                batch_metas = metadatas[i:i + batch_size]
                batch_ids = ids[i:i + batch_size]
                self.collection.add(documents=batch_docs, metadatas=batch_metas, ids=batch_ids)
                logger.info(f"Loaded batch {i // batch_size + 1}: {len(batch_docs)} documents")
            
            self.is_data_loaded = True
            logger.info(f"Successfully loaded {total_docs} documents into the vector store.")
        else:
            logger.info("Data already loaded into the vector store.")

    def query(self, question: str, k: int = 5) -> List[Dict[str, Any]]:
        if not self.is_data_loaded:
            self.load_data()
        
        results = self.collection.query(query_texts=[question], n_results=k,
                                       include=['documents', 'metadatas', 'distances'])
        relevant_data = [
            {"id": results['ids'][0][i], "document": results['documents'][0][i],
             "metadata": results['metadatas'][0][i], "distance": float(results['distances'][0][i])}
            for i in range(len(results['ids'][0]))
        ]
        logger.info(f"Query '{question}' returned {len(relevant_data)} results.")
        return relevant_data

if __name__ == "__main__":
    preprocessor = DataPreprocessor("data/hotel_bookings.csv")
    df = preprocessor.get_processed_data()
    vector_store = VectorStore(df=df)
    vector_store.load_data()