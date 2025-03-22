# Hotel Booking Analytics API


A scalable API for hotel booking analytics, integrating a large language model (LLM) with dataset-driven insights. Users can ask flexible, natural-language questions about hotel bookings and retrieve precomputed analytics with visualizations.

## Overview

This project combines **FastAPI**, a **GPT-Neo-2.7B LLM**, and **ChromaDB** vector storage to deliver:
- Dynamic question answering using retrieval-augmented generation (RAG) and direct DataFrame queries.
- Precomputed analytics (e.g., revenue trends) with embedded Matplotlib plots.
- Persistent logging of responses in JSON format.

Built for flexibility and scalability, it processes the `hotel_bookings.csv` dataset to provide insights into rates, cancellations, hotel types, and more.

## Features

- **Dynamic Queries:** Ask any question (e.g., "busiest month," "total revenue for August 2018") via the `/ask` endpoint.
- **Vector Store:** ChromaDB with `all-MiniLM-L6-v2` embeddings for efficient context retrieval.
- **Logging:** Responses and response times saved to `responses.json`.



## Usage

1. **Clone the Repository:**
   git clone https://github.com/BBhushan1/ML
   cd hotel-booking-analytics

2. **Run Api**
   uvicorn main:app --reload

3. **Ask Question**
   curl -X POST "http://127.0.0.1:8000/ask" -H "Content-Type: application/json" -d '{"question": "What is the average price of a hotel booking in July 2017?"}'

4. **Get Analytics**
   curl "http://127.0.0.1:8000/analytics"




## Challenges

LLM Resource Demands: GPT-Neo-2.7B requires ~5-6 GB VRAM; mitigated with 4-bit quantization (~2-3 GB).
Response Time: Inference takes 1-2 seconds; optimized with max_new_tokens=150.
Vector Store Precision: Summaries may lack granular stats; consider richer embeddings.
Concurrency: JSON logging risks race conditions; future use of locks or a database recommended.




## Future Improvements
Scalability: Dockerize with a PostgreSQL database for response storage.
Performance: Cache frequent queries with Redis.
Accuracy: Enhance VectorStore with detailed summaries (e.g., cancellation trends).
Usability: Add a web UI for interactive querying.




## Acknowledgments
FastAPI for the framework.
Hugging Face for GPT-Neo and SentenceTransformer models.
ChromaDB for vector storage.
