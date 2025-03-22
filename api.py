from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
import logging
from data_preprocessing import DataPreprocessor
from vector_store import VectorStore
from analytics import HotelAnalytics
from llm_interface import LLMInterface
import os
import json
from datetime import datetime
import asyncio

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI(title="Hotel Analytics RAG", version="1.0")

DATA_PATH = "data/hotel_bookings.csv"
PROCESSED_DATA_PATH = "processed_hotel_bookings.csv.gz"
RESPONSE_STORAGE_PATH = "response_history.json"

preprocessor = DataPreprocessor(DATA_PATH)
df = preprocessor.get_processed_data()

vector_store = VectorStore(df=df)
vector_store.load_data()  

analytics = HotelAnalytics(df=df)
llm = LLMInterface(vector_store=vector_store, df=df)

if not os.path.exists(RESPONSE_STORAGE_PATH):
    with open(RESPONSE_STORAGE_PATH, 'w') as f:
        json.dump([], f)

class AskRequest(BaseModel):
    question: str
    k: Optional[int] = 5

class AskResponse(BaseModel):
    question: str
    answer: str

class AnalyticsRequest(BaseModel):
    analysis_type: str  

class AnalyticsResponse(BaseModel):
    analysis_type: str
    results: Dict[str, Any]

class HealthResponse(BaseModel):
    status: str
    message: str

async def save_response_to_json(response_data: dict):
    try:
        with open(RESPONSE_STORAGE_PATH, 'r') as f:
            history = json.load(f)
        
        response_data['timestamp'] = datetime.now().isoformat()
        history.append(response_data)
        
        with open(RESPONSE_STORAGE_PATH, 'w') as f:
            json.dump(history, f, indent=2)
    except Exception as e:
        logger.error(f"Error saving response to JSON: {e}")

@app.post("/ask", response_model=AskResponse)
async def ask_question(request: AskRequest):
    try:
        logger.info(f"Received question: {request.question}")
        answer = llm.generate_response(request.question, k=request.k)
        response = AskResponse(question=request.question, answer=answer)
        
        response_dict = response.dict()
        response_dict['endpoint'] = 'ask'
        asyncio.create_task(save_response_to_json(response_dict))
        
        return response
    except Exception as e:
        logger.error(f"Error processing question '{request.question}': {e}")
        raise HTTPException(status_code=500, detail=f"Error processing question: {str(e)}")

@app.post("/analytics", response_model=AnalyticsResponse)
async def get_analytics(request: AnalyticsRequest):
    try:
        logger.info(f"Received analytics request: {request.analysis_type}")
        analysis_type = request.analysis_type.lower()
        if analysis_type == "revenue_trends":
            results = analytics.revenue_trends()
        elif analysis_type == "cancellation_analysis":
            results = analytics.cancellation_analysis()
        elif analysis_type == "geographical_distribution":
            results = analytics.geographical_distribution()
        elif analysis_type == "lead_time_analysis":
            results = analytics.lead_time_analysis()
        elif analysis_type == "all":
            results = analytics.generate_all_analytics()
        else:
            raise HTTPException(status_code=400, detail="Invalid analysis type")
        
        response = AnalyticsResponse(analysis_type=analysis_type, results=results)
        
        response_dict = response.dict()
        response_dict['endpoint'] = 'analytics'
        asyncio.create_task(save_response_to_json(response_dict))
        
        return response
    except Exception as e:
        logger.error(f"Error generating analytics '{request.analysis_type}': {e}")
        raise HTTPException(status_code=500, detail=f"Error generating analytics: {str(e)}")

@app.get("/health", response_model=HealthResponse)
async def health_check():
    try:
        if df is not None and vector_store.is_data_loaded:
            return HealthResponse(status="healthy", message="API is running and data is loaded")
        else:
            return HealthResponse(status="unhealthy", message="Data not loaded properly")
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8080)