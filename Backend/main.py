from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import logging

# Import routes
from routes.matches import router as matches_router
from routes.value_bets import router as value_bets_router
from routes.odds import router as odds_router
from routes.leagues import router as leagues_router

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Soccer Prediction API", 
    description="Backend API for the Soccer Match Prediction Application",
    version="1.0.0"
)

# Add CORS middleware to allow requests from the Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Root endpoint
@app.get("/")
async def root():
    return {"message": "Soccer Prediction API is running"}

# Health check endpoint
@app.get("/health")
async def health_check():
    from datetime import datetime
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

# Include all routers
app.include_router(matches_router)
app.include_router(value_bets_router)
app.include_router(odds_router)
app.include_router(leagues_router)

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
