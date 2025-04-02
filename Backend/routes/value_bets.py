from fastapi import APIRouter, HTTPException, Query
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import logging

# Import components from the original application
from api.odds_comparison import OddsComparison
from models.prediction.prediction_model import PredictionModel
from api.api_client import APIClient

# Setup logging
logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/api/value-bets", tags=["value-bets"])

# Initialize components
api_client = APIClient()
odds_comparison = OddsComparison(api_client)
prediction_model = PredictionModel()

@router.get("/")
async def get_value_bets(
    days_ahead: int = Query(7, description="Number of days ahead to fetch value bets"),
    min_edge: float = Query(0.1, description="Minimum edge percentage (0.1 = 10%)")
):
    """
    Get value betting opportunities
    """
    try:
        # Get date range
        today = datetime.now().date()
        end_date = today + timedelta(days=days_ahead)
        
        # Get value bets from the odds comparison module
        value_bets = odds_comparison.get_all_value_bets(
            start_date=today.isoformat(),
            end_date=end_date.isoformat(),
            min_edge=min_edge
        )
        
        return {"value_bets": value_bets}
    
    except Exception as e:
        logger.error(f"Error fetching value bets: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error fetching value bets: {str(e)}")

@router.get("/match/{match_id}")
async def get_match_value_bets(match_id: int):
    """
    Get value betting opportunities for a specific match
    """
    try:
        # Get match prediction
        match_prediction = prediction_model.get_match_prediction(match_id)
        if not match_prediction:
            raise HTTPException(status_code=404, detail=f"Prediction for match ID {match_id} not found")
        
        # Get value bets for this match
        value_bets = odds_comparison.find_value_bets(
            match_id=match_id,
            prediction=match_prediction
        )
        
        return {"match_id": match_id, "value_bets": value_bets}
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching match value bets: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error fetching match value bets: {str(e)}")

@router.get("/performance")
async def get_value_bet_performance(
    days_back: int = Query(30, description="Number of days back to analyze performance")
):
    """
    Get performance metrics for past value bets
    """
    try:
        # Calculate date range
        today = datetime.now().date()
        start_date = today - timedelta(days=days_back)
        
        # Get performance metrics
        performance = odds_comparison.analyze_value_bet_performance(
            start_date=start_date.isoformat(),
            end_date=today.isoformat()
        )
        
        return performance
    
    except Exception as e:
        logger.error(f"Error fetching value bet performance: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error fetching value bet performance: {str(e)}")
