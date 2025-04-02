from fastapi import APIRouter, HTTPException, Query
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import logging

# Import components from the original application
from api.api_client import APIClient
from api.data_fetcher import DataFetcher
from models.prediction.prediction_model import PredictionModel
from api.odds_comparison import OddsComparison

# Setup logging
logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/api/matches", tags=["matches"])

# Initialize components
api_client = APIClient()
data_fetcher = DataFetcher(api_client)
prediction_model = PredictionModel()
odds_comparison = OddsComparison(api_client)

@router.get("/")
async def get_matches(
    days_ahead: int = Query(7, description="Number of days ahead to fetch matches"),
    league_id: Optional[int] = Query(None, description="Filter by league ID")
):
    """
    Get upcoming matches with predictions
    """
    try:
        # Get date range
        today = datetime.now().date()
        end_date = today + timedelta(days=days_ahead)
        
        # Fetch matches from the data fetcher
        matches = data_fetcher.get_upcoming_matches(
            start_date=today.isoformat(),
            end_date=end_date.isoformat(),
            league_id=league_id
        )
        
        # Generate predictions for each match
        for match in matches:
            prediction = prediction_model.predict_match(
                home_team_id=match["home_team"]["id"],
                away_team_id=match["away_team"]["id"]
            )
            match["prediction"] = prediction
            
            # Add value bet information
            value_bets = odds_comparison.find_value_bets(
                match_id=match["id"],
                prediction=prediction
            )
            match["value_bets"] = value_bets
        
        return {"matches": matches}
    
    except Exception as e:
        logger.error(f"Error fetching matches: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error fetching matches: {str(e)}")

@router.get("/{match_id}")
async def get_match_details(match_id: int):
    """
    Get detailed information about a specific match
    """
    try:
        # Fetch match details
        match = data_fetcher.get_match_details(match_id)
        if not match:
            raise HTTPException(status_code=404, detail=f"Match with ID {match_id} not found")
        
        # Generate prediction
        prediction = prediction_model.predict_match(
            home_team_id=match["home_team"]["id"],
            away_team_id=match["away_team"]["id"]
        )
        match["prediction"] = prediction
        
        # Add head-to-head information
        h2h = data_fetcher.get_head_to_head(
            team1_id=match["home_team"]["id"],
            team2_id=match["away_team"]["id"]
        )
        match["head_to_head"] = h2h
        
        # Add value bet information
        value_bets = odds_comparison.find_value_bets(
            match_id=match_id,
            prediction=prediction
        )
        match["value_bets"] = value_bets
        
        return match
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching match details: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error fetching match details: {str(e)}")
