from fastapi import APIRouter, HTTPException, Query
from typing import List, Dict, Any, Optional
import logging

# Import components from the original application
from api.api_client import APIClient
from api.odds_comparison import OddsComparison

# Setup logging
logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/api/odds", tags=["odds"])

# Initialize components
api_client = APIClient()
odds_comparison = OddsComparison(api_client)

@router.get("/match/{match_id}")
async def get_match_odds(match_id: int):
    """
    Get odds from different bookmakers for a specific match
    """
    try:
        # Fetch odds from the odds comparison module
        odds = odds_comparison.get_match_odds(match_id)
        if not odds:
            raise HTTPException(status_code=404, detail=f"Odds for match ID {match_id} not found")
        
        return {"match_id": match_id, "odds": odds}
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching match odds: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error fetching match odds: {str(e)}")

@router.get("/comparison/{match_id}")
async def get_odds_comparison(match_id: int):
    """
    Get odds comparison with AI predictions for a specific match
    """
    try:
        # Get odds comparison from the odds comparison module
        comparison = odds_comparison.compare_odds_with_prediction(match_id)
        if not comparison:
            raise HTTPException(status_code=404, detail=f"Odds comparison for match ID {match_id} not found")
        
        return comparison
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching odds comparison: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error fetching odds comparison: {str(e)}")

@router.get("/bookmakers")
async def get_bookmakers():
    """
    Get list of available bookmakers
    """
    try:
        # Get bookmakers from the odds comparison module
        bookmakers = odds_comparison.get_bookmakers()
        
        return {"bookmakers": bookmakers}
    
    except Exception as e:
        logger.error(f"Error fetching bookmakers: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error fetching bookmakers: {str(e)}")
