from fastapi import APIRouter, HTTPException, Query
from typing import List, Dict, Any, Optional
import logging

# Import components from the original application
from api.data_fetcher import DataFetcher
from api.api_client import APIClient

# Setup logging
logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/api/leagues", tags=["leagues"])

# Initialize components
api_client = APIClient()
data_fetcher = DataFetcher(api_client)

@router.get("/")
async def get_leagues():
    """
    Get list of available leagues
    """
    try:
        # Fetch leagues from the data fetcher
        leagues = data_fetcher.get_leagues()
        
        return {"leagues": leagues}
    
    except Exception as e:
        logger.error(f"Error fetching leagues: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error fetching leagues: {str(e)}")

@router.get("/{league_id}")
async def get_league_details(league_id: int):
    """
    Get detailed information about a specific league
    """
    try:
        # Fetch league details
        league = data_fetcher.get_league_details(league_id)
        if not league:
            raise HTTPException(status_code=404, detail=f"League with ID {league_id} not found")
        
        return league
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching league details: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error fetching league details: {str(e)}")

@router.get("/{league_id}/standings")
async def get_league_standings(league_id: int):
    """
    Get current standings for a specific league
    """
    try:
        # Fetch league standings
        standings = data_fetcher.get_league_standings(league_id)
        if not standings:
            raise HTTPException(status_code=404, detail=f"Standings for league ID {league_id} not found")
        
        return {"league_id": league_id, "standings": standings}
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching league standings: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error fetching league standings: {str(e)}")

@router.get("/{league_id}/matches")
async def get_league_matches(
    league_id: int,
    days_ahead: int = Query(30, description="Number of days ahead to fetch matches")
):
    """
    Get upcoming matches for a specific league
    """
    try:
        # Get league matches from the data fetcher
        matches = data_fetcher.get_league_matches(
            league_id=league_id,
            days_ahead=days_ahead
        )
        
        return {"league_id": league_id, "matches": matches}
    
    except Exception as e:
        logger.error(f"Error fetching league matches: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error fetching league matches: {str(e)}")
