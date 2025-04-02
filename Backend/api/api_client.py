"""
API client for the Soccer Match Prediction Application.
This module provides functions to interact with the API-Football service.
"""

import os
import requests
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("api_client.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("api_client")

class APIFootballClient:
    """Client for interacting with the API-Football service."""
    
    BASE_URL = "https://api-football-v1.p.rapidapi.com/v3"
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the API client.
        
        Args:
            api_key: API key for API-Football. If not provided, will try to get from environment variable.
        """
        self.api_key = api_key or os.getenv("API_FOOTBALL_KEY")
        if not self.api_key:
            logger.warning("API key not provided. Please set API_FOOTBALL_KEY environment variable.")
        
        self.headers = {
            "X-RapidAPI-Key": self.api_key,
            "X-RapidAPI-Host": "api-football-v1.p.rapidapi.com"
        }
    
    def _make_request(self, endpoint: str, params: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Make a request to the API-Football service.
        
        Args:
            endpoint: API endpoint to call
            params: Query parameters for the request
            
        Returns:
            API response as a dictionary
            
        Raises:
            Exception: If the request fails
        """
        url = f"{self.BASE_URL}/{endpoint}"
        
        try:
            logger.info(f"Making request to {endpoint} with params {params}")
            response = requests.get(url, headers=self.headers, params=params)
            response.raise_for_status()
            
            data = response.json()
            logger.info(f"Request to {endpoint} successful")
            
            return data
        except requests.exceptions.RequestException as e:
            logger.error(f"Request to {endpoint} failed: {str(e)}")
            raise
    
    def get_leagues(self, country: Optional[str] = None, season: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get available leagues and cups.
        
        Args:
            country: Filter by country name
            season: Filter by season (e.g., 2023)
            
        Returns:
            List of leagues
        """
        params = {}
        if country:
            params["country"] = country
        if season:
            params["season"] = season
            
        response = self._make_request("leagues", params)
        return response.get("response", [])
    
    def get_teams(self, league_id: int, season: int) -> List[Dict[str, Any]]:
        """
        Get teams for a specific league and season.
        
        Args:
            league_id: League ID
            season: Season year (e.g., 2023)
            
        Returns:
            List of teams
        """
        params = {
            "league": league_id,
            "season": season
        }
        
        response = self._make_request("teams", params)
        return response.get("response", [])
    
    def get_players(self, team_id: int, season: int) -> List[Dict[str, Any]]:
        """
        Get players for a specific team and season.
        
        Args:
            team_id: Team ID
            season: Season year (e.g., 2023)
            
        Returns:
            List of players
        """
        params = {
            "team": team_id,
            "season": season
        }
        
        response = self._make_request("players", params)
        return response.get("response", [])
    
    def get_fixtures(self, 
                    league_id: Optional[int] = None, 
                    team_id: Optional[int] = None,
                    season: Optional[int] = None,
                    date: Optional[str] = None,
                    from_date: Optional[str] = None,
                    to_date: Optional[str] = None,
                    status: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get fixtures (matches) based on various filters.
        
        Args:
            league_id: Filter by league ID
            team_id: Filter by team ID
            season: Filter by season (e.g., 2023)
            date: Filter by specific date (format: YYYY-MM-DD)
            from_date: Filter from date (format: YYYY-MM-DD)
            to_date: Filter to date (format: YYYY-MM-DD)
            status: Filter by match status (NS, LIVE, FT, etc.)
            
        Returns:
            List of fixtures
        """
        params = {}
        if league_id:
            params["league"] = league_id
        if team_id:
            params["team"] = team_id
        if season:
            params["season"] = season
        if date:
            params["date"] = date
        if from_date:
            params["from"] = from_date
        if to_date:
            params["to"] = to_date
        if status:
            params["status"] = status
            
        response = self._make_request("fixtures", params)
        return response.get("response", [])
    
    def get_fixture_statistics(self, fixture_id: int) -> List[Dict[str, Any]]:
        """
        Get statistics for a specific fixture.
        
        Args:
            fixture_id: Fixture ID
            
        Returns:
            List of team statistics for the fixture
        """
        params = {
            "fixture": fixture_id
        }
        
        response = self._make_request("fixtures/statistics", params)
        return response.get("response", [])
    
    def get_fixture_events(self, fixture_id: int) -> List[Dict[str, Any]]:
        """
        Get events for a specific fixture (goals, cards, etc.).
        
        Args:
            fixture_id: Fixture ID
            
        Returns:
            List of events for the fixture
        """
        params = {
            "fixture": fixture_id
        }
        
        response = self._make_request("fixtures/events", params)
        return response.get("response", [])
    
    def get_fixture_lineups(self, fixture_id: int) -> List[Dict[str, Any]]:
        """
        Get lineups for a specific fixture.
        
        Args:
            fixture_id: Fixture ID
            
        Returns:
            List of team lineups for the fixture
        """
        params = {
            "fixture": fixture_id
        }
        
        response = self._make_request("fixtures/lineups", params)
        return response.get("response", [])
    
    def get_fixture_player_statistics(self, fixture_id: int) -> List[Dict[str, Any]]:
        """
        Get player statistics for a specific fixture.
        
        Args:
            fixture_id: Fixture ID
            
        Returns:
            List of player statistics for the fixture
        """
        params = {
            "fixture": fixture_id
        }
        
        response = self._make_request("fixtures/players", params)
        return response.get("response", [])
    
    def get_head_to_head(self, team1_id: int, team2_id: int, last: int = 10) -> List[Dict[str, Any]]:
        """
        Get head-to-head fixtures between two teams.
        
        Args:
            team1_id: First team ID
            team2_id: Second team ID
            last: Number of last matches to retrieve
            
        Returns:
            List of fixtures between the two teams
        """
        params = {
            "h2h": f"{team1_id}-{team2_id}",
            "last": last
        }
        
        response = self._make_request("fixtures/headtohead", params)
        return response.get("response", [])
    
    def get_odds(self, fixture_id: int, bookmaker_id: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get betting odds for a specific fixture.
        
        Args:
            fixture_id: Fixture ID
            bookmaker_id: Filter by bookmaker ID
            
        Returns:
            List of odds for the fixture
        """
        params = {
            "fixture": fixture_id
        }
        if bookmaker_id:
            params["bookmaker"] = bookmaker_id
            
        response = self._make_request("odds", params)
        return response.get("response", [])
    
    def get_predictions(self, fixture_id: int) -> Dict[str, Any]:
        """
        Get predictions for a specific fixture.
        
        Args:
            fixture_id: Fixture ID
            
        Returns:
            Prediction data for the fixture
        """
        params = {
            "fixture": fixture_id
        }
        
        response = self._make_request("predictions", params)
        if response.get("response"):
            return response["response"][0]
        return {}
