"""
Script to test the data fetching module by retrieving sample data from API-Football.
"""

import os
import sys
import logging
from dotenv import load_dotenv

# Add the project root directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from api.data_fetcher import DataFetcher

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("data_fetcher_test.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("data_fetcher_test")

# Load environment variables from .env file
load_dotenv()

def main():
    """
    Test the data fetching module by retrieving sample data from API-Football.
    """
    # Get API key from environment variable
    api_key = os.getenv("API_FOOTBALL_KEY")
    if not api_key:
        logger.error("API_FOOTBALL_KEY environment variable not set")
        print("Please set the API_FOOTBALL_KEY environment variable with your API-Football API key")
        return
    
    # Create data fetcher
    data_fetcher = DataFetcher(api_key)
    
    # Test fetching upcoming matches
    try:
        logger.info("Fetching upcoming matches for the next 7 days")
        matches = data_fetcher.fetch_upcoming_matches(days=7)
        logger.info(f"Fetched {len(matches)} upcoming matches")
        
        # Print some match details
        for i, match in enumerate(matches[:5], 1):
            logger.info(f"Match {i}: {match.home_team.name} vs {match.away_team.name} on {match.match_date}")
            
            # Fetch betting odds for this match
            odds = data_fetcher.fetch_betting_odds(match.id)
            if odds:
                logger.info(f"  Fetched betting odds from {len(odds)} bookmakers")
            
            # Fetch API predictions for this match
            prediction = data_fetcher.fetch_api_predictions(match.id)
            if prediction:
                logger.info(f"  Prediction: Home win prob: {prediction.home_win_probability}, Draw prob: {prediction.draw_probability}, Away win prob: {prediction.away_win_probability}")
    
    except Exception as e:
        logger.error(f"Error testing data fetcher: {str(e)}")

if __name__ == "__main__":
    main()
