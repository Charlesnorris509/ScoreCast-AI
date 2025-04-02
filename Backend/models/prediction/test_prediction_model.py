"""
Test script for the prediction model module.
This script trains a model on historical data and makes predictions for upcoming matches.
"""

import os
import sys
import logging
from datetime import datetime
from dotenv import load_dotenv

# Add the project root directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from models.prediction.prediction_model import MatchPredictionModel
from api.data_fetcher import DataFetcher

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("prediction_model_test.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("prediction_model_test")

# Load environment variables from .env file
load_dotenv()

def main():
    """
    Test the prediction model by training on historical data and making predictions.
    """
    # Create prediction model
    model = MatchPredictionModel(model_type="random_forest", model_version="v1.0")
    
    # Check if model file exists
    model_path = os.path.join(os.path.dirname(__file__), "../../data/models/match_prediction_model.pkl")
    
    if os.path.exists(model_path):
        # Load existing model
        logger.info(f"Loading existing model from {model_path}")
        model.load_model(model_path)
    else:
        # Train new model
        logger.info("Training new prediction model")
        try:
            metrics = model.train(days_back=365, test_size=0.2)
            logger.info(f"Model training metrics: {metrics}")
            
            # Save model
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            model.save_model(model_path)
            logger.info(f"Model saved to {model_path}")
        except Exception as e:
            logger.error(f"Error training model: {str(e)}")
            return
    
    # Make predictions for upcoming matches
    try:
        logger.info("Making predictions for upcoming matches")
        predictions = model.predict_upcoming_matches(days_ahead=7)
        
        logger.info(f"Made predictions for {len(predictions)} upcoming matches")
        
        # Print some prediction details
        for i, prediction in enumerate(predictions[:5], 1):
            logger.info(f"Prediction {i}: {prediction['home_team_name']} vs {prediction['away_team_name']}")
            logger.info(f"  Predicted result: {prediction['predicted_result']} ({prediction['predicted_home_score']}-{prediction['predicted_away_score']})")
            logger.info(f"  Confidence: {prediction['confidence']:.2f}")
            
            if prediction['is_value_bet']:
                logger.info(f"  Value bet: {prediction['value_bet_type']} (EV: {prediction['expected_value']:.2f})")
    
    except Exception as e:
        logger.error(f"Error making predictions: {str(e)}")
    
    # Evaluate past predictions
    try:
        logger.info("Evaluating past predictions")
        evaluation = model.evaluate_past_predictions(days_back=30)
        
        if evaluation:
            logger.info(f"Evaluation metrics: {evaluation}")
            logger.info(f"Prediction accuracy: {evaluation.get('accuracy', 0):.2f}")
            logger.info(f"Value bet accuracy: {evaluation.get('value_bet_accuracy', 0):.2f}")
    
    except Exception as e:
        logger.error(f"Error evaluating predictions: {str(e)}")

if __name__ == "__main__":
    main()
