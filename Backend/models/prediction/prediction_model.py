"""
Machine learning prediction models for the Soccer Match Prediction Application.
This module provides classes for training and using prediction models.
"""

import os
import sys
import logging
import pickle
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple, Optional
from datetime import datetime
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, log_loss
from sklearn.calibration import CalibratedClassifierCV
from sqlalchemy.orm import Session

# Add the project root directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from config.database import SessionLocal
from models.database.models import Match, Prediction, BettingOdds
from models.prediction.feature_engineering import FeatureEngineering

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("prediction_model.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("prediction_model")

class MatchPredictionModel:
    """Class for training and using match prediction models."""
    
    def __init__(self, model_type: str = "random_forest", model_version: str = "v1.0"):
        """
        Initialize the prediction model.
        
        Args:
            model_type: Type of model to use ("random_forest", "gradient_boosting", or "logistic_regression")
            model_version: Version identifier for the model
        """
        self.model_type = model_type
        self.model_version = model_version
        self.model = None
        self.feature_engineering = FeatureEngineering()
        self.feature_columns = None
        self.target_column = "result"  # H (home win), D (draw), A (away win)
        self.scaler = StandardScaler()
        
    def train(self, days_back: int = 365, test_size: float = 0.2, random_state: int = 42) -> Dict[str, float]:
        """
        Train the prediction model on historical match data.
        
        Args:
            days_back: Number of days to look back for training data
            test_size: Proportion of data to use for testing
            random_state: Random seed for reproducibility
            
        Returns:
            Dictionary containing model evaluation metrics
        """
        logger.info(f"Training {self.model_type} model on data from the past {days_back} days")
        
        # Prepare training dataset
        data = self.feature_engineering.prepare_training_dataset(days_back)
        
        if data.empty:
            logger.error("No training data available")
            raise ValueError("No training data available")
        
        # Remove rows with missing target values
        data = data.dropna(subset=[self.target_column])
        
        # Split features and target
        X = data.drop(columns=["match_id", "match_date", "home_score", "away_score", self.target_column])
        y = data[self.target_column]
        
        # Save feature columns for prediction
        self.feature_columns = X.columns.tolist()
        
        # Split into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        
        # Create and train the model
        if self.model_type == "random_forest":
            base_model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=random_state
            )
        elif self.model_type == "gradient_boosting":
            base_model = GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                random_state=random_state
            )
        elif self.model_type == "logistic_regression":
            base_model = LogisticRegression(
                C=1.0,
                max_iter=1000,
                random_state=random_state
            )
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
        
        # Use calibrated classifier to get better probability estimates
        self.model = CalibratedClassifierCV(base_model, cv=5, method='isotonic')
        
        # Fit the model
        self.model.fit(X_train, y_train)
        
        # Evaluate the model
        y_pred = self.model.predict(X_test)
        y_prob = self.model.predict_proba(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        loss = log_loss(y_test, y_prob)
        
        metrics = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "log_loss": loss
        }
        
        logger.info(f"Model training completed with metrics: {metrics}")
        return metrics
    
    def predict(self, match_id: int) -> Dict[str, Any]:
        """
        Make a prediction for a specific match.
        
        Args:
            match_id: Match ID
            
        Returns:
            Dictionary containing prediction results
        """
        logger.info(f"Making prediction for match {match_id}")
        
        if self.model is None:
            logger.error("Model not trained")
            raise ValueError("Model not trained")
        
        # Prepare features for the match
        features = self.feature_engineering.prepare_match_features(match_id)
        
        if not features:
            logger.error(f"Could not prepare features for match {match_id}")
            raise ValueError(f"Could not prepare features for match {match_id}")
        
        # Convert to DataFrame
        features_df = pd.DataFrame([features])
        
        # Extract features used during training
        X = features_df[self.feature_columns]
        
        # Make prediction
        result_probs = self.model.predict_proba(X)[0]
        
        # Get class labels (should be ["A", "D", "H"] or similar)
        classes = self.model.classes_
        
        # Create probability dictionary
        probs = {class_label: prob for class_label, prob in zip(classes, result_probs)}
        
        # Get most likely result
        predicted_result = classes[np.argmax(result_probs)]
        
        # Estimate scores based on predicted result and team ratings
        home_attack = features.get("home_team_attack_rating", 1.0)
        away_attack = features.get("away_team_attack_rating", 1.0)
        home_defense = features.get("home_team_defense_rating", 1.0)
        away_defense = features.get("away_team_defense_rating", 1.0)
        
        # Base expected goals
        home_xg = home_attack * (1 / away_defense) * 1.3  # Home advantage factor
        away_xg = away_attack * (1 / home_defense)
        
        # Adjust based on predicted result
        if predicted_result == "H":  # Home win
            home_xg = max(home_xg, away_xg + 0.5)
        elif predicted_result == "A":  # Away win
            away_xg = max(away_xg, home_xg + 0.5)
        elif predicted_result == "D":  # Draw
            avg_xg = (home_xg + away_xg) / 2
            home_xg = away_xg = avg_xg
        
        # Calculate confidence based on probability of predicted result
        confidence = probs.get(predicted_result, 0.33)
        
        prediction_result = {
            "match_id": match_id,
            "predicted_result": predicted_result,
            "home_win_probability": probs.get("H", 0),
            "draw_probability": probs.get("D", 0),
            "away_win_probability": probs.get("A", 0),
            "predicted_home_score": round(home_xg, 1),
            "predicted_away_score": round(away_xg, 1),
            "confidence": confidence,
            "model_version": f"{self.model_type}-{self.model_version}"
        }
        
        logger.info(f"Prediction for match {match_id}: {prediction_result}")
        return prediction_result
    
    def save_model(self, filepath: str) -> None:
        """
        Save the trained model to a file.
        
        Args:
            filepath: Path to save the model
        """
        if self.model is None:
            logger.error("No model to save")
            raise ValueError("No model to save")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save model and metadata
        model_data = {
            "model": self.model,
            "feature_columns": self.feature_columns,
            "model_type": self.model_type,
            "model_version": self.model_version,
            "target_column": self.target_column,
            "timestamp": datetime.now().isoformat()
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str) -> None:
        """
        Load a trained model from a file.
        
        Args:
            filepath: Path to the saved model
        """
        if not os.path.exists(filepath):
            logger.error(f"Model file not found: {filepath}")
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data["model"]
        self.feature_columns = model_data["feature_columns"]
        self.model_type = model_data["model_type"]
        self.model_version = model_data["model_version"]
        self.target_column = model_data["target_column"]
        
        logger.info(f"Model loaded from {filepath}")
    
    def save_prediction_to_db(self, prediction: Dict[str, Any]) -> None:
        """
        Save a prediction to the database.
        
        Args:
            prediction: Dictionary containing prediction results
        """
        logger.info(f"Saving prediction for match {prediction['match_id']} to database")
        
        db = SessionLocal()
        
        try:
            # Check if prediction already exists
            existing_prediction = db.query(Prediction).filter(
                Prediction.match_id == prediction["match_id"],
                Prediction.model_version == prediction["model_version"]
            ).first()
            
            if existing_prediction:
                # Update existing prediction
                existing_prediction.predicted_home_score = prediction["predicted_home_score"]
                existing_prediction.predicted_away_score = prediction["predicted_away_score"]
                existing_prediction.home_win_probability = prediction["home_win_probability"]
                existing_prediction.draw_probability = prediction["draw_probability"]
                existing_prediction.away_win_probability = prediction["away_win_probability"]
                existing_prediction.confidence = prediction["confidence"]
                
                db.add(existing_prediction)
                db.commit()
                
                logger.info(f"Updated prediction for match {prediction['match_id']}")
            else:
                # Create new prediction
                new_prediction = Prediction(
                    match_id=prediction["match_id"],
                    predicted_home_score=prediction["predicted_home_score"],
                    predicted_away_score=prediction["predicted_away_score"],
                    home_win_probability=prediction["home_win_probability"],
                    draw_probability=prediction["draw_probability"],
                    away_win_probability=prediction["away_win_probability"],
                    confidence=prediction["confidence"],
                    model_version=prediction["model_version"]
                )
                
                db.add(new_prediction)
                db.commit()
                
                logger.info(f"Created prediction for match {prediction['match_id']}")
        
        except Exception as e:
            db.rollback()
            logger.error(f"Error saving prediction to database: {str(e)}")
            raise
        
        finally:
            db.close()
    
    def identify_value_bets(self, match_id: int, prediction: Dict[str, Any]) -> Dict[str, Any]:
        """
        Identify value bets by comparing prediction with betting odds.
        
        Args:
            match_id: Match ID
            prediction: Dictionary containing prediction results
            
        Returns:
            Dictionary containing value bet information
        """
        logger.info(f"Identifying value bets for match {match_id}")
        
        db = SessionLocal()
        
        try:
            # Get betting odds for the match
            betting_odds = db.query(BettingOdds).filter(
                BettingOdds.match_id == match_id
            ).all()
            
            if not betting_odds:
                logger.warning(f"No betting odds found for match {match_id}")
                return {}
            
            # Get average odds across bookmakers
            home_win_odds = []
            draw_odds = []
            away_win_odds = []
            
            for odds in betting_odds:
                if odds.home_win_odds:
                    home_win_odds.append(odds.home_win_odds)
                if odds.draw_odds:
                    draw_odds.append(odds.draw_odds)
                if odds.away_win_odds:
                    away_win_odds.append(odds.away_win_odds)
            
            avg_home_win_odds = sum(home_win_odds) / len(home_win_odds) if home_win_odds else 0
            avg_draw_odds = sum(draw_odds) / len(draw_odds) if draw_odds else 0
            avg_away_win_odds = sum(away_win_odds) / len(away_win_odds) if away_win_odds else 0
            
            # Convert odds to implied probabilities
            implied_home_win_prob = 1 / avg_home_win_odds if avg_home_win_odds > 0 else 0
            implied_draw_prob = 1 / avg_draw_odds if avg_draw_odds > 0 else 0
            implied_away_win_prob = 1 / avg_away_win_odds if avg_away_win_odds > 0 else 0
            
            # Calculate value (predicted probability - implied probability)
            home_win_value = prediction["home_win_probability"] - implied_home_win_prob
            draw_value = prediction["draw_probability"] - implied_draw_prob
            away_win_value = prediction["away_win_probability"] - implied_away_win_prob
            
            # Identify best value bet
            values = {
                "home_win": home_win_value,
                "draw": draw_value,
                "away_win": away_win_value
            }
            
            best_value_bet = max(values.items(), key=lambda x: x[1])
            
            # Only consider as value bet if the difference is significant
            is_value_bet = best_value_bet[1] > 0.05
            
            value_bet_info = {
                "is_value_bet": is_value_bet,
                "value_bet_type": best_value_bet[0] if is_value_bet else None,
                "expected_value": best_value_bet[1] if is_value_bet else 0,
                "avg_home_win_odds": avg_home_win_odds,
                "avg_draw_odds": avg_draw_odds,
                "avg_away_win_odds": avg_away_win_odds,
                "implied_home_win_prob": implied_home_win_prob,
                "implied_draw_prob": implied_draw_prob,
                "implied_away_win_prob": implied_away_win_prob,
                "home_win_value": home_win_value,
                "draw_value": draw_value,
                "away_win_value": away_win_value
            }
            
            logger.info(f"Value bet analysis for match {match_id}: {value_bet_info}")
            return value_bet_info
        
        except Exception as e:
            logger.error(f"Error identifying value bets: {str(e)}")
            raise
        
        finally:
            db.close()
    
    def update_prediction_with_value_bet(self, match_id: int, value_bet_info: Dict[str, Any]) -> None:
        """
        Update a prediction in the database with value bet information.
        
        Args:
            match_id: Match ID
            value_bet_info: Dictionary containing value bet information
        """
        logger.info(f"Updating prediction for match {match_id} with value bet information")
        
        db = SessionLocal()
        
        try:
            # Get prediction from database
            prediction = db.query(Prediction).filter(
                Prediction.match_id == match_id,
                Prediction.model_version == f"{self.model_type}-{self.model_version}"
            ).first()
            
            if not prediction:
                logger.warning(f"No prediction found for match {match_id}")
                return
            
            # Update prediction with value bet information
            prediction.is_value_bet = value_bet_info.get("is_value_bet", False)
            prediction.value_bet_type = value_bet_info.get("value_bet_type")
            prediction.expected_value = value_bet_info.get("expected_value", 0)
            
            db.add(prediction)
            db.commit()
            
            logger.info(f"Updated prediction for match {match_id} with value bet information")
        
        except Exception as e:
            db.rollback()
            logger.error(f"Error updating prediction with value bet information: {str(e)}")
            raise
        
        finally:
            db.close()
    
    def predict_upcoming_matches(self, days_ahead: int = 7) -> List[Dict[str, Any]]:
        """
        Make predictions for upcoming matches.
        
        Args:
            days_ahead: Number of days to look ahead
            
        Returns:
            List of dictionaries containing prediction results
        """
        logger.info(f"Making predictions for upcoming matches in the next {days_ahead} days")
        
        db = SessionLocal()
        
        try:
            # Get upcoming matches
            current_date = datetime.now()
            end_date = current_date + pd.Timedelta(days=days_ahead)
            
            upcoming_matches = db.query(Match).filter(
                Match.match_date >= current_date,
                Match.match_date <= end_date,
                Match.status == "NS"  # Not Started
            ).all()
            
            logger.info(f"Found {len(upcoming_matches)} upcoming matches")
            
            predictions = []
            for match in upcoming_matches:
                try:
                    # Make prediction
                    prediction = self.predict(match.id)
                    
                    # Identify value bets
                    value_bet_info = self.identify_value_bets(match.id, prediction)
                    
                    # Save prediction to database
                    self.save_prediction_to_db(prediction)
                    
                    # Update prediction with value bet information
                    if value_bet_info:
                        self.update_prediction_with_value_bet(match.id, value_bet_info)
                    
                    # Add match information to prediction
                    prediction.update({
                        "home_team_name": match.home_team.name if match.home_team else None,
                        "away_team_name": match.away_team.name if match.away_team else None,
                        "league_name": match.league.name if match.league else None,
                        "match_date": match.match_date,
                        "is_value_bet": value_bet_info.get("is_value_bet", False),
                        "value_bet_type": value_bet_info.get("value_bet_type"),
                        "expected_value": value_bet_info.get("expected_value", 0)
                    })
                    
                    predictions.append(prediction)
                
                except Exception as e:
                    logger.error(f"Error making prediction for match {match.id}: {str(e)}")
            
            return predictions
        
        except Exception as e:
            logger.error(f"Error predicting upcoming matches: {str(e)}")
            raise
        
        finally:
            db.close()
    
    def evaluate_past_predictions(self, days_back: int = 30) -> Dict[str, float]:
        """
        Evaluate the accuracy of past predictions.
        
        Args:
            days_back: Number of days to look back
            
        Returns:
            Dictionary containing evaluation metrics
        """
        logger.info(f"Evaluating past predictions from the last {days_back} days")
        
        db = SessionLocal()
        
        try:
            # Calculate the date to look back to
            cutoff_date = datetime.now() - pd.Timedelta(days=days_back)
            
            # Get completed matches with predictions
            matches = db.query(Match).join(
                Prediction, Match.id == Prediction.match_id
            ).filter(
                Match.match_date >= cutoff_date,
                Match.match_date <= datetime.now(),
                Match.status == "FT",  # Full Time (completed matches)
                Prediction.model_version == f"{self.model_type}-{self.model_version}"
            ).all()
            
            if not matches:
                logger.warning("No past predictions found for evaluation")
                return {}
            
            logger.info(f"Found {len(matches)} matches with predictions")
            
            # Collect actual results and predictions
            actual_results = []
            predicted_results = []
            
            for match in matches:
                # Get actual result
                if match.home_score > match.away_score:
                    actual_result = "H"
                elif match.home_score == match.away_score:
                    actual_result = "D"
                else:
                    actual_result = "A"
                
                # Get prediction
                prediction = db.query(Prediction).filter(
                    Prediction.match_id == match.id,
                    Prediction.model_version == f"{self.model_type}-{self.model_version}"
                ).first()
                
                if not prediction:
                    continue
                
                # Determine predicted result
                probs = {
                    "H": prediction.home_win_probability,
                    "D": prediction.draw_probability,
                    "A": prediction.away_win_probability
                }
                predicted_result = max(probs.items(), key=lambda x: x[1])[0]
                
                # Update was_correct flag in prediction
                prediction.was_correct = (predicted_result == actual_result)
                db.add(prediction)
                
                actual_results.append(actual_result)
                predicted_results.append(predicted_result)
            
            db.commit()
            
            # Calculate metrics
            if not actual_results:
                logger.warning("No valid predictions found for evaluation")
                return {}
            
            accuracy = accuracy_score(actual_results, predicted_results)
            precision = precision_score(actual_results, predicted_results, average='weighted', zero_division=0)
            recall = recall_score(actual_results, predicted_results, average='weighted', zero_division=0)
            f1 = f1_score(actual_results, predicted_results, average='weighted', zero_division=0)
            
            # Calculate value bet performance
            value_bets = db.query(Prediction).filter(
                Prediction.match_id.in_([m.id for m in matches]),
                Prediction.is_value_bet == True,
                Prediction.model_version == f"{self.model_type}-{self.model_version}"
            ).all()
            
            value_bet_correct = sum(1 for vb in value_bets if vb.was_correct)
            value_bet_accuracy = value_bet_correct / len(value_bets) if value_bets else 0
            
            metrics = {
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
                "num_predictions": len(actual_results),
                "value_bet_accuracy": value_bet_accuracy,
                "num_value_bets": len(value_bets)
            }
            
            logger.info(f"Evaluation metrics: {metrics}")
            return metrics
        
        except Exception as e:
            db.rollback()
            logger.error(f"Error evaluating past predictions: {str(e)}")
            raise
        
        finally:
            db.close()
