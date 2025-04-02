"""
Feature engineering module for the Soccer Match Prediction Application.
This module provides functions to extract and transform features from match data.
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple
from datetime import datetime, timedelta
from sqlalchemy.orm import Session

# Add the project root directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from config.database import SessionLocal
from models.database.models import (
    Team, Player, League, Match, MatchStatistic, 
    PlayerStatistic, BettingOdds, Prediction
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("feature_engineering.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("feature_engineering")

class FeatureEngineering:
    """Class for extracting and transforming features from match data."""
    
    def __init__(self):
        """Initialize the feature engineering module."""
        pass
    
    def get_historical_matches(self, days_back: int = 365) -> pd.DataFrame:
        """
        Get historical matches from the database.
        
        Args:
            days_back: Number of days to look back for historical matches
            
        Returns:
            DataFrame containing historical matches
        """
        logger.info(f"Getting historical matches from the past {days_back} days")
        
        db = SessionLocal()
        
        try:
            # Calculate the date to look back to
            cutoff_date = datetime.now() - timedelta(days=days_back)
            
            # Query completed matches
            matches = db.query(Match).filter(
                Match.match_date >= cutoff_date,
                Match.status == "FT"  # Full Time (completed matches)
            ).all()
            
            logger.info(f"Found {len(matches)} historical matches")
            
            # Convert to DataFrame
            matches_data = []
            for match in matches:
                match_dict = {
                    "match_id": match.id,
                    "api_id": match.api_id,
                    "league_id": match.league_id,
                    "league_name": match.league.name if match.league else None,
                    "home_team_id": match.home_team_id,
                    "home_team_name": match.home_team.name if match.home_team else None,
                    "away_team_id": match.away_team_id,
                    "away_team_name": match.away_team.name if match.away_team else None,
                    "match_date": match.match_date,
                    "home_score": match.home_score,
                    "away_score": match.away_score,
                    "home_halftime_score": match.home_halftime_score,
                    "away_halftime_score": match.away_halftime_score,
                    "venue": match.venue,
                    "referee": match.referee,
                    "temperature": match.temperature,
                    "weather_code": match.weather_code,
                    "weather_description": match.weather_description
                }
                matches_data.append(match_dict)
            
            return pd.DataFrame(matches_data)
        
        except Exception as e:
            logger.error(f"Error getting historical matches: {str(e)}")
            raise
        
        finally:
            db.close()
    
    def get_team_form(self, team_id: int, before_date: datetime, num_matches: int = 5) -> pd.DataFrame:
        """
        Get team form (recent match results) before a specific date.
        
        Args:
            team_id: Team ID
            before_date: Date to look back from
            num_matches: Number of recent matches to include
            
        Returns:
            DataFrame containing team form data
        """
        logger.info(f"Getting form for team {team_id} before {before_date}")
        
        db = SessionLocal()
        
        try:
            # Query recent matches for the team before the specified date
            matches = db.query(Match).filter(
                ((Match.home_team_id == team_id) | (Match.away_team_id == team_id)),
                Match.match_date < before_date,
                Match.status == "FT"  # Full Time (completed matches)
            ).order_by(Match.match_date.desc()).limit(num_matches).all()
            
            # Convert to DataFrame
            form_data = []
            for match in matches:
                is_home = match.home_team_id == team_id
                opponent_id = match.away_team_id if is_home else match.home_team_id
                opponent_name = match.away_team.name if is_home else match.home_team.name
                
                team_score = match.home_score if is_home else match.away_score
                opponent_score = match.away_score if is_home else match.home_score
                
                # Determine result (win, draw, loss)
                if team_score > opponent_score:
                    result = "W"
                elif team_score == opponent_score:
                    result = "D"
                else:
                    result = "L"
                
                form_dict = {
                    "match_id": match.id,
                    "match_date": match.match_date,
                    "is_home": is_home,
                    "opponent_id": opponent_id,
                    "opponent_name": opponent_name,
                    "team_score": team_score,
                    "opponent_score": opponent_score,
                    "result": result
                }
                form_data.append(form_dict)
            
            return pd.DataFrame(form_data)
        
        except Exception as e:
            logger.error(f"Error getting team form: {str(e)}")
            raise
        
        finally:
            db.close()
    
    def get_head_to_head(self, team1_id: int, team2_id: int, before_date: datetime, num_matches: int = 5) -> pd.DataFrame:
        """
        Get head-to-head match history between two teams before a specific date.
        
        Args:
            team1_id: First team ID
            team2_id: Second team ID
            before_date: Date to look back from
            num_matches: Number of recent matches to include
            
        Returns:
            DataFrame containing head-to-head match data
        """
        logger.info(f"Getting head-to-head for teams {team1_id} and {team2_id} before {before_date}")
        
        db = SessionLocal()
        
        try:
            # Query head-to-head matches before the specified date
            matches = db.query(Match).filter(
                (
                    ((Match.home_team_id == team1_id) & (Match.away_team_id == team2_id)) |
                    ((Match.home_team_id == team2_id) & (Match.away_team_id == team1_id))
                ),
                Match.match_date < before_date,
                Match.status == "FT"  # Full Time (completed matches)
            ).order_by(Match.match_date.desc()).limit(num_matches).all()
            
            # Convert to DataFrame
            h2h_data = []
            for match in matches:
                team1_is_home = match.home_team_id == team1_id
                
                team1_score = match.home_score if team1_is_home else match.away_score
                team2_score = match.away_score if team1_is_home else match.home_score
                
                # Determine result for team1 (win, draw, loss)
                if team1_score > team2_score:
                    result = "W"
                elif team1_score == team2_score:
                    result = "D"
                else:
                    result = "L"
                
                h2h_dict = {
                    "match_id": match.id,
                    "match_date": match.match_date,
                    "team1_is_home": team1_is_home,
                    "team1_score": team1_score,
                    "team2_score": team2_score,
                    "result": result
                }
                h2h_data.append(h2h_dict)
            
            return pd.DataFrame(h2h_data)
        
        except Exception as e:
            logger.error(f"Error getting head-to-head: {str(e)}")
            raise
        
        finally:
            db.close()
    
    def get_team_stats(self, team_id: int, before_date: datetime, num_matches: int = 5) -> Dict[str, float]:
        """
        Get aggregated team statistics before a specific date.
        
        Args:
            team_id: Team ID
            before_date: Date to look back from
            num_matches: Number of recent matches to include
            
        Returns:
            Dictionary containing aggregated team statistics
        """
        logger.info(f"Getting stats for team {team_id} before {before_date}")
        
        db = SessionLocal()
        
        try:
            # Query recent matches for the team before the specified date
            matches = db.query(Match).filter(
                ((Match.home_team_id == team_id) | (Match.away_team_id == team_id)),
                Match.match_date < before_date,
                Match.status == "FT"  # Full Time (completed matches)
            ).order_by(Match.match_date.desc()).limit(num_matches).all()
            
            # Collect match statistics
            stats_list = []
            for match in matches:
                is_home = match.home_team_id == team_id
                
                # Get match statistics for the team
                match_stats = db.query(MatchStatistic).filter(
                    MatchStatistic.match_id == match.id,
                    MatchStatistic.team_id == team_id
                ).first()
                
                if match_stats:
                    stats_dict = {
                        "possession": match_stats.possession,
                        "shots_total": match_stats.shots_total,
                        "shots_on_target": match_stats.shots_on_target,
                        "shots_off_target": match_stats.shots_off_target,
                        "corners": match_stats.corners,
                        "fouls": match_stats.fouls,
                        "yellow_cards": match_stats.yellow_cards,
                        "red_cards": match_stats.red_cards,
                        "is_home": is_home
                    }
                    stats_list.append(stats_dict)
            
            # Convert to DataFrame
            if not stats_list:
                return {}
            
            stats_df = pd.DataFrame(stats_list)
            
            # Calculate aggregated statistics
            agg_stats = {
                "avg_possession": stats_df["possession"].mean() if "possession" in stats_df else None,
                "avg_shots_total": stats_df["shots_total"].mean() if "shots_total" in stats_df else None,
                "avg_shots_on_target": stats_df["shots_on_target"].mean() if "shots_on_target" in stats_df else None,
                "avg_corners": stats_df["corners"].mean() if "corners" in stats_df else None,
                "avg_fouls": stats_df["fouls"].mean() if "fouls" in stats_df else None,
                "avg_yellow_cards": stats_df["yellow_cards"].mean() if "yellow_cards" in stats_df else None,
                "avg_red_cards": stats_df["red_cards"].mean() if "red_cards" in stats_df else None
            }
            
            # Calculate home and away statistics separately
            home_stats = stats_df[stats_df["is_home"] == True]
            away_stats = stats_df[stats_df["is_home"] == False]
            
            if not home_stats.empty:
                agg_stats.update({
                    "home_avg_possession": home_stats["possession"].mean() if "possession" in home_stats else None,
                    "home_avg_shots_total": home_stats["shots_total"].mean() if "shots_total" in home_stats else None,
                    "home_avg_shots_on_target": home_stats["shots_on_target"].mean() if "shots_on_target" in home_stats else None,
                    "home_avg_corners": home_stats["corners"].mean() if "corners" in home_stats else None
                })
            
            if not away_stats.empty:
                agg_stats.update({
                    "away_avg_possession": away_stats["possession"].mean() if "possession" in away_stats else None,
                    "away_avg_shots_total": away_stats["shots_total"].mean() if "shots_total" in away_stats else None,
                    "away_avg_shots_on_target": away_stats["shots_on_target"].mean() if "shots_on_target" in away_stats else None,
                    "away_avg_corners": away_stats["corners"].mean() if "corners" in away_stats else None
                })
            
            return agg_stats
        
        except Exception as e:
            logger.error(f"Error getting team stats: {str(e)}")
            raise
        
        finally:
            db.close()
    
    def calculate_team_ratings(self, team_id: int, before_date: datetime, num_matches: int = 10) -> Dict[str, float]:
        """
        Calculate team ratings based on recent performance.
        
        Args:
            team_id: Team ID
            before_date: Date to look back from
            num_matches: Number of recent matches to include
            
        Returns:
            Dictionary containing team ratings
        """
        logger.info(f"Calculating ratings for team {team_id} before {before_date}")
        
        db = SessionLocal()
        
        try:
            # Query recent matches for the team before the specified date
            matches = db.query(Match).filter(
                ((Match.home_team_id == team_id) | (Match.away_team_id == team_id)),
                Match.match_date < before_date,
                Match.status == "FT"  # Full Time (completed matches)
            ).order_by(Match.match_date.desc()).limit(num_matches).all()
            
            if not matches:
                return {}
            
            # Calculate attack and defense ratings
            goals_for = []
            goals_against = []
            
            for match in matches:
                is_home = match.home_team_id == team_id
                
                if is_home:
                    goals_for.append(match.home_score if match.home_score is not None else 0)
                    goals_against.append(match.away_score if match.away_score is not None else 0)
                else:
                    goals_for.append(match.away_score if match.away_score is not None else 0)
                    goals_against.append(match.home_score if match.home_score is not None else 0)
            
            # Calculate ratings
            attack_rating = sum(goals_for) / len(goals_for) if goals_for else 0
            defense_rating = sum(goals_against) / len(goals_against) if goals_against else 0
            
            # Calculate form points (3 for win, 1 for draw, 0 for loss)
            form_points = 0
            for i, match in enumerate(matches):
                is_home = match.home_team_id == team_id
                team_score = match.home_score if is_home else match.away_score
                opponent_score = match.away_score if is_home else match.home_score
                
                if team_score > opponent_score:
                    form_points += 3
                elif team_score == opponent_score:
                    form_points += 1
            
            form_rating = form_points / (len(matches) * 3) if matches else 0
            
            # Calculate home/away advantage
            home_matches = [m for m in matches if m.home_team_id == team_id]
            away_matches = [m for m in matches if m.away_team_id == team_id]
            
            home_points = 0
            for match in home_matches:
                if match.home_score > match.away_score:
                    home_points += 3
                elif match.home_score == match.away_score:
                    home_points += 1
            
            away_points = 0
            for match in away_matches:
                if match.away_score > match.home_score:
                    away_points += 3
                elif match.away_score == match.home_score:
                    away_points += 1
            
            home_advantage = home_points / (len(home_matches) * 3) if home_matches else 0
            away_disadvantage = away_points / (len(away_matches) * 3) if away_matches else 0
            
            return {
                "attack_rating": attack_rating,
                "defense_rating": defense_rating,
                "form_rating": form_rating,
                "home_advantage": home_advantage,
                "away_disadvantage": away_disadvantage,
                "overall_rating": (attack_rating - defense_rating + form_rating) / 3
            }
        
        except Exception as e:
            logger.error(f"Error calculating team ratings: {str(e)}")
            raise
        
        finally:
            db.close()
    
    def prepare_match_features(self, match_id: int) -> Dict[str, Any]:
        """
        Prepare features for a specific match.
        
        Args:
            match_id: Match ID
            
        Returns:
            Dictionary containing match features
        """
        logger.info(f"Preparing features for match {match_id}")
        
        db = SessionLocal()
        
        try:
            # Get match from database
            match = db.query(Match).filter(Match.id == match_id).first()
            if not match:
                logger.error(f"Match with ID {match_id} not found in database")
                return {}
            
            # Get team form
            home_form = self.get_team_form(match.home_team_id, match.match_date)
            away_form = self.get_team_form(match.away_team_id, match.match_date)
            
            # Get head-to-head history
            h2h = self.get_head_to_head(match.home_team_id, match.away_team_id, match.match_date)
            
            # Get team statistics
            home_stats = self.get_team_stats(match.home_team_id, match.match_date)
            away_stats = self.get_team_stats(match.away_team_id, match.match_date)
            
            # Get team ratings
            home_ratings = self.calculate_team_ratings(match.home_team_id, match.match_date)
            away_ratings = self.calculate_team_ratings(match.away_team_id, match.match_date)
            
            # Calculate form metrics
            home_form_metrics = self._calculate_form_metrics(home_form)
            away_form_metrics = self._calculate_form_metrics(away_form)
            
            # Calculate head-to-head metrics
            h2h_metrics = self._calculate_h2h_metrics(h2h, match.home_team_id)
            
            # Combine all features
            features = {
                "match_id": match.id,
                "league_id": match.league_id,
                "home_team_id": match.home_team_id,
                "away_team_id": match.away_team_id,
                "match_date": match.match_date,
                
                # Home team form metrics
                "home_team_recent_wins": home_form_metrics.get("wins", 0),
                "home_team_recent_draws": home_form_metrics.get("draws", 0),
                "home_team_recent_losses": home_form_metrics.get("losses", 0),
                "home_team_recent_goals_scored": home_form_metrics.get("goals_scored", 0),
                "home_team_recent_goals_conceded": home_form_metrics.get("goals_conceded", 0),
                "home_team_form_points": home_form_metrics.get("form_points", 0),
                
                # Away team form metrics
                "away_team_recent_wins": away_form_metrics.get("wins", 0),
                "away_team_recent_draws": away_form_metrics.get("draws", 0),
                "away_team_recent_losses": away_form_metrics.get("losses", 0),
                "away_team_recent_goals_scored": away_form_metrics.get("goals_scored", 0),
                "away_team_recent_goals_conceded": away_form_metrics.get("goals_conceded", 0),
                "away_team_form_points": away_form_metrics.get("form_points", 0),
                
                # Head-to-head metrics
                "h2h_home_team_wins": h2h_metrics.get("home_team_wins", 0),
                "h2h_draws": h2h_metrics.get("draws", 0),
                "h2h_away_team_wins": h2h_metrics.get("away_team_wins", 0),
                "h2h_home_team_goals": h2h_metrics.get("home_team_goals", 0),
                "h2h_away_team_goals": h2h_metrics.get("away_team_goals", 0),
                
                # Home team statistics
                "home_team_avg_possession": home_stats.get("avg_possession", 0),
                "home_team_avg_shots_total": home_stats.get("avg_shots_total", 0),
                "home_team_avg_shots_on_target": home_stats.get("avg_shots_on_target", 0),
                "home_team_avg_corners": home_stats.get("avg_corners", 0),
                
                # Away team statistics
                "away_team_avg_possession": away_stats.get("avg_possession", 0),
                "away_team_avg_shots_total": away_stats.get("avg_shots_total", 0),
                "away_team_avg_shots_on_target": away_stats.get("avg_shots_on_target", 0),
                "away_team_avg_corners": away_stats.get("avg_corners", 0),
                
                # Home team ratings
                "home_team_attack_rating": home_ratings.get("attack_rating", 0),
                "home_team_defense_rating": home_ratings.get("defense_rating", 0),
                "home_team_form_rating": home_ratings.get("form_rating", 0),
                "home_team_home_advantage": home_ratings.get("home_advantage", 0),
                "home_team_overall_rating": home_ratings.get("overall_rating", 0),
                
                # Away team ratings
                "away_team_attack_rating": away_ratings.get("attack_rating", 0),
                "away_team_defense_rating": away_ratings.get("defense_rating", 0),
                "away_team_form_rating": away_ratings.get("form_rating", 0),
                "away_team_away_disadvantage": away_ratings.get("away_disadvantage", 0),
                "away_team_overall_rating": away_ratings.get("overall_rating", 0),
                
                # Weather features (if available)
                "temperature": match.temperature,
                
                # Target variables (for training only)
                "home_score": match.home_score,
                "away_score": match.away_score,
                "result": self._get_match_result(match)
            }
            
            return features
        
        except Exception as e:
            logger.error(f"Error preparing match features: {str(e)}")
            raise
        
        finally:
            db.close()
    
    def prepare_training_dataset(self, days_back: int = 365) -> pd.DataFrame:
        """
        Prepare a training dataset from historical matches.
        
        Args:
            days_back: Number of days to look back for historical matches
            
        Returns:
            DataFrame containing training data
        """
        logger.info(f"Preparing training dataset from the past {days_back} days")
        
        # Get historical matches
        matches_df = self.get_historical_matches(days_back)
        
        if matches_df.empty:
            logger.warning("No historical matches found")
            return pd.DataFrame()
        
        # Prepare features for each match
        features_list = []
        for _, row in matches_df.iterrows():
            try:
                features = self.prepare_match_features(row["match_id"])
                if features:
                    features_list.append(features)
            except Exception as e:
                logger.error(f"Error preparing features for match {row['match_id']}: {str(e)}")
        
        if not features_list:
            logger.warning("No features could be prepared")
            return pd.DataFrame()
        
        # Convert to DataFrame
        features_df = pd.DataFrame(features_list)
        
        # Handle missing values
        features_df = features_df.fillna(0)
        
        logger.info(f"Prepared training dataset with {len(features_df)} samples and {len(features_df.columns)} features")
        return features_df
    
    def _calculate_form_metrics(self, form_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate form metrics from team form data.
        
        Args:
            form_df: DataFrame containing team form data
            
        Returns:
            Dictionary containing form metrics
        """
        if form_df.empty:
            return {
                "wins": 0,
                "draws": 0,
                "losses": 0,
                "goals_scored": 0,
                "goals_conceded": 0,
                "form_points": 0
            }
        
        wins = sum(form_df["result"] == "W")
        draws = sum(form_df["result"] == "D")
        losses = sum(form_df["result"] == "L")
        
        goals_scored = form_df["team_score"].sum()
        goals_conceded = form_df["opponent_score"].sum()
        
        # Calculate form points (3 for win, 1 for draw, 0 for loss)
        form_points = wins * 3 + draws
        
        return {
            "wins": wins,
            "draws": draws,
            "losses": losses,
            "goals_scored": goals_scored,
            "goals_conceded": goals_conceded,
            "form_points": form_points
        }
    
    def _calculate_h2h_metrics(self, h2h_df: pd.DataFrame, home_team_id: int) -> Dict[str, Any]:
        """
        Calculate head-to-head metrics.
        
        Args:
            h2h_df: DataFrame containing head-to-head data
            home_team_id: ID of the home team
            
        Returns:
            Dictionary containing head-to-head metrics
        """
        if h2h_df.empty:
            return {
                "home_team_wins": 0,
                "draws": 0,
                "away_team_wins": 0,
                "home_team_goals": 0,
                "away_team_goals": 0
            }
        
        # Count results
        home_team_wins = sum(h2h_df["result"] == "W")
        draws = sum(h2h_df["result"] == "D")
        away_team_wins = sum(h2h_df["result"] == "L")
        
        # Sum goals
        home_team_goals = h2h_df["team1_score"].sum()
        away_team_goals = h2h_df["team2_score"].sum()
        
        return {
            "home_team_wins": home_team_wins,
            "draws": draws,
            "away_team_wins": away_team_wins,
            "home_team_goals": home_team_goals,
            "away_team_goals": away_team_goals
        }
    
    def _get_match_result(self, match: Match) -> str:
        """
        Get the result of a match (home win, draw, away win).
        
        Args:
            match: Match object
            
        Returns:
            Match result as a string: "H" (home win), "D" (draw), or "A" (away win)
        """
        if match.home_score is None or match.away_score is None:
            return None
        
        if match.home_score > match.away_score:
            return "H"
        elif match.home_score == match.away_score:
            return "D"
        else:
            return "A"
