"""
Data fetching module for the Soccer Match Prediction Application.
This module provides functions to fetch data from API-Football and store it in the database.
"""

import os
import sys
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from sqlalchemy.orm import Session

# Add the project root directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from api.api_client import APIFootballClient
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
        logging.FileHandler("data_fetcher.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("data_fetcher")

class DataFetcher:
    """Class for fetching data from API-Football and storing it in the database."""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the data fetcher.
        
        Args:
            api_key: API key for API-Football. If not provided, will try to get from environment variable.
        """
        self.api_client = APIFootballClient(api_key)
        
    def fetch_leagues(self, country: Optional[str] = None, season: int = 2023) -> List[League]:
        """
        Fetch leagues from API-Football and store them in the database.
        
        Args:
            country: Filter by country name
            season: Season year
            
        Returns:
            List of League objects
        """
        logger.info(f"Fetching leagues for season {season}")
        
        leagues_data = self.api_client.get_leagues(country, season)
        db = SessionLocal()
        
        try:
            leagues = []
            for league_data in leagues_data:
                league_info = league_data.get("league", {})
                country_info = league_data.get("country", {})
                
                # Check if league already exists in database
                existing_league = db.query(League).filter(
                    League.api_id == league_info.get("id"),
                    League.season == str(season)
                ).first()
                
                if existing_league:
                    leagues.append(existing_league)
                    continue
                
                # Create new league
                league = League(
                    api_id=league_info.get("id"),
                    name=league_info.get("name"),
                    country=country_info.get("name"),
                    logo_url=league_info.get("logo"),
                    season=str(season)
                )
                
                db.add(league)
                leagues.append(league)
            
            db.commit()
            logger.info(f"Fetched and stored {len(leagues)} leagues")
            return leagues
        
        except Exception as e:
            db.rollback()
            logger.error(f"Error fetching leagues: {str(e)}")
            raise
        
        finally:
            db.close()
    
    def fetch_teams(self, league_id: int, season: int = 2023) -> List[Team]:
        """
        Fetch teams for a specific league and store them in the database.
        
        Args:
            league_id: League ID
            season: Season year
            
        Returns:
            List of Team objects
        """
        logger.info(f"Fetching teams for league {league_id}, season {season}")
        
        teams_data = self.api_client.get_teams(league_id, season)
        db = SessionLocal()
        
        try:
            teams = []
            for team_data in teams_data:
                team_info = team_data.get("team", {})
                venue_info = team_data.get("venue", {})
                
                # Check if team already exists in database
                existing_team = db.query(Team).filter(
                    Team.api_id == team_info.get("id")
                ).first()
                
                if existing_team:
                    teams.append(existing_team)
                    continue
                
                # Create new team
                team = Team(
                    api_id=team_info.get("id"),
                    name=team_info.get("name"),
                    country=team_info.get("country"),
                    logo_url=team_info.get("logo"),
                    founded=team_info.get("founded"),
                    venue_name=venue_info.get("name"),
                    venue_capacity=venue_info.get("capacity")
                )
                
                db.add(team)
                teams.append(team)
            
            db.commit()
            logger.info(f"Fetched and stored {len(teams)} teams")
            return teams
        
        except Exception as e:
            db.rollback()
            logger.error(f"Error fetching teams: {str(e)}")
            raise
        
        finally:
            db.close()
    
    def fetch_players(self, team_id: int, season: int = 2023) -> List[Player]:
        """
        Fetch players for a specific team and store them in the database.
        
        Args:
            team_id: Team ID
            season: Season year
            
        Returns:
            List of Player objects
        """
        logger.info(f"Fetching players for team {team_id}, season {season}")
        
        players_data = self.api_client.get_players(team_id, season)
        db = SessionLocal()
        
        try:
            # Get team from database
            team = db.query(Team).filter(Team.api_id == team_id).first()
            if not team:
                logger.error(f"Team with API ID {team_id} not found in database")
                return []
            
            players = []
            for player_data in players_data:
                player_info = player_data.get("player", {})
                
                # Check if player already exists in database
                existing_player = db.query(Player).filter(
                    Player.api_id == player_info.get("id")
                ).first()
                
                if existing_player:
                    # Update team if needed
                    if existing_player.team_id != team.id:
                        existing_player.team_id = team.id
                        db.add(existing_player)
                    
                    players.append(existing_player)
                    continue
                
                # Create new player
                player = Player(
                    api_id=player_info.get("id"),
                    name=player_info.get("name"),
                    position=player_info.get("position"),
                    nationality=player_info.get("nationality"),
                    age=player_info.get("age"),
                    height=player_info.get("height"),
                    weight=player_info.get("weight"),
                    team_id=team.id
                )
                
                db.add(player)
                players.append(player)
            
            db.commit()
            logger.info(f"Fetched and stored {len(players)} players")
            return players
        
        except Exception as e:
            db.rollback()
            logger.error(f"Error fetching players: {str(e)}")
            raise
        
        finally:
            db.close()
    
    def fetch_fixtures(self, 
                      league_id: Optional[int] = None,
                      team_id: Optional[int] = None,
                      date_from: Optional[str] = None,
                      date_to: Optional[str] = None,
                      season: int = 2023) -> List[Match]:
        """
        Fetch fixtures (matches) and store them in the database.
        
        Args:
            league_id: Filter by league ID
            team_id: Filter by team ID
            date_from: Start date (format: YYYY-MM-DD)
            date_to: End date (format: YYYY-MM-DD)
            season: Season year
            
        Returns:
            List of Match objects
        """
        logger.info(f"Fetching fixtures for season {season}")
        
        # Set default date range to last 30 days if not provided
        if not date_from:
            date_from = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
        if not date_to:
            date_to = datetime.now().strftime("%Y-%m-%d")
        
        fixtures_data = self.api_client.get_fixtures(
            league_id=league_id,
            team_id=team_id,
            season=season,
            from_date=date_from,
            to_date=date_to
        )
        
        db = SessionLocal()
        
        try:
            matches = []
            for fixture_data in fixtures_data:
                fixture_info = fixture_data.get("fixture", {})
                league_info = fixture_data.get("league", {})
                teams_info = fixture_data.get("teams", {})
                goals_info = fixture_data.get("goals", {})
                score_info = fixture_data.get("score", {})
                
                # Get or create league
                league = db.query(League).filter(
                    League.api_id == league_info.get("id"),
                    League.season == str(season)
                ).first()
                
                if not league:
                    league = League(
                        api_id=league_info.get("id"),
                        name=league_info.get("name"),
                        country=league_info.get("country"),
                        logo_url=league_info.get("logo"),
                        season=str(season)
                    )
                    db.add(league)
                    db.flush()
                
                # Get or create home team
                home_team_info = teams_info.get("home", {})
                home_team = db.query(Team).filter(
                    Team.api_id == home_team_info.get("id")
                ).first()
                
                if not home_team:
                    home_team = Team(
                        api_id=home_team_info.get("id"),
                        name=home_team_info.get("name"),
                        logo_url=home_team_info.get("logo")
                    )
                    db.add(home_team)
                    db.flush()
                
                # Get or create away team
                away_team_info = teams_info.get("away", {})
                away_team = db.query(Team).filter(
                    Team.api_id == away_team_info.get("id")
                ).first()
                
                if not away_team:
                    away_team = Team(
                        api_id=away_team_info.get("id"),
                        name=away_team_info.get("name"),
                        logo_url=away_team_info.get("logo")
                    )
                    db.add(away_team)
                    db.flush()
                
                # Check if match already exists in database
                existing_match = db.query(Match).filter(
                    Match.api_id == fixture_info.get("id")
                ).first()
                
                if existing_match:
                    # Update match status and scores
                    existing_match.status = fixture_info.get("status", {}).get("short")
                    existing_match.home_score = goals_info.get("home")
                    existing_match.away_score = goals_info.get("away")
                    existing_match.home_halftime_score = score_info.get("halftime", {}).get("home")
                    existing_match.away_halftime_score = score_info.get("halftime", {}).get("away")
                    
                    db.add(existing_match)
                    matches.append(existing_match)
                    continue
                
                # Create new match
                match_date = datetime.fromtimestamp(fixture_info.get("timestamp", 0))
                match = Match(
                    api_id=fixture_info.get("id"),
                    league_id=league.id,
                    home_team_id=home_team.id,
                    away_team_id=away_team.id,
                    match_date=match_date,
                    status=fixture_info.get("status", {}).get("short"),
                    home_score=goals_info.get("home"),
                    away_score=goals_info.get("away"),
                    home_halftime_score=score_info.get("halftime", {}).get("home"),
                    away_halftime_score=score_info.get("halftime", {}).get("away"),
                    venue=fixture_info.get("venue", {}).get("name"),
                    referee=fixture_info.get("referee")
                )
                
                db.add(match)
                matches.append(match)
            
            db.commit()
            logger.info(f"Fetched and stored {len(matches)} matches")
            return matches
        
        except Exception as e:
            db.rollback()
            logger.error(f"Error fetching fixtures: {str(e)}")
            raise
        
        finally:
            db.close()
    
    def fetch_match_statistics(self, match_id: int) -> List[MatchStatistic]:
        """
        Fetch statistics for a specific match and store them in the database.
        
        Args:
            match_id: Match ID in the database
            
        Returns:
            List of MatchStatistic objects
        """
        db = SessionLocal()
        
        try:
            # Get match from database
            match = db.query(Match).filter(Match.id == match_id).first()
            if not match:
                logger.error(f"Match with ID {match_id} not found in database")
                return []
            
            logger.info(f"Fetching statistics for match {match.api_id}")
            
            statistics_data = self.api_client.get_fixture_statistics(match.api_id)
            
            match_statistics = []
            for team_stats in statistics_data:
                team_id = team_stats.get("team", {}).get("id")
                
                # Get team from database
                team = db.query(Team).filter(Team.api_id == team_id).first()
                if not team:
                    logger.warning(f"Team with API ID {team_id} not found in database")
                    continue
                
                # Check if statistics already exist
                existing_stats = db.query(MatchStatistic).filter(
                    MatchStatistic.match_id == match.id,
                    MatchStatistic.team_id == team.id
                ).first()
                
                if existing_stats:
                    match_statistics.append(existing_stats)
                    continue
                
                # Extract statistics
                stats = team_stats.get("statistics", [])
                stats_dict = {stat.get("type"): stat.get("value") for stat in stats}
                
                # Convert percentage strings to floats
                possession = stats_dict.get("Ball Possession")
                if possession and isinstance(possession, str) and "%" in possession:
                    possession = float(possession.replace("%", ""))
                
                # Create new match statistics
                match_stat = MatchStatistic(
                    match_id=match.id,
                    team_id=team.id,
                    possession=possession,
                    shots_total=stats_dict.get("Total Shots"),
                    shots_on_target=stats_dict.get("Shots on Goal"),
                    shots_off_target=stats_dict.get("Shots off Goal"),
                    corners=stats_dict.get("Corner Kicks"),
                    free_kicks=stats_dict.get("Free Kicks"),
                    throw_ins=stats_dict.get("Throw-in"),
                    fouls=stats_dict.get("Fouls"),
                    yellow_cards=stats_dict.get("Yellow Cards"),
                    red_cards=stats_dict.get("Red Cards"),
                    offsides=stats_dict.get("Offsides"),
                    saves=stats_dict.get("Goalkeeper Saves")
                )
                
                db.add(match_stat)
                match_statistics.append(match_stat)
            
            db.commit()
            logger.info(f"Fetched and stored statistics for {len(match_statistics)} teams")
            return match_statistics
        
        except Exception as e:
            db.rollback()
            logger.error(f"Error fetching match statistics: {str(e)}")
            raise
        
        finally:
            db.close()
    
    def fetch_betting_odds(self, match_id: int) -> List[BettingOdds]:
        """
        Fetch betting odds for a specific match and store them in the database.
        
        Args:
            match_id: Match ID in the database
            
        Returns:
            List of BettingOdds objects
        """
        db = SessionLocal()
        
        try:
            # Get match from database
            match = db.query(Match).filter(Match.id == match_id).first()
            if not match:
                logger.error(f"Match with ID {match_id} not found in database")
                return []
            
            logger.info(f"Fetching betting odds for match {match.api_id}")
            
            odds_data = self.api_client.get_odds(match.api_id)
            
            betting_odds_list = []
            for bookmaker_data in odds_data:
                bookmaker_info = bookmaker_data.get("bookmaker", {})
                bookmaker_name = bookmaker_info.get("name")
                
                for bet_data in bookmaker_data.get("bets", []):
                    bet_name = bet_data.get("name")
                    values = bet_data.get("values", [])
                    
                    # Process different types of odds
                    if bet_name == "Match Winner":  # 1X2
                        home_win_odds = None
                        draw_odds = None
                        away_win_odds = None
                        
                        for value in values:
                            if value.get("value") == "Home":
                                home_win_odds = float(value.get("odd"))
                            elif value.get("value") == "Draw":
                                draw_odds = float(value.get("odd"))
                            elif value.get("value") == "Away":
                                away_win_odds = float(value.get("odd"))
                        
                        # Check if odds already exist
                        existing_odds = db.query(BettingOdds).filter(
                            BettingOdds.match_id == match.id,
                            BettingOdds.bookmaker == bookmaker_name
                        ).first()
                        
                        if existing_odds:
                            existing_odds.home_win_odds = home_win_odds
                            existing_odds.draw_odds = draw_odds
                            existing_odds.away_win_odds = away_win_odds
                            existing_odds.timestamp = datetime.utcnow()
                            
                            db.add(existing_odds)
                            betting_odds_list.append(existing_odds)
                        else:
                            # Create new betting odds
                            betting_odds = BettingOdds(
                                match_id=match.id,
                                bookmaker=bookmaker_name,
                                home_win_odds=home_win_odds,
                                draw_odds=draw_odds,
                                away_win_odds=away_win_odds,
                                timestamp=datetime.utcnow()
                            )
                            
                            db.add(betting_odds)
                            betting_odds_list.append(betting_odds)
                    
                    elif bet_name == "Goals Over/Under":  # Over/Under 2.5
                        for value in values:
                            if value.get("value") == "Over 2.5":
                                over_odds = float(value.get("odd"))
                                
                                # Find existing odds
                                existing_odds = db.query(BettingOdds).filter(
                                    BettingOdds.match_id == match.id,
                                    BettingOdds.bookmaker == bookmaker_name
                                ).first()
                                
                                if existing_odds:
                                    existing_odds.over_2_5_odds = over_odds
                                    db.add(existing_odds)
                                else:
                                    # Create new betting odds if not found
                                    betting_odds = BettingOdds(
                                        match_id=match.id,
                                        bookmaker=bookmaker_name,
                                        over_2_5_odds=over_odds,
                                        timestamp=datetime.utcnow()
                                    )
                                    
                                    db.add(betting_odds)
                                    betting_odds_list.append(betting_odds)
                            
                            elif value.get("value") == "Under 2.5":
                                under_odds = float(value.get("odd"))
                                
                                # Find existing odds
                                existing_odds = db.query(BettingOdds).filter(
                                    BettingOdds.match_id == match.id,
                                    BettingOdds.bookmaker == bookmaker_name
                                ).first()
                                
                                if existing_odds:
                                    existing_odds.under_2_5_odds = under_odds
                                    db.add(existing_odds)
                                else:
                                    # Create new betting odds if not found
                                    betting_odds = BettingOdds(
                                        match_id=match.id,
                                        bookmaker=bookmaker_name,
                                        under_2_5_odds=under_odds,
                                        timestamp=datetime.utcnow()
                                    )
                                    
                                    db.add(betting_odds)
                                    betting_odds_list.append(betting_odds)
                    
                    elif bet_name == "Both Teams Score":  # BTTS
                        for value in values:
                            if value.get("value") == "Yes":
                                btts_yes_odds = float(value.get("odd"))
                                
                                # Find existing odds
                                existing_odds = db.query(BettingOdds).filter(
                                    BettingOdds.match_id == match.id,
                                    BettingOdds.bookmaker == bookmaker_name
                                ).first()
                                
                                if existing_odds:
                                    existing_odds.btts_yes_odds = btts_yes_odds
                                    db.add(existing_odds)
                                else:
                                    # Create new betting odds if not found
                                    betting_odds = BettingOdds(
                                        match_id=match.id,
                                        bookmaker=bookmaker_name,
                                        btts_yes_odds=btts_yes_odds,
                                        timestamp=datetime.utcnow()
                                    )
                                    
                                    db.add(betting_odds)
                                    betting_odds_list.append(betting_odds)
                            
                            elif value.get("value") == "No":
                                btts_no_odds = float(value.get("odd"))
                                
                                # Find existing odds
                                existing_odds = db.query(BettingOdds).filter(
                                    BettingOdds.match_id == match.id,
                                    BettingOdds.bookmaker == bookmaker_name
                                ).first()
                                
                                if existing_odds:
                                    existing_odds.btts_no_odds = btts_no_odds
                                    db.add(existing_odds)
                                else:
                                    # Create new betting odds if not found
                                    betting_odds = BettingOdds(
                                        match_id=match.id,
                                        bookmaker=bookmaker_name,
                                        btts_no_odds=btts_no_odds,
                                        timestamp=datetime.utcnow()
                                    )
                                    
                                    db.add(betting_odds)
                                    betting_odds_list.append(betting_odds)
            
            db.commit()
            logger.info(f"Fetched and stored betting odds for match {match_id}")
            return betting_odds_list
        
        except Exception as e:
            db.rollback()
            logger.error(f"Error fetching betting odds: {str(e)}")
            raise
        
        finally:
            db.close()
    
    def fetch_api_predictions(self, match_id: int) -> Optional[Prediction]:
        """
        Fetch predictions from API-Football for a specific match and store them in the database.
        
        Args:
            match_id: Match ID in the database
            
        Returns:
            Prediction object if successful, None otherwise
        """
        db = SessionLocal()
        
        try:
            # Get match from database
            match = db.query(Match).filter(Match.id == match_id).first()
            if not match:
                logger.error(f"Match with ID {match_id} not found in database")
                return None
            
            logger.info(f"Fetching predictions for match {match.api_id}")
            
            prediction_data = self.api_client.get_predictions(match.api_id)
            if not prediction_data:
                logger.warning(f"No predictions available for match {match.api_id}")
                return None
            
            # Extract prediction information
            predictions = prediction_data.get("predictions", {})
            
            # Check if prediction already exists
            existing_prediction = db.query(Prediction).filter(
                Prediction.match_id == match.id,
                Prediction.model_version == "api-football-v1"
            ).first()
            
            if existing_prediction:
                # Update existing prediction
                existing_prediction.home_win_probability = predictions.get("percent", {}).get("home") / 100 if predictions.get("percent", {}).get("home") else None
                existing_prediction.draw_probability = predictions.get("percent", {}).get("draw") / 100 if predictions.get("percent", {}).get("draw") else None
                existing_prediction.away_win_probability = predictions.get("percent", {}).get("away") / 100 if predictions.get("percent", {}).get("away") else None
                
                # Extract predicted scores if available
                winner = predictions.get("winner", {}).get("name")
                if winner == match.home_team.name:
                    existing_prediction.predicted_home_score = 1
                    existing_prediction.predicted_away_score = 0
                elif winner == match.away_team.name:
                    existing_prediction.predicted_home_score = 0
                    existing_prediction.predicted_away_score = 1
                else:
                    existing_prediction.predicted_home_score = 0
                    existing_prediction.predicted_away_score = 0
                
                # Set confidence based on highest probability
                probabilities = [
                    existing_prediction.home_win_probability or 0,
                    existing_prediction.draw_probability or 0,
                    existing_prediction.away_win_probability or 0
                ]
                existing_prediction.confidence = max(probabilities) if probabilities else 0.5
                
                db.add(existing_prediction)
                db.commit()
                
                logger.info(f"Updated prediction for match {match_id}")
                return existing_prediction
            else:
                # Create new prediction
                home_win_prob = predictions.get("percent", {}).get("home") / 100 if predictions.get("percent", {}).get("home") else None
                draw_prob = predictions.get("percent", {}).get("draw") / 100 if predictions.get("percent", {}).get("draw") else None
                away_win_prob = predictions.get("percent", {}).get("away") / 100 if predictions.get("percent", {}).get("away") else None
                
                # Extract predicted scores if available
                predicted_home_score = 0
                predicted_away_score = 0
                winner = predictions.get("winner", {}).get("name")
                if winner == match.home_team.name:
                    predicted_home_score = 1
                    predicted_away_score = 0
                elif winner == match.away_team.name:
                    predicted_home_score = 0
                    predicted_away_score = 1
                
                # Set confidence based on highest probability
                probabilities = [
                    home_win_prob or 0,
                    draw_prob or 0,
                    away_win_prob or 0
                ]
                confidence = max(probabilities) if probabilities else 0.5
                
                prediction = Prediction(
                    match_id=match.id,
                    predicted_home_score=predicted_home_score,
                    predicted_away_score=predicted_away_score,
                    home_win_probability=home_win_prob,
                    draw_probability=draw_prob,
                    away_win_probability=away_win_prob,
                    confidence=confidence,
                    model_version="api-football-v1"
                )
                
                db.add(prediction)
                db.commit()
                
                logger.info(f"Created prediction for match {match_id}")
                return prediction
        
        except Exception as e:
            db.rollback()
            logger.error(f"Error fetching predictions: {str(e)}")
            raise
        
        finally:
            db.close()
            
    def fetch_upcoming_matches(self, days: int = 7) -> List[Match]:
        """
        Fetch upcoming matches for the next specified number of days.
        
        Args:
            days: Number of days to look ahead
            
        Returns:
            List of Match objects
        """
        date_from = datetime.now().strftime("%Y-%m-%d")
        date_to = (datetime.now() + timedelta(days=days)).strftime("%Y-%m-%d")
        
        return self.fetch_fixtures(date_from=date_from, date_to=date_to)
    
    def fetch_all_data_for_league(self, league_id: int, season: int = 2023):
        """
        Fetch all data for a specific league including teams, players, fixtures, and odds.
        
        Args:
            league_id: League ID from API-Football
            season: Season year
        """
        logger.info(f"Fetching all data for league {league_id}, season {season}")
        
        # Fetch league
        leagues = self.fetch_leagues(season=season)
        league = next((l for l in leagues if l.api_id == league_id), None)
        
        if not league:
            logger.error(f"League with API ID {league_id} not found")
            return
        
        # Fetch teams
        teams = self.fetch_teams(league_id, season)
        
        # Fetch players for each team
        for team in teams:
            self.fetch_players(team.api_id, season)
        
        # Fetch fixtures
        matches = self.fetch_fixtures(league_id=league_id, season=season)
        
        # Fetch statistics and odds for each match
        for match in matches:
            self.fetch_match_statistics(match.id)
            self.fetch_betting_odds(match.id)
            self.fetch_api_predictions(match.id)
        
        logger.info(f"Completed fetching all data for league {league_id}, season {season}")
