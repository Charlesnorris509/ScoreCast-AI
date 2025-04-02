"""
Database models for the Soccer Match Prediction Application.
This module defines the SQLAlchemy ORM models for storing soccer data and predictions.
"""

from datetime import datetime
from sqlalchemy import Column, Integer, String, Float, DateTime, Boolean, ForeignKey, Table
from sqlalchemy.orm import relationship
from sqlalchemy.ext.declarative import declarative_base

from config.database import Base

# Association tables for many-to-many relationships
match_players = Table(
    'match_players',
    Base.metadata,
    Column('match_id', Integer, ForeignKey('matches.id'), primary_key=True),
    Column('player_id', Integer, ForeignKey('players.id'), primary_key=True)
)


class Team(Base):
    """Team model for storing team information."""
    
    __tablename__ = "teams"
    
    id = Column(Integer, primary_key=True, index=True)
    api_id = Column(Integer, unique=True, index=True)
    name = Column(String, index=True)
    country = Column(String)
    logo_url = Column(String)
    founded = Column(Integer, nullable=True)
    venue_name = Column(String, nullable=True)
    venue_capacity = Column(Integer, nullable=True)
    
    # Relationships
    home_matches = relationship("Match", foreign_keys="Match.home_team_id", back_populates="home_team")
    away_matches = relationship("Match", foreign_keys="Match.away_team_id", back_populates="away_team")
    players = relationship("Player", back_populates="team")
    
    def __repr__(self):
        return f"<Team {self.name}>"


class Player(Base):
    """Player model for storing player information."""
    
    __tablename__ = "players"
    
    id = Column(Integer, primary_key=True, index=True)
    api_id = Column(Integer, unique=True, index=True)
    name = Column(String, index=True)
    position = Column(String, nullable=True)
    nationality = Column(String, nullable=True)
    age = Column(Integer, nullable=True)
    height = Column(Integer, nullable=True)  # in cm
    weight = Column(Integer, nullable=True)  # in kg
    team_id = Column(Integer, ForeignKey("teams.id"))
    
    # Relationships
    team = relationship("Team", back_populates="players")
    matches = relationship("Match", secondary=match_players, back_populates="players")
    statistics = relationship("PlayerStatistic", back_populates="player")
    
    def __repr__(self):
        return f"<Player {self.name}>"


class League(Base):
    """League model for storing league information."""
    
    __tablename__ = "leagues"
    
    id = Column(Integer, primary_key=True, index=True)
    api_id = Column(Integer, unique=True, index=True)
    name = Column(String, index=True)
    country = Column(String)
    logo_url = Column(String, nullable=True)
    season = Column(String)
    
    # Relationships
    matches = relationship("Match", back_populates="league")
    
    def __repr__(self):
        return f"<League {self.name} ({self.season})>"


class Match(Base):
    """Match model for storing match information."""
    
    __tablename__ = "matches"
    
    id = Column(Integer, primary_key=True, index=True)
    api_id = Column(Integer, unique=True, index=True)
    league_id = Column(Integer, ForeignKey("leagues.id"))
    home_team_id = Column(Integer, ForeignKey("teams.id"))
    away_team_id = Column(Integer, ForeignKey("teams.id"))
    match_date = Column(DateTime, index=True)
    status = Column(String)  # NS (Not Started), LIVE, FT (Full Time), etc.
    home_score = Column(Integer, nullable=True)
    away_score = Column(Integer, nullable=True)
    home_halftime_score = Column(Integer, nullable=True)
    away_halftime_score = Column(Integer, nullable=True)
    venue = Column(String, nullable=True)
    referee = Column(String, nullable=True)
    round = Column(String, nullable=True)
    
    # Weather information
    weather_code = Column(String, nullable=True)
    weather_description = Column(String, nullable=True)
    temperature = Column(Float, nullable=True)
    humidity = Column(Integer, nullable=True)
    wind_speed = Column(Float, nullable=True)
    
    # Relationships
    league = relationship("League", back_populates="matches")
    home_team = relationship("Team", foreign_keys=[home_team_id], back_populates="home_matches")
    away_team = relationship("Team", foreign_keys=[away_team_id], back_populates="away_matches")
    players = relationship("Player", secondary=match_players, back_populates="matches")
    match_statistics = relationship("MatchStatistic", back_populates="match")
    player_statistics = relationship("PlayerStatistic", back_populates="match")
    betting_odds = relationship("BettingOdds", back_populates="match")
    predictions = relationship("Prediction", back_populates="match")
    
    def __repr__(self):
        return f"<Match {self.home_team.name} vs {self.away_team.name} on {self.match_date}>"


class MatchStatistic(Base):
    """Match statistics model for storing match-level statistics."""
    
    __tablename__ = "match_statistics"
    
    id = Column(Integer, primary_key=True, index=True)
    match_id = Column(Integer, ForeignKey("matches.id"))
    team_id = Column(Integer, ForeignKey("teams.id"))
    
    # Possession and shots
    possession = Column(Float, nullable=True)
    shots_total = Column(Integer, nullable=True)
    shots_on_target = Column(Integer, nullable=True)
    shots_off_target = Column(Integer, nullable=True)
    
    # Set pieces
    corners = Column(Integer, nullable=True)
    free_kicks = Column(Integer, nullable=True)
    throw_ins = Column(Integer, nullable=True)
    
    # Other stats
    fouls = Column(Integer, nullable=True)
    yellow_cards = Column(Integer, nullable=True)
    red_cards = Column(Integer, nullable=True)
    offsides = Column(Integer, nullable=True)
    saves = Column(Integer, nullable=True)
    
    # Relationships
    match = relationship("Match", back_populates="match_statistics")
    
    def __repr__(self):
        return f"<MatchStatistic for match_id={self.match_id}, team_id={self.team_id}>"


class PlayerStatistic(Base):
    """Player statistics model for storing player-level statistics in a match."""
    
    __tablename__ = "player_statistics"
    
    id = Column(Integer, primary_key=True, index=True)
    match_id = Column(Integer, ForeignKey("matches.id"))
    player_id = Column(Integer, ForeignKey("players.id"))
    
    # Playing time
    minutes_played = Column(Integer, nullable=True)
    
    # Goals and assists
    goals = Column(Integer, nullable=True)
    assists = Column(Integer, nullable=True)
    
    # Shots
    shots_total = Column(Integer, nullable=True)
    shots_on_target = Column(Integer, nullable=True)
    
    # Passes
    passes_total = Column(Integer, nullable=True)
    passes_accuracy = Column(Float, nullable=True)
    key_passes = Column(Integer, nullable=True)
    
    # Defensive actions
    tackles = Column(Integer, nullable=True)
    blocks = Column(Integer, nullable=True)
    interceptions = Column(Integer, nullable=True)
    
    # Disciplinary
    yellow_cards = Column(Integer, nullable=True)
    red_cards = Column(Integer, nullable=True)
    
    # Relationships
    match = relationship("Match", back_populates="player_statistics")
    player = relationship("Player", back_populates="statistics")
    
    def __repr__(self):
        return f"<PlayerStatistic for player_id={self.player_id} in match_id={self.match_id}>"


class BettingOdds(Base):
    """Betting odds model for storing odds from various bookmakers."""
    
    __tablename__ = "betting_odds"
    
    id = Column(Integer, primary_key=True, index=True)
    match_id = Column(Integer, ForeignKey("matches.id"))
    bookmaker = Column(String, index=True)
    
    # 1X2 odds (home win, draw, away win)
    home_win_odds = Column(Float, nullable=True)
    draw_odds = Column(Float, nullable=True)
    away_win_odds = Column(Float, nullable=True)
    
    # Over/Under goals odds
    over_2_5_odds = Column(Float, nullable=True)
    under_2_5_odds = Column(Float, nullable=True)
    
    # Both teams to score odds
    btts_yes_odds = Column(Float, nullable=True)
    btts_no_odds = Column(Float, nullable=True)
    
    # Asian handicap odds
    handicap_home_odds = Column(Float, nullable=True)
    handicap_away_odds = Column(Float, nullable=True)
    handicap_value = Column(Float, nullable=True)
    
    # Timestamp for when odds were fetched
    timestamp = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    match = relationship("Match", back_populates="betting_odds")
    
    def __repr__(self):
        return f"<BettingOdds for match_id={self.match_id} from {self.bookmaker}>"


class Prediction(Base):
    """Prediction model for storing match outcome predictions."""
    
    __tablename__ = "predictions"
    
    id = Column(Integer, primary_key=True, index=True)
    match_id = Column(Integer, ForeignKey("matches.id"))
    
    # Predicted scores
    predicted_home_score = Column(Float)
    predicted_away_score = Column(Float)
    
    # Predicted probabilities
    home_win_probability = Column(Float)
    draw_probability = Column(Float)
    away_win_probability = Column(Float)
    
    # Confidence level (0-1)
    confidence = Column(Float)
    
    # Value bet indicators
    is_value_bet = Column(Boolean, default=False)
    value_bet_type = Column(String, nullable=True)  # e.g., "home_win", "under_2_5", etc.
    expected_value = Column(Float, nullable=True)
    
    # Model information
    model_version = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Evaluation (to be filled after match is completed)
    was_correct = Column(Boolean, nullable=True)
    
    # Relationships
    match = relationship("Match", back_populates="predictions")
    
    def __repr__(self):
        return f"<Prediction for match_id={self.match_id}>"
