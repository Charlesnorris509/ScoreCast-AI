"""
Database initialization and migration script for the Soccer Match Prediction Application.
This module creates the database tables based on the SQLAlchemy models.
"""

import sys
import os

# Get the absolute path to the project root directory
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, project_root)

from config.database import Base, engine
from models.database.models import (
    Team, Player, League, Match, MatchStatistic, 
    PlayerStatistic, BettingOdds, Prediction
)

def init_db():
    """
    Initialize the database by creating all tables defined in the models.
    """
    # Create all tables
    Base.metadata.create_all(bind=engine)
    print("Database tables created successfully.")

if __name__ == "__main__":
    init_db()
