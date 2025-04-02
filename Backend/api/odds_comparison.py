"""
Betting odds comparison module for the Soccer Match Prediction Application.
This module provides functionality for comparing betting odds from different bookmakers
and identifying value betting opportunities.
"""

import os
import sys
import logging
import json
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from sqlalchemy import func

# Add the project root directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.database.models import Match, Team, League, Prediction, BettingOdds
from config.database import SessionLocal

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("odds_comparison.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("odds_comparison")

class OddsComparison:
    """
    Class for comparing betting odds and identifying value betting opportunities.
    """
    
    def __init__(self):
        """
        Initialize the OddsComparison class.
        """
        self.db = SessionLocal()
    
    def __del__(self):
        """
        Close database session when object is destroyed.
        """
        self.db.close()
    
    def get_best_odds(self, match_id):
        """
        Get the best available odds for a match from all bookmakers.
        
        Args:
            match_id: Match ID
            
        Returns:
            dict: Best odds for each market
        """
        try:
            # Get all odds for the match
            odds = self.db.query(BettingOdds).filter(
                BettingOdds.match_id == match_id
            ).all()
            
            if not odds:
                return None
            
            # Initialize best odds with worst possible values
            best_odds = {
                'home_win': 0,
                'draw': 0,
                'away_win': 0,
                'over_2_5': 0,
                'under_2_5': 0,
                'btts_yes': 0,
                'btts_no': 0
            }
            
            # Find best odds for each market
            for odd in odds:
                if odd.home_win_odds and odd.home_win_odds > best_odds['home_win']:
                    best_odds['home_win'] = odd.home_win_odds
                
                if odd.draw_odds and odd.draw_odds > best_odds['draw']:
                    best_odds['draw'] = odd.draw_odds
                
                if odd.away_win_odds and odd.away_win_odds > best_odds['away_win']:
                    best_odds['away_win'] = odd.away_win_odds
                
                if odd.over_2_5_odds and odd.over_2_5_odds > best_odds['over_2_5']:
                    best_odds['over_2_5'] = odd.over_2_5_odds
                
                if odd.under_2_5_odds and odd.under_2_5_odds > best_odds['under_2_5']:
                    best_odds['under_2_5'] = odd.under_2_5_odds
                
                if odd.btts_yes_odds and odd.btts_yes_odds > best_odds['btts_yes']:
                    best_odds['btts_yes'] = odd.btts_yes_odds
                
                if odd.btts_no_odds and odd.btts_no_odds > best_odds['btts_no']:
                    best_odds['btts_no'] = odd.btts_no_odds
            
            # Set to None if no odds found for a market
            for key in best_odds:
                if best_odds[key] == 0:
                    best_odds[key] = None
            
            return best_odds
        
        except Exception as e:
            logger.error(f"Error getting best odds: {str(e)}")
            return None
    
    def calculate_implied_probabilities(self, odds):
        """
        Calculate implied probabilities from odds.
        
        Args:
            odds: Odds dictionary
            
        Returns:
            dict: Implied probabilities
        """
        if not odds:
            return None
        
        implied_probs = {}
        
        for key, value in odds.items():
            if value:
                implied_probs[key] = 1 / value
            else:
                implied_probs[key] = None
        
        return implied_probs
    
    def calculate_overround(self, odds):
        """
        Calculate bookmaker overround (margin).
        
        Args:
            odds: Odds dictionary
            
        Returns:
            float: Overround percentage
        """
        if not odds or not odds['home_win'] or not odds['draw'] or not odds['away_win']:
            return None
        
        # Calculate implied probabilities
        home_prob = 1 / odds['home_win']
        draw_prob = 1 / odds['draw']
        away_prob = 1 / odds['away_win']
        
        # Calculate overround
        overround = (home_prob + draw_prob + away_prob) - 1
        
        return overround
    
    def identify_value_bets(self, match_id):
        """
        Identify value betting opportunities for a match.
        
        Args:
            match_id: Match ID
            
        Returns:
            dict: Value betting opportunities
        """
        try:
            # Get match
            match = self.db.query(Match).filter(Match.id == match_id).first()
            
            if not match:
                return None
            
            # Get prediction for the match
            prediction = self.db.query(Prediction).filter(
                Prediction.match_id == match.id
            ).order_by(Prediction.id.desc()).first()
            
            if not prediction:
                return None
            
            # Get best odds for the match
            best_odds = self.get_best_odds(match_id)
            
            if not best_odds:
                return None
            
            # Calculate implied probabilities from odds
            implied_probs = self.calculate_implied_probabilities(best_odds)
            
            # Initialize value bets
            value_bets = {
                'home_win': {
                    'is_value': False,
                    'our_probability': prediction.home_win_probability,
                    'implied_probability': implied_probs['home_win'],
                    'odds': best_odds['home_win'],
                    'expected_value': None,
                    'edge': None
                },
                'draw': {
                    'is_value': False,
                    'our_probability': prediction.draw_probability,
                    'implied_probability': implied_probs['draw'],
                    'odds': best_odds['draw'],
                    'expected_value': None,
                    'edge': None
                },
                'away_win': {
                    'is_value': False,
                    'our_probability': prediction.away_win_probability,
                    'implied_probability': implied_probs['away_win'],
                    'odds': best_odds['away_win'],
                    'expected_value': None,
                    'edge': None
                }
            }
            
            # Calculate expected value and edge for each market
            for key in value_bets:
                if (value_bets[key]['our_probability'] is not None and 
                    value_bets[key]['implied_probability'] is not None):
                    
                    # Calculate edge (difference between our probability and implied probability)
                    value_bets[key]['edge'] = value_bets[key]['our_probability'] - value_bets[key]['implied_probability']
                    
                    # Calculate expected value
                    value_bets[key]['expected_value'] = (value_bets[key]['our_probability'] * (value_bets[key]['odds'] - 1)) - (1 - value_bets[key]['our_probability'])
                    
                    # Determine if it's a value bet (our probability > implied probability)
                    value_bets[key]['is_value'] = value_bets[key]['edge'] > 0.05  # 5% edge threshold
            
            return value_bets
        
        except Exception as e:
            logger.error(f"Error identifying value bets: {str(e)}")
            return None
    
    def get_bookmaker_comparison(self, match_id):
        """
        Get odds comparison across different bookmakers for a match.
        
        Args:
            match_id: Match ID
            
        Returns:
            dict: Odds comparison data
        """
        try:
            # Get match
            match = self.db.query(Match).filter(Match.id == match_id).first()
            
            if not match:
                return None
            
            # Get all odds for the match
            odds = self.db.query(BettingOdds).filter(
                BettingOdds.match_id == match.id
            ).all()
            
            if not odds:
                return None
            
            # Get prediction for the match
            prediction = self.db.query(Prediction).filter(
                Prediction.match_id == match.id
            ).order_by(Prediction.id.desc()).first()
            
            # Prepare bookmaker comparison data
            bookmakers = []
            
            for odd in odds:
                bookmaker_data = {
                    'bookmaker': odd.bookmaker,
                    'home_win': odd.home_win_odds,
                    'draw': odd.draw_odds,
                    'away_win': odd.away_win_odds,
                    'over_2_5': odd.over_2_5_odds,
                    'under_2_5': odd.under_2_5_odds,
                    'btts_yes': odd.btts_yes_odds,
                    'btts_no': odd.btts_no_odds,
                    'timestamp': odd.timestamp.isoformat() if odd.timestamp else None,
                    'overround': self.calculate_overround({
                        'home_win': odd.home_win_odds,
                        'draw': odd.draw_odds,
                        'away_win': odd.away_win_odds
                    })
                }
                
                bookmakers.append(bookmaker_data)
            
            # Add our prediction as a "bookmaker" for comparison
            if prediction:
                # Calculate implied odds from our probabilities
                home_implied_odds = 1 / prediction.home_win_probability if prediction.home_win_probability else None
                draw_implied_odds = 1 / prediction.draw_probability if prediction.draw_probability else None
                away_implied_odds = 1 / prediction.away_win_probability if prediction.away_win_probability else None
                
                our_prediction = {
                    'bookmaker': 'Our Prediction',
                    'home_win': round(home_implied_odds, 2) if home_implied_odds else None,
                    'draw': round(draw_implied_odds, 2) if draw_implied_odds else None,
                    'away_win': round(away_implied_odds, 2) if away_implied_odds else None,
                    'over_2_5': None,
                    'under_2_5': None,
                    'btts_yes': None,
                    'btts_no': None,
                    'timestamp': prediction.timestamp.isoformat() if prediction.timestamp else None,
                    'overround': 0  # Our prediction has no overround
                }
                
                bookmakers.append(our_prediction)
            
            # Get value bets
            value_bets = self.identify_value_bets(match_id)
            
            comparison_data = {
                'match': {
                    'id': match.id,
                    'home_team': match.home_team.name if match.home_team else "Unknown",
                    'away_team': match.away_team.name if match.away_team else "Unknown",
                    'league': match.league.name if match.league else "Unknown",
                    'match_date': match.match_date.isoformat()
                },
                'bookmakers': bookmakers,
                'value_bets': value_bets
            }
            
            return comparison_data
        
        except Exception as e:
            logger.error(f"Error getting bookmaker comparison: {str(e)}")
            return None
    
    def get_best_value_bets(self, days_ahead=7, min_edge=0.05, min_odds=1.5):
        """
        Get the best value betting opportunities for upcoming matches.
        
        Args:
            days_ahead: Number of days ahead to look for matches
            min_edge: Minimum edge (difference between our probability and implied probability)
            min_odds: Minimum odds to consider
            
        Returns:
            list: Best value betting opportunities
        """
        try:
            # Get upcoming matches
            current_date = datetime.now()
            end_date = current_date + timedelta(days=days_ahead)
            
            # Get matches with predictions and odds
            matches = self.db.query(Match).filter(
                Match.match_date >= current_date,
                Match.match_date <= end_date
            ).all()
            
            if not matches:
                return []
            
            # Find value bets for each match
            value_bet_opportunities = []
            
            for match in matches:
                value_bets = self.identify_value_bets(match.id)
                
                if not value_bets:
                    continue
                
                # Check each market for value bets
                for market, data in value_bets.items():
                    if (data['is_value'] and 
                        data['edge'] >= min_edge and 
                        data['odds'] >= min_odds):
                        
                        opportunity = {
                            'match_id': match.id,
                            'home_team': match.home_team.name if match.home_team else "Unknown",
                            'away_team': match.away_team.name if match.away_team else "Unknown",
                            'league': match.league.name if match.league else "Unknown",
                            'match_date': match.match_date.isoformat(),
                            'market': market,
                            'odds': data['odds'],
                            'our_probability': data['our_probability'],
                            'implied_probability': data['implied_probability'],
                            'edge': data['edge'],
                            'expected_value': data['expected_value']
                        }
                        
                        value_bet_opportunities.append(opportunity)
            
            # Sort by expected value (descending)
            value_bet_opportunities.sort(key=lambda x: x['expected_value'], reverse=True)
            
            return value_bet_opportunities
        
        except Exception as e:
            logger.error(f"Error getting best value bets: {str(e)}")
            return []
    
    def calculate_closing_line_value(self, prediction_id):
        """
        Calculate closing line value (CLV) for a prediction.
        CLV measures how much value you captured compared to closing odds.
        
        Args:
            prediction_id: Prediction ID
            
        Returns:
            float: Closing line value
        """
        try:
            # Get prediction
            prediction = self.db.query(Prediction).filter(Prediction.id == prediction_id).first()
            
            if not prediction:
                return None
            
            # Get match
            match = self.db.query(Match).filter(Match.id == prediction.match_id).first()
            
            if not match:
                return None
            
            # Get odds at prediction time
            prediction_time_odds = self.db.query(BettingOdds).filter(
                BettingOdds.match_id == match.id,
                BettingOdds.timestamp <= prediction.timestamp
            ).order_by(BettingOdds.timestamp.desc()).first()
            
            # Get closing odds (last odds before match start)
            closing_odds = self.db.query(BettingOdds).filter(
                BettingOdds.match_id == match.id,
                BettingOdds.timestamp <= match.match_date
            ).order_by(BettingOdds.timestamp.desc()).first()
            
            if not prediction_time_odds or not closing_odds:
                return None
            
            # Calculate CLV based on the value bet type
            if prediction.value_bet_type == 'home_win':
                prediction_odds = prediction_time_odds.home_win_odds
                closing_odds_value = closing_odds.home_win_odds
            elif prediction.value_bet_type == 'draw':
                prediction_odds = prediction_time_odds.draw_odds
                closing_odds_value = closing_odds.draw_odds
            elif prediction.value_bet_type == 'away_win':
                prediction_odds = prediction_time_odds.away_win_odds
                closing_odds_value = closing_odds.away_win_odds
            else:
                return None
            
            # Calculate CLV
            prediction_implied_prob = 1 / prediction_odds
            closing_implied_prob = 1 / closing_odds_value
            
            # CLV formula: (prediction_odds / closing_odds) - 1
            clv = (prediction_odds / closing_odds_value) - 1
            
            return clv
        
        except Exception as e:
            logger.error(f"Error calculating closing line value: {str(e)}")
            return None
    
    def analyze_value_bet_performance(self, days_back=30):
        """
        Analyze the performance of value bets over a period of time.
        
        Args:
            days_back: Number of days to look back
            
        Returns:
            dict: Value bet performance statistics
        """
        try:
            # Get completed matches with value bet predictions
            start_date = datetime.now() - timedelta(days=days_back)
            end_date = datetime.now()
            
            # Get predictions with value bets
            value_bet_predictions = self.db.query(Prediction).join(
                Match, Prediction.match_id == Match.id
            ).filter(
                Match.match_date >= start_date,
                Match.match_date <= end_date,
                Match.status == "FT",  # Full Time (completed matches)
                Prediction.is_value_bet == True
            ).all()
            
            if not value_bet_predictions:
                return {
                    'total_value_bets': 0,
                    'correct_value_bets': 0,
                    'accuracy': 0,
                    'profit_loss': 0,
                    'roi': 0,
                    'average_odds': 0,
                    'average_edge': 0,
                    'average_clv': 0,
                    'market_breakdown': {
                        'home_win': {'total': 0, 'correct': 0, 'accuracy': 0},
                        'draw': {'total': 0, 'correct': 0, 'accuracy': 0},
                        'away_win': {'total': 0, 'correct': 0, 'accuracy': 0}
                    }
                }
            
            # Initialize statistics
            total_value_bets = len(value_bet_predictions)
            correct_value_bets = 0
            total_profit_loss = 0
            total_stake = total_value_bets  # Assuming 1 unit stake per bet
            total_odds = 0
            total_edge = 0
            total_clv = 0
            valid_clv_count = 0
            
            # Market breakdown
            market_breakdown = {
                'home_win': {'total': 0, 'correct': 0, 'profit_loss': 0},
                'draw': {'total': 0, 'correct': 0, 'profit_loss': 0},
                'away_win': {'total': 0, 'correct': 0, 'profit_loss': 0}
            }
            
            # Analyze each value bet
            for prediction in value_bet_predictions:
                # Get match
                match = self.db.query(Match).filter(Match.id == prediction.match_id).first()
                
                if not match:
                    continue
                
                # Get odds at prediction time
                odds = self.db.query(BettingOdds).filter(
                    BettingOdds.match_id == match.id,
                    BettingOdds.timestamp <= prediction.timestamp
                ).order_by(BettingOdds.timestamp.desc()).first()
                
                if not odds:
                    continue
                
                # Get bet odds based on value bet type
                if prediction.value_bet_type == 'home_win':
                    bet_odds = odds.home_win_odds
                    market_breakdown['home_win']['total'] += 1
                elif prediction.value_bet_type == 'draw':
                    bet_odds = odds.draw_odds
                    market_breakdown['draw']['total'] += 1
                elif prediction.value_bet_type == 'away_win':
                    bet_odds = odds.away_win_odds
                    market_breakdown['away_win']['total'] += 1
                else:
                    continue
                
                # Calculate edge
                implied_prob = 1 / bet_odds
                edge = 0
                
                if prediction.value_bet_type == 'home_win':
                    edge = prediction.home_win_probability - implied_prob
                elif prediction.value_bet_type == 'draw':
                    edge = prediction.draw_probability - implied_prob
                elif prediction.value_bet_type == 'away_win':
                    edge = prediction.away_win_probability - implied_prob
                
                # Calculate CLV
                clv = self.calculate_closing_line_value(prediction.id)
                
                if clv is not None:
                    total_clv += clv
                    valid_clv_count += 1
                
                # Check if prediction was correct
                if prediction.was_correct:
                    correct_value_bets += 1
                    total_profit_loss += bet_odds - 1  # Profit (odds - 1)
                    
                    # Update market breakdown
                    if prediction.value_bet_type == 'home_win':
                        market_breakdown['home_win']['correct'] += 1
                        market_breakdown['home_win']['profit_loss'] += bet_odds - 1
                    elif prediction.value_bet_type == 'draw':
                        market_breakdown['draw']['correct'] += 1
                        market_breakdown['draw']['profit_loss'] += bet_odds - 1
                    elif prediction.value_bet_type == 'away_win':
                        market_breakdown['away_win']['correct'] += 1
                        market_breakdown['away_win']['profit_loss'] += bet_odds - 1
                else:
                    total_profit_loss -= 1  # Loss (stake)
                    
                    # Update market breakdown
                    if prediction.value_bet_type == 'home_win':
                        market_breakdown['home_win']['profit_loss'] -= 1
                    elif prediction.value_bet_type == 'draw':
                        market_breakdown['draw']['profit_loss'] -= 1
                    elif prediction.value_bet_type == 'away_win':
                        market_breakdown['away_win']['profit_loss'] -= 1
                
                # Update totals
                total_odds += bet_odds
                total_edge += edge
            
            # Calculate averages and percentages
            accuracy = correct_value_bets / total_value_bets if total_value_bets > 0 else 0
            roi = total_profit_loss / total_stake if total_stake > 0 else 0
            average_odds = total_odds / total_value_bets if total_value_bets > 0 else 0
            average_edge = total_edge / total_value_bets if total_value_bets > 0 else 0
            average_clv = total_clv / valid_clv_count if valid_clv_count > 0 else 0
            
            # Calculate market breakdown percentages
            for market in market_breakdown:
                market_breakdown[market]['accuracy'] = (
                    market_breakdown[market]['correct'] / market_breakdown[market]['total']
                    if market_breakdown[market]['total'] > 0 else 0
                )
            
            # Prepare performance statistics
            performance_stats = {
                'total_value_bets': total_value_bets,
                'correct_value_bets': correct_value_bets,
                'accuracy': accuracy,
                'profit_loss': total_profit_loss,
                'roi': roi,
                'average_odds': average_odds,
                'average_edge': average_edge,
                'average_clv': average_clv,
                'market_breakdown': market_breakdown
            }
            
            return performance_stats
        
        except Exception as e:
            logger.error(f"Error analyzing value bet performance: {str(e)}")
            return None

# Example usage
if __name__ == "__main__":
    odds_comparison = OddsComparison()
    
    # Get best value bets
    value_bets = odds_comparison.get_best_value_bets()
    print(f"Found {len(value_bets)} value betting opportunities")
    
    # Analyze value bet performance
    performance = odds_comparison.analyze_value_bet_performance()
    if performance:
        print(f"Value bet accuracy: {performance['accuracy'] * 100:.1f}%")
        print(f"ROI: {performance['roi'] * 100:.1f}%")
