import pandas as pd
import numpy as np
from typing import Set, Dict, Optional, Tuple
from pathlib import Path

def calculate_z_scores(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate z-scores for AdjOE, AdjDE, and AdjEM metrics.
    Higher values are better for all metrics.
    Note: For AdjDE, we multiply by -1 since lower values are better for defense.
    """
    df_with_z = df.copy()
    
    # Calculate z-scores for offensive efficiency (higher is better)
    df_with_z['AdjOE z-score'] = (df['AdjOE'] - df['AdjOE'].mean()) / df['AdjOE'].std()
    
    # Calculate z-scores for defensive efficiency (lower is better, so multiply by -1)
    df_with_z['AdjDE z-score'] = -1 * (df['AdjDE'] - df['AdjDE'].mean()) / df['AdjDE'].std()
    
    # Calculate z-scores for efficiency margin (higher is better)
    df_with_z['AdjEM z-score'] = (df['AdjEM'] - df['AdjEM'].mean()) / df['AdjEM'].std()
    
    return df_with_z

def load_dataset(file_path: str) -> Optional[pd.DataFrame]:
    try:
        return pd.read_csv(Path(file_path))
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None
    except Exception as e:
        print(f"Error loading file: {str(e)}")
        return None

def validate_columns(df: pd.DataFrame, required_columns: Set[str]) -> bool:
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        print(f"Error: Missing required columns: {missing_columns}")
        return False
    return True

def calculate_percentiles(df: pd.DataFrame, filter_condition: str, confidence_level: float) -> Dict[str, float]:
    """
    Calculate percentiles based on confidence level. Will be used to create our pools of teams in current dataset.
    """
    if filter_condition == 'champions':
        filtered_df = df[df['Games won'] == 6]
    elif filter_condition == 'championship_game':
        filtered_df = df[df['Games won'].isin([6, 5])]
    elif filter_condition == 'final_4':
        filtered_df = df[df['Games won'].isin([6, 5, 4])]
    elif filter_condition == 'elite_8':
        filtered_df = df[df['Games won'].isin([6, 5, 4, 3])]
    elif filter_condition == 'sweet_16':
        filtered_df = df[df['Games won'].isin([6, 5, 4, 3, 2])]
    elif filter_condition == 'round_32':
        filtered_df = df[df['Games won'].isin([6, 5, 4, 3, 2, 1])]
    else:
        raise ValueError(f"Unknown filter condition: {filter_condition}")
    
    return {
        "AdjOE_pct": filtered_df['AdjOE z-score'].quantile(confidence_level),
        "AdjDE_pct": filtered_df['AdjDE z-score'].quantile(confidence_level),
        "AdjEM_pct": filtered_df['AdjEM z-score'].quantile(confidence_level)
    }

def calculate_metric_correlations(historical_df: pd.DataFrame, thresholds: Dict[str, float]) -> Dict[str, float]:
    """
    Calculate correlations between metrics and games won for teams meeting thresholds.
    """
    # Filter for teams meeting all thresholds
    qualifying_teams = historical_df[
        (historical_df['AdjOE z-score'] > thresholds['AdjOE_pct']) &
        (historical_df['AdjDE z-score'] > thresholds['AdjDE_pct']) &
        (historical_df['AdjEM z-score'] > thresholds['AdjEM_pct'])
    ]
    
    if len(qualifying_teams) < 2:
        return {'oe': 1/3, 'de': 1/3, 'em': 1/3}
    
    # Calculate correlations
    correlations = {
        'oe': qualifying_teams['AdjOE z-score'].corr(qualifying_teams['Games won']),
        'de': qualifying_teams['AdjDE z-score'].corr(qualifying_teams['Games won']),
        'em': qualifying_teams['AdjEM z-score'].corr(qualifying_teams['Games won'])
    }
    
    # Convert negative correlations to 0 and normalize
    correlations = {k: max(0, v) for k, v in correlations.items()}
    total = sum(correlations.values())
    
    if total == 0:
        return {'oe': 1/3, 'de': 1/3, 'em': 1/3}
    
    return {k: v/total for k, v in correlations.items()}

def calculate_similarity_score(team_metrics: pd.Series, thresholds: Dict[str, float], 
                               correlations: Dict[str, float]) -> float:
    """Calculate correlation-weighted similarity score."""
    oe_score = max(0, team_metrics['AdjOE z-score'] - thresholds['AdjOE_pct'])
    de_score = max(0, team_metrics['AdjDE z-score'] - thresholds['AdjDE_pct'])
    em_score = max(0, team_metrics['AdjEM z-score'] - thresholds['AdjEM_pct'])
    
    weighted_score = (
        oe_score * correlations['oe'] +
        de_score * correlations['de'] +
        em_score * correlations['em']
    )
    
    return weighted_score

def analyze_teams_with_scores(qualifying_teams: pd.DataFrame, thresholds: Dict[str, float], 
                              correlations: Dict[str, float]) -> pd.DataFrame:
    """
    Add correlation-weighted similarity scores to already qualified teams and sort by score.
    """
    teams_with_scores = qualifying_teams.copy()
    teams_with_scores['similarity_score'] = teams_with_scores.apply(
        lambda row: calculate_similarity_score(row, thresholds, correlations), 
        axis=1
    )
    return teams_with_scores.sort_values('similarity_score', ascending=False)

def analyze_teams_meeting_criteria(df: pd.DataFrame, thresholds: Dict[str, float], 
                                   correlations: Dict[str, float]) -> Tuple[int, pd.DataFrame]:
    """Identify teams meeting criteria and calculate weighted scores."""
    criteria = (
        (df['AdjOE z-score'] > thresholds['AdjOE_pct']) & 
        (df['AdjDE z-score'] > thresholds['AdjDE_pct']) & 
        (df['AdjEM z-score'] > thresholds['AdjEM_pct'])
    )
    
    qualifying_teams = df[criteria].copy()
    qualifying_teams['similarity_score'] = qualifying_teams.apply(
        lambda row: calculate_similarity_score(row, thresholds, correlations),
        axis=1
    )
    qualifying_teams = qualifying_teams.sort_values('similarity_score', ascending=False)
    
    return len(qualifying_teams), qualifying_teams

def calculate_normalized_probabilities(qualifying_teams: pd.DataFrame, thresholds: Dict[str, float],
                                       correlations: Dict[str, float], target_sum: int) -> pd.DataFrame:
    """Calculate normalized probabilities with correlation weights."""
    if qualifying_teams.empty:
        return qualifying_teams
    
    if 'similarity_score' not in qualifying_teams.columns:
        qualifying_teams = analyze_teams_with_scores(qualifying_teams, thresholds, correlations)[0]
    
    total_similarity = qualifying_teams['similarity_score'].sum()
    
    if total_similarity == 0:
        qualifying_teams['normalized_probability'] = target_sum / len(qualifying_teams)
    else:
        qualifying_teams['normalized_probability'] = (qualifying_teams['similarity_score'] / total_similarity) * target_sum
    
    max_prob = qualifying_teams['normalized_probability'].max()
    if max_prob > 1:
        scaling_factor = 0.99 / max_prob
        qualifying_teams['normalized_probability'] *= scaling_factor
    
    return qualifying_teams.sort_values('normalized_probability', ascending=False)

def calculate_implied_odds(prob: float) -> int:
    """Calculate implied implied betting odds from probability."""
    if prob < 0.50:
        return int(((1 - prob) / prob) * 100)
    else:
        return int(-((prob / (1 - prob)) * 100))

def print_qualifying_teams_with_probabilities(teams: pd.DataFrame, round_name: str, 
                                              thresholds: Dict[str, float], correlations: Dict[str, float],
                                              target_sum: int) -> None:
    """Print teams with correlation-weighted probabilities and implied odds."""
    teams_with_probs = calculate_normalized_probabilities(teams, thresholds, correlations, target_sum)
    
    # Add implied odds calculation
    teams_with_probs['implied_odds'] = teams_with_probs['normalized_probability'].apply(calculate_implied_odds)
    
    print(f"\nTeams meeting {round_name} criteria (sorted by probability):")
    print(f"{'Team':<35} {'Conference':<15} {'Prob':>8} {'Imp Odds':>8} {'Sim Score':>10} {'AdjEM z':>8} {'AdjOE z':>8} {'AdjDE z':>8}")
    print("-" * 108)
    
    for _, team in teams_with_probs.iterrows():
        print(f"{team['TeamName']:<35} {team['CONF']:<15} {team['normalized_probability']:>8.3f} "
              f"{'+' + str(team['implied_odds']) if team['implied_odds'] > 0 else team['implied_odds']:>9} {team['similarity_score']:>10.3f} {team['AdjEM z-score']:>8.2f} "
              f"{team['AdjOE z-score']:>8.2f} {team['AdjDE z-score']:>8.2f}")
    
    total_prob = teams_with_probs['normalized_probability'].sum()
    print(f"\nMetric Correlations with Wins (weights):")
    print(f"AdjOE: {correlations['oe']:.3f}")
    print(f"AdjDE: {correlations['de']:.3f}")
    print(f"AdjEM: {correlations['em']:.3f}")
    print(f"Total probability sum: {total_prob:.3f} (target: {target_sum})")

def select_tournament_teams(df: pd.DataFrame, total_teams: int = 68) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Select tournament teams based on conference auto-bids and at-large selections.
    """
    # Get auto bids (highest AdjEM team from each conference)
    auto_bids = df.loc[df.groupby('CONF')['AdjEM'].idxmax()].copy()
    auto_bids = auto_bids.sort_values('AdjEM', ascending=False)
    
    # Get remaining teams for at-large consideration
    remaining_teams = df[~df.index.isin(auto_bids.index)].copy()
    remaining_teams = remaining_teams.sort_values('AdjEM', ascending=False)
    
    # Select at-large teams (top AdjEM teams not already selected)
    at_large_spots = total_teams - len(auto_bids)
    at_large = remaining_teams.head(at_large_spots)
    
    return auto_bids, at_large

def print_tournament_field(auto_bids: pd.DataFrame, at_large: pd.DataFrame) -> None:
    """Print the tournament field."""
    print("\nTOURNAMENT FIELD")
    print("=" * 80)
    
    print("\nAUTO BID TEAMS:")
    print("-" * 80)
    print(f"{'Conference':<15} {'Team':<35} {'AdjEM':<10}")
    print("-" * 80)
    for _, team in auto_bids.iterrows():
        print(f"{team['CONF']:<15} {team['TeamName']:<35} {team['AdjEM']:>8.2f}")
    
    print("\nAT-LARGE TEAMS:")
    print("-" * 80)
    print(f"{'Team':<35} {'Conference':<15} {'AdjEM':<10}")
    print("-" * 80)
    for _, team in at_large.iterrows():
        print(f"{team['TeamName']:<35} {team['CONF']:<15} {team['AdjEM']:>8.2f}")
    
    print("\nSUMMARY:")
    print("-" * 80)
    print(f"Total Auto Bids: {len(auto_bids)}")
    print(f"Total At-Large: {len(at_large)}")
    print(f"Total Teams: {len(auto_bids) + len(at_large)}")

def get_current_file_path() -> str:
    """
    Prompt the user for the current season's data file path.
    """
    while True:
        file_path = input("\nPlease enter the path to the current season's data file: ").strip()
        if file_path:
            return file_path
        print("Error: File path cannot be empty. Please try again.")

def main():
    """
    Main execution function for the tournament analysis program.
    
    Performs the following steps:
    1. Loads and validates historical and current season data
    2. Calculates z-scores for current teams
    3. Selects tournament field (auto-bids and at-large)
    4. Analyzes each tournament round:
        - Calculates thresholds from historical data
        - Identifies qualifying teams
        - Calculates probabilities and implied odds
    5. Exports complete analysis to a text file
    
    Configuration:
        - Historical data file path
        - Current season data file path
        - Required columns for validation
        - Round definitions and percentiles for confidence levels
    """
    HISTORICAL_FILE = "/Users/isaiahnick/Desktop/NCAAM/2001-24 Tournament Teams.csv"
    CURRENT_FILE = get_current_file_path()
    
    HISTORICAL_REQUIRED_COLUMNS = {'Games won', 'AdjOE z-score', 'AdjDE z-score', 'AdjEM z-score'}
    CURRENT_REQUIRED_COLUMNS = {'AdjOE', 'AdjDE', 'AdjEM', 'CONF', 'TeamName'}
    
    # List of tournament rounds with format: round_id, round_name, confidence_level, target_sum
    # Percentiles can be edited with the confidence level
    ROUNDS = [
        ('champions', 'Champions', 0.05, 1),
        ('championship_game', 'Championship Game', 0.10, 2),
        ('final_4', 'Final Four', 0.10, 4),
        ('elite_8', 'Elite Eight', 0.10, 8),
        ('sweet_16', 'Sweet Sixteen', 0.10, 16),
        ('round_32', 'Round of 32', 0.10, 32)
    ]
    
    # Load and validate data
    historical_df = load_dataset(HISTORICAL_FILE)
    if historical_df is None or not validate_columns(historical_df, HISTORICAL_REQUIRED_COLUMNS):
        return
    
    current_df = load_dataset(CURRENT_FILE)
    if current_df is None or not validate_columns(current_df, CURRENT_REQUIRED_COLUMNS):
        return
    
    current_df = calculate_z_scores(current_df)
    
    # Select tournament field
    auto_bids, at_large = select_tournament_teams(current_df)
    tournament_field = pd.concat([auto_bids, at_large])
    print_tournament_field(auto_bids, at_large)
    
    print("\nAnalyzing tournament field with correlation-weighted metrics:")
    print("=" * 80)
    
    round_results = {}
    
    for round_id, round_name, confidence_level, target_sum in ROUNDS:
        print(f"\n{round_name.upper()}")
        print("=" * 80)
        
        # Calculate thresholds and correlations
        thresholds = calculate_percentiles(historical_df, round_id, confidence_level)
        correlations = calculate_metric_correlations(historical_df, thresholds)
        
        print(f"Confidence Level: {confidence_level:.2%}")
        print(f"Thresholds:")
        print(f"  AdjOE z-score > {thresholds['AdjOE_pct']:.2f}")
        print(f"  AdjDE z-score > {thresholds['AdjDE_pct']:.2f}")
        print(f"  AdjEM z-score > {thresholds['AdjEM_pct']:.2f}")
        
        team_count, qualifying_teams = analyze_teams_meeting_criteria(tournament_field, thresholds, correlations)
        print(f"\nNumber of teams meeting criteria: {team_count}")
        
        print_qualifying_teams_with_probabilities(qualifying_teams, round_name, thresholds, correlations, target_sum)
        print("-" * 100)
        
        round_results[round_name] = {
            'thresholds': thresholds,
            'correlations': correlations,
            'qualifying_teams': qualifying_teams,
            'team_count': team_count
        }
    
# Export results to a single text file
    with open('tournament_analysis_results.txt', 'w') as f:
        f.write("Analyzing tournament field with correlation-weighted metrics:\n")
        f.write("=" * 80 + "\n\n")
        
        for round_name, results in round_results.items():
            # Write round header
            f.write(f"{round_name}\n")
            f.write("=" * 80 + "\n")
            
            # Write confidence level
            conf_level = next(cl for rd, rn, cl, _ in ROUNDS if rn == round_name)
            f.write(f"Confidence Level: {conf_level:.2%}\n")
            
            # Write thresholds
            f.write("Thresholds:\n")
            f.write(f"  AdjOE z-score > {results['thresholds']['AdjOE_pct']:.2f}\n")
            f.write(f"  AdjDE z-score > {results['thresholds']['AdjDE_pct']:.2f}\n")
            f.write(f"  AdjEM z-score > {results['thresholds']['AdjEM_pct']:.2f}\n\n")
            
            # Write team count
            f.write(f"Number of teams meeting criteria: {results['team_count']}\n\n")
            
            # Write team details header
            f.write(f"Teams meeting {round_name} criteria (sorted by probability):\n")
            f.write(f"{'Team':<35} {'Conference':<15} {'Prob':>8} {'Imp Odds':>8} {'Sim Score':>10} {'AdjEM z':>8} {'AdjOE z':>8} {'AdjDE z':>8}\n")
            f.write("-" * 108 + "\n")
            
            # Write team details
            if results['qualifying_teams'] is not None:
                # Add implied odds to qualifying teams
                results['qualifying_teams']['implied_odds'] = results['qualifying_teams']['normalized_probability'].apply(calculate_implied_odds)
                
                for _, team in results['qualifying_teams'].iterrows():
                    f.write(f"{team['TeamName']:<35} {team['CONF']:<15} {team['normalized_probability']:>8.3f} "
                           f"{'+' + str(team['implied_odds']) if team['implied_odds'] > 0 else team['implied_odds']:>9} {team['similarity_score']:>10.3f} {team['AdjEM z-score']:>8.2f} "
                           f"{team['AdjOE z-score']:>8.2f} {team['AdjDE z-score']:>8.2f}\n")
            
            # Write correlation weights
            f.write("\nMetric Correlations with Wins (weights):\n")
            f.write(f"AdjOE: {results['correlations']['oe']:.3f}\n")
            f.write(f"AdjDE: {results['correlations']['de']:.3f}\n")
            f.write(f"AdjEM: {results['correlations']['em']:.3f}\n")
            
            # Write target sum info
            target_sum = next(ts for rd, rn, _, ts in ROUNDS if rn == round_name)
            total_prob = results['qualifying_teams']['normalized_probability'].sum() if results['qualifying_teams'] is not None else 0
            f.write(f"Total probability sum: {total_prob:.3f} (target: {target_sum})\n")
            f.write("-" * 108 + "\n\n")
        
        print(f"\nExported all results to tournament_analysis_results.txt")

    return round_results

if __name__ == "__main__":
    main()