"""
Create Individual Matchup - Simple Example
Creates a matchup between two fighters with odds and date specified.
Applies the same feature selection as training data.
"""
from src.data_processing.cleaning.data_cleaner import MatchupProcessor

# Initialize the processor
mp = MatchupProcessor(data_dir="../../../data")

# Create matchup with odds and date
print("=" * 80)
print("CREATING INDIVIDUAL MATCHUP WITH FEATURE SELECTION")
print("=" * 80)

matchup = mp.create_individual_matchup(
    fighter_a_name="Jeremiah Wells",
    fighter_b_name="Themba Gorimbo",
    reference_date="2025-11-01",  # UFC 307 date
    fighter_a_odds=-550,  # Pereira heavily favored
    fighter_b_odds=400,   # Rountree underdog
    fighter_a_closing_odds=-650,  # Line moved toward Pereira
    fighter_b_closing_odds=450,
    n_past_fights=3,
    n_detailed_results=3,
    apply_feature_selection=True,
)

print("\n" + "=" * 80)
print("MATCHUP SUMMARY")
print("=" * 80)

# Display key information
print(f"\nFighters: {matchup['fighter_a'].iloc[0]} vs {matchup['fighter_b'].iloc[0]}")
print(f"Total Features: {len(matchup.columns)}")

# Show odds features
print("\n--- ODDS INFORMATION ---")
odds_cols = [col for col in matchup.columns if 'odds' in col.lower()]
for col in odds_cols[:8]:  # Show first 8 odds columns
    print(f"{col}: {matchup[col].iloc[0]}")

# Show key stats
print("\n--- KEY STATISTICS ---")
key_stats = [
    'current_fight_age', 'current_fight_age_b',
    'current_fight_pre_fight_elo_a', 'current_fight_pre_fight_elo_b',
    'current_fight_win_streak_a', 'current_fight_win_streak_b',
    'current_fight_days_since_last_a', 'current_fight_days_since_last_b'
]

for stat in key_stats:
    if stat in matchup.columns:
        print(f"{stat}: {matchup[stat].iloc[0]}")

print("\n" + "=" * 80)
print("✓ Matchup saved to: data/train_test/individual_matchup.csv")
print("=" * 80)
print("\nFormat matches batch matchup data:")
print("  - Contains 'fighter_a', 'fighter_b' columns")
print("  - Contains 'winner' column (set to -1 for prediction)")
print("  - Contains 'current_fight_date' column")
print("  - Feature selection applied - columns match training data!")

print("\nReady for model predictions:")
print("  import pandas as pd")
print("  matchup = pd.read_csv('data/train_test/individual_matchup.csv')")
print("  # Drop non-feature columns before prediction")
print("  X = matchup.drop(['fighter_a', 'fighter_b', 'winner', 'current_fight_date'], axis=1)")
print("  prediction = model.predict(X)")
