"""
UFC Fight Analysis Module

This module contains classes and functions for processing and analyzing UFC fight data.
It handles data loading, preprocessing, feature engineering, and dataset preparation
for machine learning. Includes integrated data leakage verification and advanced features.
"""

import warnings
from pathlib import Path
import pandas as pd
import numpy as np
from typing import List, Tuple, Dict, Optional, Union
import os
from datetime import datetime
from src.data_processing.features.Elo import calculate_elo_ratings
from src.data_processing.features.helper import DataUtils, OddsUtils, FighterUtils, DateUtils

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)


class FightDataProcessor:
    """Process and transform UFC fight data for analysis."""

    def __init__(self, data_dir: str = "../../../data", enable_verification: bool = True):
        """Initialize the processor with data directory."""
        module_dir = Path(__file__).resolve().parent
        repo_root = module_dir.parents[2]

        candidate_dir = Path(data_dir).expanduser()
        if not candidate_dir.is_absolute():
            module_relative = (module_dir / candidate_dir).resolve(strict=False)
            repo_relative = (repo_root / candidate_dir).resolve(strict=False)
            if module_relative.exists():
                candidate_dir = module_relative
            elif repo_relative.exists():
                candidate_dir = repo_relative
            else:
                candidate_dir = module_relative

        self.data_dir = candidate_dir
        self.utils = DataUtils()
        self.odds_utils = OddsUtils(data_dir=self.data_dir)
        self.fighter_utils = FighterUtils(enable_verification=enable_verification)
        self.enable_verification = enable_verification

    def _load_csv(self, filepath: str) -> pd.DataFrame:
        """Load a CSV file into a DataFrame."""
        filepath = Path(filepath.replace('/', os.sep))

        if not filepath.is_absolute():
            filepath = self.data_dir / filepath

        filepath = filepath.expanduser()
        if not filepath.exists():
            raise FileNotFoundError(f"CSV file not found at {filepath}")

        return pd.read_csv(filepath)

    def _save_csv(self, df: pd.DataFrame, filepath: str) -> None:
        """Save DataFrame to CSV file."""
        filepath = Path(filepath.replace('/', os.sep))

        if not filepath.is_absolute():
            filepath = self.data_dir / filepath

        filepath = filepath.expanduser()
        filepath.parent.mkdir(parents=True, exist_ok=True)

        df.to_csv(filepath, index=False)
        print(f"Saved to {filepath}")

    def combine_rounds_stats(self, file_path: str) -> pd.DataFrame:
        """Process round-level fight data into fighter career statistics."""
        print("Loading and preprocessing data...")
        ufc_stats = self._load_csv(file_path)
        fighter_stats = self._load_csv('raw/ufc_fighter_tott.csv')
        ufc_stats = self.utils.preprocess_data(ufc_stats, fighter_stats)

        numeric_columns = self._get_numeric_columns(ufc_stats)

        print("Aggregating stats...")
        max_round_data = ufc_stats.groupby('id').agg({
            'last_round': 'max',
            'time': 'max'
        }).reset_index()

        aggregated_stats = ufc_stats.groupby(['id', 'fighter'])[numeric_columns].sum().reset_index()
        aggregated_stats = self._calculate_basic_rates(aggregated_stats)

        non_numeric_data = self._extract_non_numeric_data(ufc_stats)

        print("Merging aggregated stats with non-numeric data...")
        merged_stats = pd.merge(aggregated_stats, non_numeric_data, on=['id', 'fighter'], how='left')
        merged_stats = pd.merge(merged_stats, max_round_data, on='id', how='left')

        print("Calculating career stats...")
        final_stats = merged_stats.groupby('fighter', group_keys=False).apply(
            lambda x: self.fighter_utils.aggregate_fighter_stats(x, numeric_columns)
        )

        final_stats = self._calculate_per_minute_stats(final_stats)
        final_stats = self._calculate_additional_rates(final_stats)
        final_stats = self._filter_unwanted_results(final_stats)
        final_stats = self._factorize_categorical_columns(final_stats)
        final_stats = self.odds_utils.process_odds_data(final_stats)

        columns_to_drop = ['new_Open', 'new_Closing Range Start', 'new_Closing Range End', 'new_Movement', 'dob']
        final_stats = final_stats.drop(columns=columns_to_drop, errors='ignore')

        duplicate_columns = final_stats.columns[final_stats.columns.duplicated()]
        final_stats = final_stats.loc[:, ~final_stats.columns.duplicated()]
        if len(duplicate_columns) > 0:
            print(f"Dropped duplicate columns: {list(duplicate_columns)}")

        print("Calculating additional stats...")
        final_stats = final_stats.sort_values(['fighter', 'fight_date'])

        final_stats = final_stats.groupby('fighter', group_keys=False).apply(
            self.fighter_utils.calculate_experience_and_days
        )
        final_stats = final_stats.groupby('fighter', group_keys=False).apply(
            self.fighter_utils.update_streaks
        )
        final_stats['days_since_last_fight'] = final_stats['days_since_last_fight'].fillna(0)

        print("Calculating takedowns and knockdowns per 15 minutes...")
        final_stats = self.fighter_utils.calculate_time_based_stats(final_stats)

        print("Calculating total fights, wins, and losses...")
        final_stats = final_stats.groupby('fighter', group_keys=False).apply(
            self.fighter_utils.calculate_total_fight_stats
        )

        print("Calculating fighting styles...")
        final_stats = final_stats.groupby('fighter', group_keys=False).apply(
            self.fighter_utils.calculate_fighting_style
        )

        if self.enable_verification:
            self.fighter_utils.print_verification_summary()

        print("Saving processed data...")
        self._save_csv(final_stats, 'processed/combined_rounds.csv')

        return final_stats

    def _get_numeric_columns(self, df: pd.DataFrame) -> List[str]:
        """Extract relevant numeric columns for aggregation."""
        numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        numeric_columns = [col for col in numeric_columns if col not in ['id', 'last_round', 'age']]
        if 'time' not in numeric_columns:
            numeric_columns.append('time')
        return numeric_columns

    def _calculate_basic_rates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate basic strike and takedown rates."""
        df['significant_strikes_rate'] = self.utils.safe_divide(
            df['significant_strikes_landed'],
            df['significant_strikes_attempted']
        )
        df['takedown_rate'] = self.utils.safe_divide(
            df['takedown_successful'],
            df['takedown_attempted']
        )
        return df

    def _extract_non_numeric_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract non-numeric columns from the DataFrame."""
        non_numeric_columns = df.select_dtypes(exclude=['int64', 'float64']).columns.difference(
            ['id', 'fighter']
        )
        return df.drop_duplicates(subset=['id', 'fighter'])[
            ['id', 'fighter', 'age'] + list(non_numeric_columns)
        ]

    def _calculate_per_minute_stats(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate per-minute statistics."""
        df['fight_duration_minutes'] = df['time'] / 60
        for col in ['significant_strikes_landed', 'significant_strikes_attempted',
                    'total_strikes_landed', 'total_strikes_attempted']:
            df[f'{col}_per_min'] = self.utils.safe_divide(df[col], df['fight_duration_minutes'])
        return df

    def _calculate_additional_rates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate additional rate statistics."""
        df["total_strikes_rate"] = self.utils.safe_divide(
            df["total_strikes_landed"],
            df["total_strikes_attempted"]
        )
        df["combined_success_rate"] = (df["takedown_rate"] + df["total_strikes_rate"]) / 2
        return df

    def _filter_unwanted_results(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter out unwanted fight results."""
        df = df[~df['winner'].isin(['NC/NC', 'D/D'])]
        df = df[~df['result'].isin(['DQ', 'DQ ', 'Could Not Continue ', 'Overturned ', 'Other '])]
        return df

    def _factorize_categorical_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert categorical columns to numeric codes and print the mapping."""
        for column in ['result', 'winner', 'scheduled_rounds']:
            df[column], unique = pd.factorize(df[column])
            mapping = {index: label for index, label in enumerate(unique)}
            print(f"Mapping for {column}: {mapping}")
        return df

    def combine_fighters_stats(self, file_path: str) -> pd.DataFrame:
        """
        Create pairwise fighter statistics for all fights.
        FEATURE 4: Calculate defensive metrics using opponent data.
        FEATURE 1: Track opponent quality metrics.
        """
        df = self._load_csv(file_path)

        df = df.drop(columns=[col for col in df.columns if 'event' in col.lower()])
        df = df.sort_values(by=['id', 'fighter'])

        # Create mirrored fight pairs
        fights_dict = {}
        for _, row in df.iterrows():
            fight_id = row['id']
            fights_dict.setdefault(fight_id, []).append(row)

        combined_fights = []
        skipped_fights = 0

        for fight_id, fighters in fights_dict.items():
            if len(fighters) == 2:
                fighter_1, fighter_2 = fighters
                original = pd.concat([pd.Series(fighter_1), pd.Series(fighter_2).add_suffix('_b')])
                mirrored = pd.concat([pd.Series(fighter_2), pd.Series(fighter_1).add_suffix('_b')])
                combined_fights.extend([original, mirrored])
            else:
                skipped_fights += 1

        if skipped_fights > 0:
            print(f"Skipped {skipped_fights} fights with missing fighter data")

        final_combined_df = pd.DataFrame(combined_fights).reset_index(drop=True)

        # === FEATURE 4: Calculate Defensive Metrics ===
        print("Calculating defensive metrics...")
        final_combined_df = self._calculate_defensive_stats(final_combined_df)

        # === FEATURE 1: Calculate Opponent Quality Metrics ===
        print("Calculating opponent quality metrics...")
        final_combined_df = self._calculate_opponent_quality_features(final_combined_df)

        # Calculate differential and ratio features
        final_combined_df = self._calculate_differential_and_ratio_features(final_combined_df)

        final_combined_df = final_combined_df[~final_combined_df['winner'].isin(['NC', 'D'])]
        final_combined_df['fight_date'] = pd.to_datetime(final_combined_df['fight_date'])
        final_combined_df = final_combined_df.sort_values(
            by=['fighter', 'fight_date'],
            ascending=[True, True]
        )

        self._save_csv(final_combined_df, 'processed/combined_sorted_fighter_stats.csv')

        return final_combined_df

    def _calculate_defensive_stats(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        FEATURE 4: Calculate defensive statistics using opponent data.
        Now that we have _b columns, we can calculate how well each fighter defends.
        """
        # Significant strikes absorbed per minute (opponent's strikes that landed on us)
        df['sig_strikes_absorbed_per_min'] = self.utils.safe_divide(
            df['significant_strikes_landed_b'],
            df['fight_duration_minutes']
        )

        # Total strikes absorbed per minute
        df['total_strikes_absorbed_per_min'] = self.utils.safe_divide(
            df['total_strikes_landed_b'],
            df['fight_duration_minutes']
        )

        # Takedown defense rate (percentage of opponent's takedowns we stuffed)
        df['takedown_defense_rate'] = 1 - self.utils.safe_divide(
            df['takedown_successful_b'],
            df['takedown_attempted_b']
        )

        # Strike defense rate (percentage of opponent's strikes we avoided)
        df['strike_defense_rate'] = 1 - self.utils.safe_divide(
            df['significant_strikes_landed_b'],
            df['significant_strikes_attempted_b']
        )

        # Damage ratio (our offense vs their offense)
        df['damage_ratio'] = self.utils.safe_divide(
            df['significant_strikes_landed'],
            df['sig_strikes_absorbed_per_min'] + 1  # +1 to avoid division issues
        )

        # Submission defense (inverse of submission losses)
        df['submission_defense'] = (df['losses_by_submission'] == 0).astype(float)

        # Control differential (our control time vs opponent's)
        if 'control' in df.columns and 'control_b' in df.columns:
            df['control_advantage'] = df['control'] - df['control_b']

        return df

    def _calculate_opponent_quality_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        FEATURE 1: Calculate opponent quality metrics.
        Uses opponent data that's already time-frozen in the _b columns.
        """
        # Opponent's Elo at time of fight (quality indicator)
        if 'pre_fight_elo_b' in df.columns:
            df['opponent_quality'] = df['pre_fight_elo_b']

            # Quality-adjusted win (beating a strong opponent counts more)
            df['quality_adjusted_win'] = df['winner'] * (df['pre_fight_elo_b'] / 1500)

        # Opponent's recent form
        if 'ewm_win_rate_b' in df.columns:
            df['opponent_recent_form'] = df['ewm_win_rate_b']

        # Opponent's momentum
        if 'win_streak_b' in df.columns and 'loss_streak_b' in df.columns:
            df['opponent_momentum'] = df['win_streak_b'] - df['loss_streak_b']

        # Opponent's finish rate
        if 'ewm_finish_rate_b' in df.columns:
            df['opponent_finish_threat'] = df['ewm_finish_rate_b']

        return df

    def _calculate_differential_and_ratio_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate differential and ratio features between fighter pairs."""
        base_columns = [
            'knockdowns', 'significant_strikes_landed', 'significant_strikes_attempted',
            'significant_strikes_rate', 'total_strikes_landed', 'total_strikes_attempted',
            'takedown_successful', 'takedown_attempted', 'takedown_rate', 'submission_attempt',
            'reversals', 'head_landed', 'head_attempted', 'body_landed', 'body_attempted',
            'leg_landed', 'leg_attempted', 'distance_landed', 'distance_attempted',
            'clinch_landed', 'clinch_attempted', 'ground_landed', 'ground_attempted'
        ]

        # Add new feature columns
        new_feature_columns = [
            'ewm_win_rate', 'ewm_finish_rate', 'ewm_strike_accuracy',
            'win_rate_trajectory', 'finish_rate_trajectory', 'momentum',
            'layoff_penalty', 'rushed_return', 'ring_rust',
            'striker_score', 'grappler_score', 'pressure_score', 'style_confidence',
            'sig_strikes_absorbed_per_min', 'takedown_defense_rate', 'strike_defense_rate',
            'damage_ratio', 'opponent_quality', 'opponent_recent_form', 'opponent_momentum'
        ]

        other_columns = [
            'open_odds', 'closing_range_start', 'closing_range_end', 'pre_fight_elo',
            'years_of_experience', 'win_streak', 'loss_streak', 'days_since_last_fight',
            'significant_strikes_landed_per_min', 'significant_strikes_attempted_per_min',
            'total_strikes_landed_per_min', 'total_strikes_attempted_per_min', 'takedowns_per_15min',
            'knockdowns_per_15min', 'total_fights', 'total_wins', 'total_losses',
            'wins_by_ko', 'losses_by_ko', 'wins_by_submission', 'losses_by_submission', 'wins_by_decision',
            'losses_by_decision', 'win_rate_by_ko', 'loss_rate_by_ko', 'win_rate_by_submission',
            'loss_rate_by_submission', 'win_rate_by_decision', 'loss_rate_by_decision'
        ]

        columns_to_process = (
            base_columns +
            [f"{col}_career" for col in base_columns] +
            [f"{col}_career_avg" for col in base_columns] +
            other_columns +
            new_feature_columns
        )

        # Calculate differential features
        diff_features = {}
        for col in columns_to_process:
            if col in df.columns and f"{col}_b" in df.columns:
                diff_features[f"{col}_diff"] = df[col] - df[f"{col}_b"]

        # Calculate ratio features
        ratio_features = {}
        for col in columns_to_process:
            if col in df.columns and f"{col}_b" in df.columns:
                ratio_features[f"{col}_ratio"] = self.utils.safe_divide(df[col], df[f"{col}_b"])

        return pd.concat([df, pd.DataFrame(diff_features), pd.DataFrame(ratio_features)], axis=1)


class MatchupProcessor:
    """Process and prepare matchup data for predictive modeling."""

    def __init__(self, data_dir: str = "../../../data", enable_verification: bool = True):
        """Initialize the processor with data directory."""
        self.fight_processor = FightDataProcessor(data_dir, enable_verification=enable_verification)
        self.data_dir = self.fight_processor.data_dir
        self.utils = DataUtils()
        self.odds_utils = OddsUtils(data_dir=self.data_dir)
        self.enable_verification = enable_verification
        self.leakage_warnings = []

    def create_matchup_data(self, file_path: str, tester: int, include_names: bool = False) -> pd.DataFrame:
        """Create matchup data for predictive modeling with advanced features."""
        print(f"Creating matchup data with {tester} recent fights...")
        df = self.fight_processor._load_csv(file_path)
        n_past_fights = 6 - tester

        columns_to_exclude = [
            'fighter', 'id', 'fighter_b', 'fight_date', 'fight_date_b',
            'result', 'winner', 'weight_class', 'scheduled_rounds',
            'result_b', 'winner_b', 'weight_class_b', 'scheduled_rounds_b',
            'primary_style', 'primary_style_b'  # Categorical, will be handled separately
        ]

        features_to_include = [
            col for col in df.columns if col not in columns_to_exclude and
            col != 'age' and not col.endswith('_age')
        ]

        method_columns = ['winner']

        matchup_data = self._process_matchups(
            df, features_to_include, method_columns, n_past_fights, tester, include_names
        )

        column_names = self._generate_column_names(
            features_to_include, method_columns, n_past_fights, tester, include_names
        )
        matchup_df = pd.DataFrame(matchup_data, columns=column_names)

        matchup_df = matchup_df.drop(columns=['fight_date'], errors='ignore')
        matchup_df.columns = [self.utils.rename_columns_general(col) for col in matchup_df.columns]

        # Calculate additional matchup-specific features
        matchup_df = self._calculate_matchup_features(matchup_df, features_to_include, n_past_fights)
        matchup_df = self._add_quick_win_features(matchup_df, n_past_fights)

        if self.enable_verification and self.leakage_warnings:
            print("\n" + "="*60)
            print("MATCHUP DATA LEAKAGE WARNINGS")
            print("="*60)
            for warning in self.leakage_warnings[:10]:
                print(warning)
            if len(self.leakage_warnings) > 10:
                print(f"... and {len(self.leakage_warnings) - 10} more warnings")
            print("="*60)

        output_filename = f'matchup data/matchup_data_{n_past_fights}_avg{"_name" if include_names else ""}.csv'
        self.fight_processor._save_csv(matchup_df, output_filename)

        return matchup_df

    def _add_quick_win_features(self, df: pd.DataFrame, n_fights: int) -> pd.DataFrame:
        """
        Add quick-win features that are easy to implement and high value.
        Features 2, 6, and simple calculations.
        """
        # Feature: Finish Rate
        finish_cols_a = ['wins_by_ko_fighter_a_avg_last_' + str(n_fights),
                         'wins_by_submission_fighter_a_avg_last_' + str(n_fights)]
        finish_cols_b = ['wins_by_ko_fighter_b_avg_last_' + str(n_fights),
                         'wins_by_submission_fighter_b_avg_last_' + str(n_fights)]

        if all(col in df.columns for col in finish_cols_a + finish_cols_b):
            df['finish_rate_a'] = (
                df[finish_cols_a[0]] + df[finish_cols_a[1]]
            ) / df[f'total_fights_fighter_a_avg_last_{n_fights}'].replace(0, 1)

            df['finish_rate_b'] = (
                df[finish_cols_b[0]] + df[finish_cols_b[1]]
            ) / df[f'total_fights_fighter_b_avg_last_{n_fights}'].replace(0, 1)

            df['finish_rate_diff'] = df['finish_rate_a'] - df['finish_rate_b']

        # Feature: Activity Penalty (layoff impact)
        if 'current_fight_days_since_last_a' in df.columns and 'current_fight_days_since_last_b' in df.columns:
            df['activity_penalty_a'] = np.exp(
                -((df['current_fight_days_since_last_a'] - 135) ** 2) / (2 * 90 ** 2)
            )
            df['activity_penalty_b'] = np.exp(
                -((df['current_fight_days_since_last_b'] - 135) ** 2) / (2 * 90 ** 2)
            )
            df['activity_advantage'] = df['activity_penalty_a'] - df['activity_penalty_b']

        # Feature: Recent Momentum (already have the data)
        if all(col in df.columns for col in ['current_fight_win_streak_a', 'current_fight_loss_streak_a',
                                              'current_fight_win_streak_b', 'current_fight_loss_streak_b']):
            df['momentum_a'] = (
                df['current_fight_win_streak_a'] -
                df['current_fight_loss_streak_a']
            )
            df['momentum_b'] = (
                df['current_fight_win_streak_b'] -
                df['current_fight_loss_streak_b']
            )
            df['momentum_diff'] = df['momentum_a'] - df['momentum_b']

        return df

    def _process_matchups(
        self,
        df: pd.DataFrame,
        features_to_include: List[str],
        method_columns: List[str],
        n_past_fights: int,
        tester: int,
        include_names: bool
    ) -> List[List]:
        """Process each fight to create matchup feature vectors."""
        matchup_data = []
        skipped_count = 0
        processed_count = 0
        partial_data_count = 0

        verification_sample_size = 5
        verification_counter = 0

        for idx, current_fight in df.iterrows():
            fighter_a_name = current_fight['fighter']
            fighter_b_name = current_fight['fighter_b']

            fighter_a_df = df[
                (df['fighter'] == fighter_a_name) &
                (df['fight_date'] < current_fight['fight_date'])
            ].sort_values(by='fight_date', ascending=False).head(n_past_fights)

            fighter_b_df = df[
                (df['fighter'] == fighter_b_name) &
                (df['fight_date'] < current_fight['fight_date'])
            ].sort_values(by='fight_date', ascending=False).head(n_past_fights)

            # Leakage verification
            if self.enable_verification and verification_counter < verification_sample_size:
                if len(fighter_a_df) > 0:
                    latest_past_fight_a = fighter_a_df.iloc[0]
                    if latest_past_fight_a['fight_date'] >= current_fight['fight_date']:
                        warning = f"CRITICAL LEAKAGE: Fighter {fighter_a_name} using future fight data!"
                        self.leakage_warnings.append(warning)
                        print(warning)

                if len(fighter_b_df) > 0:
                    latest_past_fight_b = fighter_b_df.iloc[0]
                    if latest_past_fight_b['fight_date'] >= current_fight['fight_date']:
                        warning = f"CRITICAL LEAKAGE: Fighter {fighter_b_name} using future fight data!"
                        self.leakage_warnings.append(warning)
                        print(warning)

                verification_counter += 1

            if len(fighter_a_df) == 0 or len(fighter_b_df) == 0:
                skipped_count += 1
                continue

            has_partial_data = len(fighter_a_df) < n_past_fights or len(fighter_b_df) < n_past_fights
            if has_partial_data:
                partial_data_count += 1

            fighter_a_features = fighter_a_df[features_to_include].mean().values
            fighter_b_features = fighter_b_df[features_to_include].mean().values

            num_a_results = min(len(fighter_a_df), tester)
            num_b_results = min(len(fighter_b_df), tester)

            results_fighter_a = fighter_a_df[['result', 'winner', 'weight_class', 'scheduled_rounds']].head(
                num_a_results).values.flatten() if num_a_results > 0 else np.array([])

            results_fighter_b = fighter_b_df[['result_b', 'winner_b', 'weight_class_b', 'scheduled_rounds_b']].head(
                num_b_results).values.flatten() if num_b_results > 0 else np.array([])

            results_fighter_a = np.pad(
                results_fighter_a,
                (0, tester * 4 - len(results_fighter_a)),
                'constant',
                constant_values=np.nan
            )
            results_fighter_b = np.pad(
                results_fighter_b,
                (0, tester * 4 - len(results_fighter_b)),
                'constant',
                constant_values=np.nan
            )

            labels = current_fight[method_columns].values

            current_fight_odds, current_fight_odds_diff, current_fight_odds_ratio = self._process_fight_odds(
                current_fight['open_odds'], current_fight['open_odds_b']
            )

            current_fight_closing_odds, current_fight_closing_odds_diff, current_fight_closing_odds_ratio = self._process_fight_odds(
                current_fight['closing_range_end'], current_fight['closing_range_end_b']
            )

            current_fight_closing_open_diff_a = current_fight['closing_range_end'] - current_fight['open_odds']
            current_fight_closing_open_diff_b = current_fight['closing_range_end_b'] - current_fight['open_odds_b']

            current_fight_ages = [current_fight['age'], current_fight['age_b']]
            current_fight_age_diff = current_fight['age'] - current_fight['age_b']
            current_fight_age_ratio = self.utils.safe_divide(current_fight['age'], current_fight['age_b'])

            elo_stats, elo_ratio = self._process_elo_stats(current_fight)
            other_stats = self._process_other_stats(current_fight)

            combined_features = np.concatenate([
                fighter_a_features, fighter_b_features, results_fighter_a, results_fighter_b,
                current_fight_odds, [current_fight_odds_diff, current_fight_odds_ratio],
                current_fight_closing_odds, [current_fight_closing_odds_diff, current_fight_closing_odds_ratio,
                                            current_fight_closing_open_diff_a, current_fight_closing_open_diff_b],
                current_fight_ages, [current_fight_age_diff, current_fight_age_ratio],
                elo_stats, [elo_ratio], other_stats
            ])
            combined_row = np.concatenate([combined_features, labels])

            most_recent_date_a = fighter_a_df['fight_date'].max() if len(fighter_a_df) > 0 else None
            most_recent_date_b = fighter_b_df['fight_date'].max() if len(fighter_b_df) > 0 else None
            most_recent_date = max(most_recent_date_a, most_recent_date_b) if most_recent_date_a and most_recent_date_b else most_recent_date_a or most_recent_date_b
            current_fight_date = current_fight['fight_date']

            if not include_names:
                matchup_data.append([most_recent_date] + combined_row.tolist() + [current_fight_date])
            else:
                matchup_data.append(
                    [fighter_a_name, fighter_b_name, most_recent_date] + combined_row.tolist() + [current_fight_date]
                )

            processed_count += 1

        print(f"Processed {processed_count} matchups (including {partial_data_count} with partial fight history)")
        print(f"Skipped {skipped_count} matchups where at least one fighter had no previous fights")

        return matchup_data

    def _process_fight_odds(self, odds_a: float, odds_b: float) -> Tuple[List[float], float, float]:
        """Process betting odds for a fight."""
        return self.odds_utils.process_odds_pair(odds_a, odds_b)

    def _process_elo_stats(self, current_fight: pd.Series) -> Tuple[List[float], float]:
        """Process Elo rating statistics."""
        elo_a = current_fight['pre_fight_elo']
        elo_b = current_fight['pre_fight_elo_b']
        elo_diff = current_fight['pre_fight_elo_diff']

        a_win_prob = 1 / (1 + 10 ** ((elo_b - elo_a) / 400))
        b_win_prob = 1 / (1 + 10 ** ((elo_a - elo_b) / 400))

        elo_stats = [elo_a, elo_b, elo_diff, a_win_prob, b_win_prob]
        elo_ratio = self.utils.safe_divide(elo_a, elo_b)

        return elo_stats, elo_ratio

    def _process_other_stats(self, current_fight: pd.Series) -> List[float]:
        """Process other fighter statistics."""
        win_streak_a = current_fight['win_streak']
        win_streak_b = current_fight['win_streak_b']
        win_streak_diff = win_streak_a - win_streak_b
        win_streak_ratio = self.utils.safe_divide(win_streak_a, win_streak_b)

        loss_streak_a = current_fight['loss_streak']
        loss_streak_b = current_fight['loss_streak_b']
        loss_streak_diff = loss_streak_a - loss_streak_b
        loss_streak_ratio = self.utils.safe_divide(loss_streak_a, loss_streak_b)

        exp_a = current_fight['years_of_experience']
        exp_b = current_fight['years_of_experience_b']
        exp_diff = exp_a - exp_b
        exp_ratio = self.utils.safe_divide(exp_a, exp_b)

        days_since_a = current_fight['days_since_last_fight']
        days_since_b = current_fight['days_since_last_fight_b']
        days_since_diff = days_since_a - days_since_b
        days_since_ratio = self.utils.safe_divide(days_since_a, days_since_b)

        return [
            win_streak_a, win_streak_b, win_streak_diff, win_streak_ratio,
            loss_streak_a, loss_streak_b, loss_streak_diff, loss_streak_ratio,
            exp_a, exp_b, exp_diff, exp_ratio,
            days_since_a, days_since_b, days_since_diff, days_since_ratio
        ]

    def _generate_column_names(
        self,
        features_to_include: List[str],
        method_columns: List[str],
        n_past_fights: int,
        tester: int,
        include_names: bool
    ) -> List[str]:
        """Generate column names for the matchup DataFrame."""
        results_columns = []
        for i in range(1, tester + 1):
            results_columns += [
                f"result_fight_{i}", f"winner_fight_{i}", f"weight_class_fight_{i}", f"scheduled_rounds_fight_{i}",
                f"result_b_fight_{i}", f"winner_b_fight_{i}", f"weight_class_b_fight_{i}",
                f"scheduled_rounds_b_fight_{i}"
            ]

        new_columns = [
            'current_fight_pre_fight_elo_a', 'current_fight_pre_fight_elo_b', 'current_fight_pre_fight_elo_diff',
            'current_fight_pre_fight_elo_a_win_chance', 'current_fight_pre_fight_elo_b_win_chance',
            'current_fight_pre_fight_elo_ratio', 'current_fight_win_streak_a', 'current_fight_win_streak_b',
            'current_fight_win_streak_diff', 'current_fight_win_streak_ratio', 'current_fight_loss_streak_a',
            'current_fight_loss_streak_b', 'current_fight_loss_streak_diff', 'current_fight_loss_streak_ratio',
            'current_fight_years_experience_a', 'current_fight_years_experience_b',
            'current_fight_years_experience_diff',
            'current_fight_years_experience_ratio', 'current_fight_days_since_last_a',
            'current_fight_days_since_last_b', 'current_fight_days_since_last_diff',
            'current_fight_days_since_last_ratio'
        ]

        base_columns = ['fight_date'] if not include_names else ['fighter_a', 'fighter_b', 'fight_date']

        feature_columns = (
            [f"{feature}_fighter_avg_last_{n_past_fights}" for feature in features_to_include] +
            [f"{feature}_fighter_b_avg_last_{n_past_fights}" for feature in features_to_include]
        )

        odds_age_columns = [
            'current_fight_open_odds', 'current_fight_open_odds_b', 'current_fight_open_odds_diff',
            'current_fight_open_odds_ratio',
            'current_fight_closing_odds', 'current_fight_closing_odds_b', 'current_fight_closing_odds_diff',
            'current_fight_closing_odds_ratio', 'current_fight_closing_open_diff_a',
            'current_fight_closing_open_diff_b',
            'current_fight_age', 'current_fight_age_b', 'current_fight_age_diff', 'current_fight_age_ratio'
        ]

        return (
            base_columns + feature_columns + results_columns + odds_age_columns + new_columns +
            [f"{method}" for method in method_columns] + ['current_fight_date']
        )

    def _calculate_matchup_features(
        self,
        df: pd.DataFrame,
        features_to_include: List[str],
        n_past_fights: int
    ) -> pd.DataFrame:
        """Calculate additional differential and ratio features."""
        diff_columns = {}
        ratio_columns = {}

        for feature in features_to_include:
            col_a = f"{feature}_fighter_a_avg_last_{n_past_fights}"
            col_b = f"{feature}_fighter_b_avg_last_{n_past_fights}"

            if col_a in df.columns and col_b in df.columns:
                diff_columns[f"matchup_{feature}_diff_avg_last_{n_past_fights}"] = df[col_a] - df[col_b]
                ratio_columns[f"matchup_{feature}_ratio_avg_last_{n_past_fights}"] = self.utils.safe_divide(
                    df[col_a], df[col_b]
                )

        return pd.concat([df, pd.DataFrame(diff_columns), pd.DataFrame(ratio_columns)], axis=1)

    def split_train_val_test(
        self,
        matchup_data_file: str,
        start_date: str,
        end_date: str,
        years_back: int
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Split matchup data into training, validation, and test sets ensuring no date overlap."""
        print(f"Splitting data from {start_date} to {end_date} with {years_back} years history...")
        matchup_df = self.fight_processor._load_csv(matchup_data_file)

        matchup_df['current_fight_date'] = pd.to_datetime(matchup_df['current_fight_date'])
        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)
        years_before = start_date - pd.DateOffset(years=years_back)

        test_data = matchup_df[
            (matchup_df['current_fight_date'] >= start_date) &
            (matchup_df['current_fight_date'] <= end_date)
        ].copy()

        remaining_data = matchup_df[
            (matchup_df['current_fight_date'] >= years_before) &
            (matchup_df['current_fight_date'] < start_date)
        ].copy()

        remaining_data = remaining_data.sort_values(by='current_fight_date', ascending=True)

        unique_dates = sorted(remaining_data['current_fight_date'].unique())
        n_dates = len(unique_dates)
        split_date_idx = int(n_dates * 0.8)

        if split_date_idx < n_dates:
            cutoff_date = unique_dates[split_date_idx]
            train_data = remaining_data[remaining_data['current_fight_date'] < cutoff_date].copy()
            val_data = remaining_data[remaining_data['current_fight_date'] >= cutoff_date].copy()
        else:
            train_data = remaining_data.copy()
            val_data = pd.DataFrame()

        print(f"Split using cutoff date: {cutoff_date if split_date_idx < n_dates else 'N/A'}")

        test_data = self._remove_duplicate_fights(test_data, random=False)

        train_data = train_data.sort_values(by='current_fight_date', ascending=True)
        val_data = val_data.sort_values(by='current_fight_date', ascending=True) if not val_data.empty else val_data
        test_data = test_data.sort_values(by=['current_fight_date', 'fighter_a'], ascending=[True, True])

        removed_features: List[str] = []
        if not train_data.empty:
            train_data, removed_features = self.utils.remove_correlated_features(
                train_data,
                correlation_threshold=0.95,
                protected_columns=[
                    'winner',
                    'current_fight_open_odds',
                    'current_fight_open_odds_b',
                    'current_fight_open_odds_diff',
                    'current_fight_open_odds_ratio',
                    'current_fight_closing_odds',
                    'current_fight_closing_odds_b',
                    'current_fight_closing_odds_diff',
                    'current_fight_closing_odds_ratio',
                    'current_fight_closing_range_end',
                    'current_fight_closing_range_end_b',
                    'current_fight_closing_open_diff_a',
                    'current_fight_closing_open_diff_b'
                ]
            )

            if removed_features:
                val_data = val_data.drop(columns=removed_features, errors='ignore')
                test_data = test_data.drop(columns=removed_features, errors='ignore')

        if self.enable_verification:
            print("\n" + "=" * 60)
            print("LEAKAGE CHECK #5: Train/Test Split Verification")
            print("=" * 60)

            if not train_data.empty:
                print(f"Train date range: {train_data['current_fight_date'].min()} to {train_data['current_fight_date'].max()}")
            if not val_data.empty:
                print(f"Val date range: {val_data['current_fight_date'].min()} to {val_data['current_fight_date'].max()}")
            if not test_data.empty:
                print(f"Test date range: {test_data['current_fight_date'].min()} to {test_data['current_fight_date'].max()}")

            overlap_issues = []
            if not train_data.empty and not val_data.empty:
                if train_data['current_fight_date'].max() >= val_data['current_fight_date'].min():
                    overlap_issues.append("Train and validation dates overlap")
                    print("LEAKAGE: Train and validation dates overlap!")

            if not val_data.empty and not test_data.empty:
                if val_data['current_fight_date'].max() >= test_data['current_fight_date'].min():
                    overlap_issues.append("Validation and test dates overlap")
                    print("LEAKAGE: Validation and test dates overlap!")

            if not overlap_issues:
                print("No date overlap between train/val/test sets")

            print("=" * 60)

        self.fight_processor._save_csv(train_data, 'train_test/train_data.csv')
        self.fight_processor._save_csv(val_data, 'train_test/val_data.csv')
        self.fight_processor._save_csv(test_data, 'train_test/test_data.csv')

        with open(os.path.join(self.fight_processor.data_dir, 'train_test/removed_features.txt'), 'w') as file:
            file.write(','.join(removed_features))

        print(f"Train set size: {len(train_data)}, Validation set size: {len(val_data)}, Test set size: {len(test_data)}")
        print(f"Removed {len(removed_features)} correlated features")

        return train_data, val_data, test_data

    def _remove_duplicate_fights(self, df: pd.DataFrame, random=True) -> pd.DataFrame:
        """Remove duplicate fights."""
        df = df.copy()

        df['fight_pair'] = df.apply(
            lambda row: tuple(sorted([row['fighter_a'], row['fighter_b']])) + (row['current_fight_date'],),
            axis=1
        )

        if random:
            df = df.sample(frac=1, random_state=42)
            df = df.drop_duplicates(subset=['fight_pair'], keep='first')
        else:
            result_rows = []
            for pair, group in df.groupby('fight_pair'):
                alpha_rows = group[group['fighter_a'] <= group['fighter_b']]
                if len(alpha_rows) > 0:
                    result_rows.append(alpha_rows.iloc[0])
                else:
                    result_rows.append(group.iloc[0])

            df = pd.DataFrame(result_rows)
            df = df.sort_values(by=['current_fight_date', 'fighter_a'], ascending=[True, True])

        return df.drop(columns=['fight_pair']).reset_index(drop=True)


def main():
    """Main execution function."""
    fight_processor = FightDataProcessor(enable_verification=True)
    matchup_processor = MatchupProcessor(data_dir=str(fight_processor.data_dir), enable_verification=True)

    print("Starting UFC data processing pipeline with advanced features and leakage verification...")

    fight_processor.combine_rounds_stats('processed/ufc_fight_processed.csv')

    combined_rounds_path = fight_processor.data_dir / 'processed' / 'combined_rounds.csv'
    calculate_elo_ratings(str(combined_rounds_path))

    fight_processor.combine_fighters_stats('processed/combined_rounds.csv')

    matchup_processor.create_matchup_data('processed/combined_sorted_fighter_stats.csv', 3, True)

    matchup_processor.split_train_val_test(
        'matchup data/matchup_data_3_avg_name.csv',
        '2025-01-01',
        '2025-12-31',
        15
    )

    print("\nData processing completed successfully with all advanced features!")


if __name__ == "__main__":
    start_time = datetime.now()
    main()
    end_time = datetime.now()
    print(f"\nTotal runtime: {end_time - start_time}")