"""
UFC Data Analysis Utilities

This module contains utility classes and functions for processing UFC fight data.
Includes integrated data leakage verification checks and advanced feature engineering.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Union, Any


class DataUtils:
    """General data processing utilities."""

    @staticmethod
    def safe_divide(
        numerator: Union[float, np.ndarray, pd.Series],
        denominator: Union[float, np.ndarray, pd.Series],
        default: float = 0
    ) -> Union[float, np.ndarray, pd.Series]:
        """
        Safely divide with protection against division by zero.

        Args:
            numerator: Value or array to divide
            denominator: Value or array to divide by
            default: Default value when denominator is zero

        Returns:
            Result of division with zeros replaced by default
        """
        if isinstance(numerator, (pd.Series, pd.DataFrame)) or isinstance(denominator, (pd.Series, pd.DataFrame)):
            result = pd.Series(numerator).div(pd.Series(denominator))
            return result.fillna(default).replace([np.inf, -np.inf], default)
        elif isinstance(numerator, np.ndarray) and isinstance(denominator, np.ndarray):
            result = np.zeros_like(numerator, dtype=float)
            mask = denominator != 0
            result[mask] = numerator[mask] / denominator[mask]
            result[~mask] = default
            return result
        else:
            return numerator / denominator if denominator != 0 else default

    def preprocess_data(self, ufc_stats: pd.DataFrame, fighter_stats: pd.DataFrame) -> pd.DataFrame:
        """Preprocess the UFC and fighter stats dataframes."""
        ufc_stats['fighter'] = ufc_stats['fighter'].astype(str).str.lower()
        ufc_stats['fight_date'] = pd.to_datetime(ufc_stats['fight_date'])
        fighter_stats['name'] = fighter_stats['FIGHTER'].astype(str).str.lower().str.strip()
        fighter_stats['dob'] = fighter_stats['DOB'].replace(['--', '', 'NA', 'N/A'], np.nan).apply(DateUtils.parse_date)

        ufc_stats = pd.merge(
            ufc_stats,
            fighter_stats[['name', 'dob']],
            left_on='fighter', right_on='name',
            how='left'
        )
        ufc_stats['age'] = (ufc_stats['fight_date'] - ufc_stats['dob']).dt.days / 365.25
        ufc_stats['age'] = ufc_stats['age'].fillna(np.nan).round().astype(float)
        ufc_stats.loc[ufc_stats['age'] < 0, 'age'] = np.nan

        ufc_stats = ufc_stats.drop(['round', 'location', 'name'], axis=1)
        # ufc_stats = ufc_stats[~ufc_stats['weight_class'].str.contains("Women's")]

        ufc_stats['time'] = (
                pd.to_datetime(ufc_stats['time'], format='%M:%S').dt.minute * 60 +
                pd.to_datetime(ufc_stats['time'], format='%M:%S').dt.second
        )

        return ufc_stats

    def rename_columns_general(self, col: str) -> str:
        """Rename columns for clarity."""
        if 'fighter' in col and not col.startswith('fighter'):
            if 'b_fighter_b' in col:
                return col.replace('b_fighter_b', 'fighter_b_opponent')
            elif 'b_fighter' in col:
                return col.replace('b_fighter', 'fighter_a_opponent')
            elif 'fighter' in col and 'fighter_b' not in col:
                return col.replace('fighter', 'fighter_a')
        return col

    def get_opponent(self, fighter: str, fight_id: str, ufc_stats: pd.DataFrame) -> Optional[str]:
        """Get a fighter's opponent for a specific fight."""
        fight_fighters = ufc_stats[ufc_stats['id'] == fight_id]['fighter'].unique()
        if len(fight_fighters) < 2:
            return None
        return fight_fighters[0] if fight_fighters[0] != fighter else fight_fighters[1]

    def remove_correlated_features(
        self,
        df: pd.DataFrame,
        correlation_threshold: float = 0.95,
        protected_columns: Optional[List[str]] = None
    ) -> Tuple[pd.DataFrame, List[str]]:
        """Remove highly correlated features from DataFrame."""
        protected_columns = protected_columns or []

        numeric_columns = df.select_dtypes(include=[np.number]).columns
        numeric_df = df[numeric_columns]

        corr_matrix = numeric_df.corr().abs()
        upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

        columns_to_drop = [
            column for column in upper_tri.columns
            if any(upper_tri[column] > correlation_threshold)
            and column not in protected_columns
        ]

        cleaned_df = df.drop(columns=columns_to_drop)
        return cleaned_df, columns_to_drop


class OddsUtils:
    """Utilities for processing betting odds."""

    def __init__(
        self,
        data_dir: Optional[Union[str, Path]] = None,
        odds_filename: Union[str, Path] = "processed/cleaned_fight_odds.csv"
    ) -> None:
        """Initialize the utility with an optional data directory and odds filename."""
        module_dir = Path(__file__).resolve().parent
        repo_root = module_dir.parents[2]

        if data_dir is None:
            base_dir = repo_root / "data"
        else:
            base_dir = Path(data_dir).expanduser()
            if not base_dir.is_absolute():
                module_relative = (module_dir / base_dir).resolve(strict=False)
                repo_relative = (repo_root / base_dir).resolve(strict=False)
                if module_relative.exists():
                    base_dir = module_relative
                elif repo_relative.exists():
                    base_dir = repo_relative
                else:
                    base_dir = module_relative

        self._data_dir = base_dir
        self._odds_filename = Path(odds_filename)

    def _resolve_odds_path(self, odds_filepath: Optional[Union[str, Path]] = None) -> Path:
        """Resolve the odds data path to an absolute location and validate it exists."""
        if odds_filepath is not None:
            candidate = Path(odds_filepath).expanduser()
            if not candidate.is_absolute():
                candidate = self._data_dir / candidate
        else:
            candidate = self._data_dir / self._odds_filename

        candidate = candidate.expanduser()
        if not candidate.exists():
            raise FileNotFoundError(
                f"Odds data file not found at {candidate}. "
                "Provide a valid path or ensure the data directory is correct."
            )

        return candidate

    @staticmethod
    def round_to_nearest_1(x: float) -> int:
        """Round to nearest integer."""
        return round(x)

    @staticmethod
    def calculate_complementary_odd(odd: float) -> float:
        """Calculate complementary betting odd."""
        if odd > 0:
            prob = 100 / (odd + 100)
        else:
            prob = abs(odd) / (abs(odd) + 100)

        complementary_prob = 1.045 - prob

        if complementary_prob >= 0.5:
            complementary_odd = -100 * complementary_prob / (1 - complementary_prob)
        else:
            complementary_odd = 100 * (1 - complementary_prob) / complementary_prob

        return OddsUtils.round_to_nearest_1(complementary_odd)

    def process_odds_pair(
        self,
        odds_a: Optional[float],
        odds_b: Optional[float]
    ) -> Tuple[List[float], float, float]:
        """Process a pair of betting odds."""
        utils = DataUtils()

        if pd.notna(odds_a) and pd.notna(odds_b):
            odds_list = [odds_a, odds_b]
            odds_diff = odds_a - odds_b
            odds_ratio = utils.safe_divide(odds_a, odds_b)
        elif pd.notna(odds_a):
            odds_a_rounded = self.round_to_nearest_1(odds_a)
            odds_b_calc = self.calculate_complementary_odd(odds_a_rounded)
            odds_list = [odds_a_rounded, odds_b_calc]
            odds_diff = odds_a_rounded - odds_b_calc
            odds_ratio = utils.safe_divide(odds_a_rounded, odds_b_calc)
        elif pd.notna(odds_b):
            odds_b_rounded = self.round_to_nearest_1(odds_b)
            odds_a_calc = self.calculate_complementary_odd(odds_b_rounded)
            odds_list = [odds_a_calc, odds_b_rounded]
            odds_diff = odds_a_calc - odds_b_rounded
            odds_ratio = utils.safe_divide(odds_a_calc, odds_b_rounded)
        else:
            odds_list = [-111, -111]
            odds_diff = 0
            odds_ratio = 1

        return odds_list, odds_diff, odds_ratio

    def process_odds_data(
        self,
        final_stats: pd.DataFrame,
        odds_filepath: Optional[Union[str, Path]] = None
    ) -> pd.DataFrame:
        """Process and merge betting odds data with fight statistics."""
        final_stats = final_stats.copy().loc[:, ~final_stats.columns.duplicated()]

        odds_path = self._resolve_odds_path(odds_filepath)
        try:
            odds_df = pd.read_csv(odds_path)
        except Exception as exc:
            raise RuntimeError(f"Failed to load odds data from {odds_path}") from exc

        final_stats['fighter'] = final_stats['fighter'].str.lower().str.strip()
        odds_df['Matchup'] = odds_df['Matchup'].str.lower().str.strip()

        odds_df.rename(columns={'Matchup': 'fighter'}, inplace=True)

        final_stats['fight_date'] = pd.to_datetime(final_stats['fight_date'])
        odds_df['Date'] = pd.to_datetime(odds_df['Date'], format='%Y-%m-%d')

        final_stats.sort_values('fight_date', inplace=True)
        odds_df.sort_values('Date', inplace=True)

        merged_df = pd.merge_asof(
            final_stats,
            odds_df,
            left_on='fight_date',
            right_on='Date',
            by='fighter',
            tolerance=pd.Timedelta("1D"),
            direction='nearest'
        )

        merged_df.drop(columns=['Date'], inplace=True)

        merged_df.rename(
            columns={
                'Open': 'open_odds',
                'Closing Range Start': 'closing_range_start',
                'Closing Range End': 'closing_range_end',
                'Movement': 'odds_movement'
            },
            inplace=True
        )

        return merged_df


class FighterUtils:
    """Utilities for processing fighter statistics with advanced features."""

    def __init__(self, enable_verification: bool = True):
        """Initialize with DataUtils instance."""
        self.utils = DataUtils()
        self.enable_verification = enable_verification
        self.verification_results = []

    def aggregate_fighter_stats(self, group: pd.DataFrame, numeric_columns: List[str]) -> pd.DataFrame:
        """
        Calculate cumulative career statistics for a fighter.
        FEATURE 1: Includes opponent quality tracking (will be populated after pairing).
        """
        group = group.sort_values('fight_date')
        cumulative_stats = group[numeric_columns].cumsum(skipna=True)
        fight_count = group.groupby('fighter').cumcount() + 1

        for col in numeric_columns:
            group[f"{col}_career"] = cumulative_stats[col]
            group[f"{col}_career_avg"] = self.utils.safe_divide(cumulative_stats[col], fight_count)

        # Leakage check
        if self.enable_verification and len(group) > 0:
            fighter_name = group['fighter'].iloc[0]
            verification_passed = True

            for i in range(min(3, len(group))):
                fight = group.iloc[i]

                if 'knockdowns' in numeric_columns and 'knockdowns_career' in group.columns:
                    expected_knockdowns = group['knockdowns'].iloc[:i+1].sum()
                    actual_knockdowns = fight.get('knockdowns_career', 0)

                    if abs(expected_knockdowns - actual_knockdowns) > 0.01:
                        verification_passed = False
                        print(f"❌ LEAKAGE CHECK #1: Career stats for {fighter_name}, fight {i+1}")
                        print(f"   Expected knockdowns_career: {expected_knockdowns}, got {actual_knockdowns}")
                        break

            if verification_passed:
                self.verification_results.append(('career_stats', fighter_name, True))

        # Calculate career rate stats
        group['significant_strikes_rate_career'] = self.utils.safe_divide(
            cumulative_stats.get('significant_strikes_landed', 0),
            cumulative_stats.get('significant_strikes_attempted', 1)
        )
        group['takedown_rate_career'] = self.utils.safe_divide(
            cumulative_stats.get('takedown_successful', 0),
            cumulative_stats.get('takedown_attempted', 1)
        )
        group['total_strikes_rate_career'] = self.utils.safe_divide(
            cumulative_stats.get('total_strikes_landed', 0),
            cumulative_stats.get('total_strikes_attempted', 1)
        )
        group["combined_success_rate_career"] = (group["takedown_rate_career"] + group["total_strikes_rate_career"]) / 2

        return group

    def calculate_experience_and_days(self, group: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate fighter experience and days between fights.
        FEATURE 6: Includes layoff effects calculation.
        """
        group = group.sort_values('fight_date')
        group['years_of_experience'] = (group['fight_date'] - group['fight_date'].iloc[0]).dt.days / 365.25
        group['days_since_last_fight'] = (group['fight_date'] - group['fight_date'].shift()).dt.days

        # FEATURE 6: Layoff Effects
        optimal_days = 135
        group['layoff_penalty'] = np.exp(
            -((group['days_since_last_fight'].fillna(0) - optimal_days) ** 2) / (2 * 90 ** 2)
        )

        group['rushed_return'] = (group['days_since_last_fight'].fillna(999) < 60).astype(int)
        group['ring_rust'] = (group['days_since_last_fight'].fillna(0) > 365).astype(int)

        return group

    def update_streaks(self, group: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate win and loss streaks for a fighter.
        FEATURE 2: Includes momentum indicators.
        """
        group = group.sort_values('fight_date')
        group_copy = group.copy()

        group_copy['win_streak'] = 0
        group_copy['loss_streak'] = 0

        for i in range(1, len(group_copy)):
            if group_copy.iloc[i-1]['winner'] == 1:
                group_copy.iloc[i, group_copy.columns.get_loc('win_streak')] = group_copy.iloc[i-1]['win_streak'] + 1
                group_copy.iloc[i, group_copy.columns.get_loc('loss_streak')] = 0
            else:
                group_copy.iloc[i, group_copy.columns.get_loc('win_streak')] = 0
                group_copy.iloc[i, group_copy.columns.get_loc('loss_streak')] = group_copy.iloc[i-1]['loss_streak'] + 1

        # FEATURE 2: Momentum Indicator
        group_copy['momentum'] = group_copy['win_streak'] - group_copy['loss_streak']

        # Leakage check
        if self.enable_verification and len(group_copy) > 0:
            fighter_name = group_copy['fighter'].iloc[0]
            first_fight = group_copy.iloc[0]

            if first_fight['win_streak'] != 0 or first_fight['loss_streak'] != 0:
                print(f"❌ LEAKAGE CHECK #2: Streaks for {fighter_name}")
                self.verification_results.append(('streaks', fighter_name, False))
            else:
                streak_valid = True
                for i in range(1, min(3, len(group_copy))):
                    prev_fight = group_copy.iloc[i-1]
                    curr_fight = group_copy.iloc[i]

                    if prev_fight['winner'] == 1:
                        expected_win = prev_fight['win_streak'] + 1
                        if curr_fight['win_streak'] != expected_win or curr_fight['loss_streak'] != 0:
                            streak_valid = False
                            break
                    else:
                        expected_loss = prev_fight['loss_streak'] + 1
                        if curr_fight['loss_streak'] != expected_loss or curr_fight['win_streak'] != 0:
                            streak_valid = False
                            break

                if streak_valid:
                    self.verification_results.append(('streaks', fighter_name, True))

        return group_copy

    def calculate_time_based_stats(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate time-normalized statistics."""
        df['time_career_minutes'] = df['time_career'] / 60

        df['takedowns_per_15min'] = self.utils.safe_divide(
            df['takedown_successful_career'], df['time_career_minutes']
        ) * 15

        df['knockdowns_per_15min'] = self.utils.safe_divide(
            df['knockdowns_career'], df['time_career_minutes']
        ) * 15

        return df

    def calculate_total_fight_stats(self, group: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate cumulative fight statistics.
        FEATURE 2: Includes recent form (EWMA).
        FEATURE 1: Stores opponent quality data for later use.
        """
        group = group.sort_values('fight_date').reset_index(drop=True)

        group['total_fights'] = range(1, len(group) + 1)
        group['total_wins'] = group['winner'].cumsum()
        group['total_losses'] = group['total_fights'] - group['total_wins']

        # FEATURE 1: Opponent Quality Storage (populated after pairing)
        if 'pre_fight_elo_b' in group.columns:
            group['opponent_elo_at_fight'] = group['pre_fight_elo_b']
            group['quality_adjusted_win'] = group['winner'] * (group['pre_fight_elo_b'] / 1500)
            group['avg_opponent_elo'] = group['pre_fight_elo_b'].expanding().mean()

        # Leakage check
        if self.enable_verification and len(group) > 0:
            fighter_name = group['fighter'].iloc[0]
            verification_passed = True

            first_fight = group.iloc[0]
            if first_fight['total_fights'] != 1:
                print(f"❌ LEAKAGE CHECK #3: Total fights for {fighter_name}")
                verification_passed = False

            for i in range(min(3, len(group))):
                fight = group.iloc[i]
                if fight['total_fights'] != i + 1:
                    print(f"❌ LEAKAGE CHECK #3: Total fights progression for {fighter_name}")
                    verification_passed = False
                    break

            if verification_passed:
                self.verification_results.append(('total_fights', fighter_name, True))

        # Calculate outcome types FIRST (before they're referenced)
        ko_mask = group['result'].isin([0, 3])
        submission_mask = group['result'] == 1
        decision_mask = group['result'].isin([2, 4])

        win_mask = group['winner'] == 1
        loss_mask = ~win_mask

        group['wins_by_ko'] = (ko_mask & win_mask).cumsum()
        group['wins_by_submission'] = (submission_mask & win_mask).cumsum()
        group['wins_by_decision'] = (decision_mask & win_mask).cumsum()

        group['losses_by_ko'] = (ko_mask & loss_mask).cumsum()
        group['losses_by_submission'] = (submission_mask & loss_mask).cumsum()
        group['losses_by_decision'] = (decision_mask & loss_mask).cumsum()

        # NOW calculate rates (after the count columns exist)
        for outcome in ['ko', 'submission', 'decision']:
            group[f'win_rate_by_{outcome}'] = self.utils.safe_divide(
                group[f'wins_by_{outcome}'], group['total_wins']
            )
            group[f'loss_rate_by_{outcome}'] = self.utils.safe_divide(
                group[f'losses_by_{outcome}'], group['total_losses']
            )

        # FEATURE 2: Recent Form (Exponentially Weighted Moving Average)
        # Now we can safely calculate these since the columns exist
        group['ewm_win_rate'] = group['winner'].ewm(span=3, adjust=False, min_periods=1).mean()

        ko_submission_mask = group['result'].isin([0, 1, 3])
        group['ewm_finish_rate'] = ko_submission_mask.astype(float).ewm(span=3, adjust=False, min_periods=1).mean()

        strike_acc = self.utils.safe_divide(
            group['significant_strikes_landed'],
            group['significant_strikes_attempted']
        )
        group['ewm_strike_accuracy'] = pd.Series(strike_acc).ewm(span=3, adjust=False, min_periods=1).mean()

        # Performance Trajectory (now safe to calculate)
        career_win_rate = group['total_wins'] / group['total_fights']
        group['win_rate_trajectory'] = group['ewm_win_rate'] - career_win_rate

        career_finish_rate = (
                (group['wins_by_ko'] + group['wins_by_submission']) /
                group['total_fights']
        )
        group['finish_rate_trajectory'] = group['ewm_finish_rate'] - career_finish_rate

        return group

    def calculate_fighting_style(self, group: pd.DataFrame) -> pd.DataFrame:
        """
        FEATURE 5: Calculate fighting style indicators.
        Classifies fighters as strikers, grapplers, or pressure fighters.
        """
        group = group.sort_values('fight_date')

        # Striker Score
        striker_score = (
            (group['significant_strikes_landed_per_min'] / 5.0).clip(0, 1) +
            (group['distance_landed_career_avg'] / 30.0).clip(0, 1)
        ) / 2
        group['striker_score'] = striker_score.clip(0, 1)

        # Grappler Score
        grappler_score = (
            (group['takedowns_per_15min'] / 3.0).clip(0, 1) +
            (group['submission_attempt_career_avg'] / 2.0).clip(0, 1) +
            (group['ground_landed_career_avg'] / 20.0).clip(0, 1)
        ) / 3
        group['grappler_score'] = grappler_score.clip(0, 1)

        # Pressure Fighter Score
        pressure_score = (
            (group['total_strikes_attempted_per_min'] / 8.0).clip(0, 1) +
            (group['clinch_landed_career_avg'] / 10.0).clip(0, 1)
        ) / 2
        group['pressure_score'] = pressure_score.clip(0, 1)

        # Primary Style Classification
        style_scores = pd.DataFrame({
            'striker': striker_score,
            'grappler': grappler_score,
            'pressure': pressure_score
        })
        group['primary_style'] = style_scores.idxmax(axis=1)
        group['style_confidence'] = style_scores.max(axis=1)

        return group

    def print_verification_summary(self):
        """Print summary of verification results."""
        if self.verification_results:
            print("\n" + "="*60)
            print("LEAKAGE VERIFICATION SUMMARY")
            print("="*60)

            passed = sum(1 for _, _, result in self.verification_results if result)
            total = len(self.verification_results)

            print(f"Checks passed: {passed}/{total}")

            if passed == total:
                print("✅ All verification checks passed - No leakage detected")
            else:
                print("❌ Some verification checks failed - Review the output above")

            print("="*60)


class DateUtils:
    """Utilities for date processing."""

    @staticmethod
    def parse_date(date_str: Any) -> pd.Timestamp:
        """Parse date string in various formats."""
        if pd.isna(date_str):
            return pd.NaT
        try:
            return pd.to_datetime(date_str, format='%d-%b-%y')
        except ValueError:
            try:
                return pd.to_datetime(date_str, format='%b %d, %Y')
            except ValueError:
                return pd.NaT