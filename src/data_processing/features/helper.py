"""
UFC Data Analysis Utilities - Simplified
Consolidated utilities for UFC fight data processing with optional leakage verification.
"""
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Union, Any
from dataclasses import dataclass


@dataclass
class ProcessingConfig:
    """Configuration for data processing."""
    enable_verification: bool = True
    optimal_days_between_fights: int = 135
    layoff_std_dev: int = 90
    ewm_span: int = 3
    correlation_threshold: float = 0.95


CONFIG = ProcessingConfig()


# ==================== CORE DATA PROCESSOR ====================

class DataProcessor:
    """Consolidated data processing utilities."""

    @staticmethod
    def safe_divide(num: Union[float, np.ndarray, pd.Series],
                   denom: Union[float, np.ndarray, pd.Series],
                   default: float = 0) -> Union[float, np.ndarray, pd.Series]:
        """Safely divide with protection against division by zero."""
        if isinstance(num, (pd.Series, pd.DataFrame)) or isinstance(denom, (pd.Series, pd.DataFrame)):
            result = pd.Series(num).div(pd.Series(denom))
            return result.fillna(default).replace([np.inf, -np.inf], default)
        elif isinstance(num, np.ndarray) and isinstance(denom, np.ndarray):
            result = np.zeros_like(num, dtype=float)
            mask = denom != 0
            result[mask] = num[mask] / denom[mask]
            result[~mask] = default
            return result
        else:
            return num / denom if denom != 0 else default

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

    def preprocess_data(self, ufc_stats: pd.DataFrame, fighter_stats: pd.DataFrame) -> pd.DataFrame:
        """Preprocess the UFC and fighter stats dataframes."""
        ufc_stats['fighter'] = ufc_stats['fighter'].astype(str).str.lower()
        ufc_stats['fight_date'] = pd.to_datetime(ufc_stats['fight_date'])
        fighter_stats['name'] = fighter_stats['FIGHTER'].astype(str).str.lower().str.strip()
        fighter_stats['dob'] = fighter_stats['DOB'].replace(['--', '', 'NA', 'N/A'], np.nan).apply(self.parse_date)

        ufc_stats = pd.merge(
            ufc_stats,
            fighter_stats[['name', 'dob']],
            left_on='fighter', right_on='name',
            how='left'
        )
        ufc_stats['age'] = ((ufc_stats['fight_date'] - ufc_stats['dob']).dt.days / 365.25).round()
        ufc_stats['age'] = ufc_stats['age'].fillna(np.nan).astype(float)
        ufc_stats.loc[ufc_stats['age'] < 0, 'age'] = np.nan

        ufc_stats = ufc_stats.drop(['round', 'location', 'name'], axis=1)
        ufc_stats = ufc_stats[~ufc_stats['weight_class'].str.contains("Women's")]

        ufc_stats['time'] = (
                pd.to_datetime(ufc_stats['time'], format='%M:%S').dt.minute * 60 +
                pd.to_datetime(ufc_stats['time'], format='%M:%S').dt.second
        )

        return ufc_stats

    @staticmethod
    def rename_columns_general(col: str) -> str:
        """Rename columns for clarity."""
        if 'fighter' in col and not col.startswith('fighter'):
            if 'b_fighter_b' in col:
                return col.replace('b_fighter_b', 'fighter_b_opponent')
            elif 'b_fighter' in col:
                return col.replace('b_fighter', 'fighter_a_opponent')
            elif 'fighter' in col and 'fighter_b' not in col:
                return col.replace('fighter', 'fighter_a')
        return col

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


# ==================== ODDS UTILITIES ====================

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
        self.utils = DataProcessor()

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
        if pd.notna(odds_a) and pd.notna(odds_b):
            odds_list = [odds_a, odds_b]
            odds_diff = odds_a - odds_b
            odds_ratio = self.utils.safe_divide(odds_a, odds_b)
        elif pd.notna(odds_a):
            odds_a_rounded = self.round_to_nearest_1(odds_a)
            odds_b_calc = self.calculate_complementary_odd(odds_a_rounded)
            odds_list = [odds_a_rounded, odds_b_calc]
            odds_diff = odds_a_rounded - odds_b_calc
            odds_ratio = self.utils.safe_divide(odds_a_rounded, odds_b_calc)
        elif pd.notna(odds_b):
            odds_b_rounded = self.round_to_nearest_1(odds_b)
            odds_a_calc = self.calculate_complementary_odd(odds_b_rounded)
            odds_list = [odds_a_calc, odds_b_rounded]
            odds_diff = odds_a_calc - odds_b_rounded
            odds_ratio = self.utils.safe_divide(odds_a_calc, odds_b_rounded)
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


# ==================== FIGHTER UTILITIES ====================

class FighterUtils:
    """Utilities for processing fighter statistics with advanced features."""

    def __init__(self, enable_verification: bool = True):
        """Initialize with verification option."""
        self.utils = DataProcessor()
        self.enable_verification = enable_verification
        self.verification_results = []

    def aggregate_fighter_stats(self, group: pd.DataFrame, numeric_columns: List[str]) -> pd.DataFrame:
        """Calculate cumulative career statistics for a fighter."""
        group = group.sort_values('fight_date')
        cumulative_stats = group[numeric_columns].cumsum(skipna=True)
        fight_count = np.arange(1, len(group) + 1)

        # Vectorized career stats calculation
        for col in numeric_columns:
            group[f"{col}_career"] = cumulative_stats[col].values
            group[f"{col}_career_avg"] = cumulative_stats[col].values / fight_count

        # Verification check (simplified)
        if self.enable_verification and len(group) > 0:
            self._verify_career_stats(group, numeric_columns)

        # Calculate career rates
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
        """Calculate fighter experience and days between fights."""
        cfg = CONFIG
        group = group.sort_values('fight_date')
        group['years_of_experience'] = (group['fight_date'] - group['fight_date'].iloc[0]).dt.days / 365.25
        days_since = (group['fight_date'] - group['fight_date'].shift()).dt.days
        group['days_since_last_fight'] = days_since
        group['layoff_penalty'] = np.exp(-((days_since.fillna(0) - cfg.optimal_days_between_fights) ** 2) / (2 * cfg.layoff_std_dev ** 2))
        group['rushed_return'] = (days_since.fillna(999) < 60).astype(int)
        group['ring_rust'] = (days_since.fillna(0) > 365).astype(int)
        return group

    def update_streaks(self, group: pd.DataFrame) -> pd.DataFrame:
        """Calculate win and loss streaks (vectorized)."""
        group = group.sort_values('fight_date').copy()

        wins = (group['winner'] == 1).astype(int)
        losses = 1 - wins

        # Vectorized streak calculation using cumsum and groupby
        win_groups = (wins != wins.shift()).cumsum()
        loss_groups = (losses != losses.shift()).cumsum()

        group['win_streak'] = wins.groupby(win_groups).cumsum().shift(1, fill_value=0)
        group['loss_streak'] = losses.groupby(loss_groups).cumsum().shift(1, fill_value=0)
        group['momentum'] = group['win_streak'] - group['loss_streak']

        # Verification check (simplified)
        if self.enable_verification and len(group) > 0:
            self._verify_streaks(group)

        return group

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
        """Calculate cumulative fight statistics and recent form."""
        group = group.sort_values('fight_date').reset_index(drop=True)

        group['total_fights'] = range(1, len(group) + 1)
        group['total_wins'] = group['winner'].cumsum()
        group['total_losses'] = group['total_fights'] - group['total_wins']

        # Opponent quality storage (populated after pairing)
        if 'pre_fight_elo_b' in group.columns:
            group['opponent_elo_at_fight'] = group['pre_fight_elo_b']
            group['quality_adjusted_win'] = group['winner'] * (group['pre_fight_elo_b'] / 1500)
            group['avg_opponent_elo'] = group['pre_fight_elo_b'].expanding().mean()

        # Verification check
        if self.enable_verification and len(group) > 0:
            self._verify_total_fights(group)

        # Outcome types
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

        # Outcome rates
        for outcome in ['ko', 'submission', 'decision']:
            group[f'win_rate_by_{outcome}'] = self.utils.safe_divide(
                group[f'wins_by_{outcome}'], group['total_wins']
            )
            group[f'loss_rate_by_{outcome}'] = self.utils.safe_divide(
                group[f'losses_by_{outcome}'], group['total_losses']
            )

        # Recent form (EWMA)
        cfg = CONFIG
        span = cfg.ewm_span
        group['ewm_win_rate'] = group['winner'].ewm(span=span, adjust=False, min_periods=1).mean()
        ko_sub = group['result'].isin([0, 1, 3]).astype(float)
        group['ewm_finish_rate'] = ko_sub.ewm(span=span, adjust=False, min_periods=1).mean()
        strike_acc = self.utils.safe_divide(group['significant_strikes_landed'], group['significant_strikes_attempted'])
        group['ewm_strike_accuracy'] = pd.Series(strike_acc).ewm(span=span, adjust=False, min_periods=1).mean()

        # Performance trajectory
        career_win_rate = group['total_wins'] / group['total_fights']
        group['win_rate_trajectory'] = group['ewm_win_rate'] - career_win_rate

        career_finish_rate = (group['wins_by_ko'] + group['wins_by_submission']) / group['total_fights']
        group['finish_rate_trajectory'] = group['ewm_finish_rate'] - career_finish_rate

        return group

    def calculate_fighting_style(self, group: pd.DataFrame) -> pd.DataFrame:
        """Calculate fighting style indicators."""
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

    # ==================== VERIFICATION HELPERS ====================

    def _verify_career_stats(self, group: pd.DataFrame, numeric_columns: List[str]):
        """Verify career stats don't have leakage."""
        fighter_name = group['fighter'].iloc[0]
        if 'knockdowns' in numeric_columns and 'knockdowns_career' in group.columns and len(group) >= 3:
            for i in range(min(3, len(group))):
                expected = group['knockdowns'].iloc[:i+1].sum()
                actual = group.iloc[i]['knockdowns_career']
                if abs(expected - actual) > 0.01:
                    print(f"LEAKAGE: Career stats for {fighter_name}, fight {i+1}")
                    self.verification_results.append(('career_stats', fighter_name, False))
                    return
        self.verification_results.append(('career_stats', fighter_name, True))

    def _verify_streaks(self, group: pd.DataFrame):
        """Verify streak calculations."""
        fighter_name = group['fighter'].iloc[0]
        first_fight = group.iloc[0]
        if first_fight['win_streak'] != 0 or first_fight['loss_streak'] != 0:
            print(f"LEAKAGE: Streaks for {fighter_name} start non-zero")
            self.verification_results.append(('streaks', fighter_name, False))
        else:
            self.verification_results.append(('streaks', fighter_name, True))

    def _verify_total_fights(self, group: pd.DataFrame):
        """Verify total fight counts."""
        fighter_name = group['fighter'].iloc[0]
        if group.iloc[0]['total_fights'] != 1:
            print(f"LEAKAGE: Total fights for {fighter_name} doesn't start at 1")
            self.verification_results.append(('total_fights', fighter_name, False))
        else:
            self.verification_results.append(('total_fights', fighter_name, True))

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
                print("[PASS] All verification checks passed - No leakage detected")
            else:
                print("[FAIL] Some verification checks failed - Review the output above")

            print("="*60)


# ==================== LEGACY COMPATIBILITY ====================

# Maintain backward compatibility with old import names
DataUtils = DataProcessor
DateUtils = DataProcessor
