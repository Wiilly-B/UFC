"""UFC Fighter Matchup Prediction System - Compact Version"""
import os, sys, argparse, re, pickle, json
import numpy as np
import pandas as pd
from datetime import datetime
from typing import List, Dict, Tuple, Optional, Union
import xgboost as xgb
from sklearn.calibration import CalibratedClassifierCV
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

# Try importing original utilities, fallback if unavailable
try:
    from src.data_processing.cleaning.data_cleaner import DataUtils, OddsUtils
    _UTILS = (DataUtils(), OddsUtils())
except ImportError:
    class _Utils:
        def safe_divide(self, a, b): return 1.0 if pd.isna(a) or pd.isna(b) or b == 0 else a / b
        def process_odds_pair(self, o_a, o_b):
            d_a = (o_a / 100) + 1 if o_a > 0 else (100 / abs(o_a)) + 1
            d_b = (o_b / 100) + 1 if o_b > 0 else (100 / abs(o_b)) + 1
            return [d_a, d_b], d_a - d_b, d_a / d_b if d_b else 1.0
    _UTILS = (_Utils(), _Utils())

safe_divide = _UTILS[0].safe_divide

def safe_process_odds(o_a, o_b):
    """Process odds pair, handling None values."""
    if o_a is None or o_b is None:
        return [0, 0], 0, 0
    return _UTILS[1].process_odds_pair(o_a, o_b)

process_odds_pair = safe_process_odds


def resolve_data_dir():
    """Find data directory from standard locations."""
    for path in [os.path.join(d, "data") for d in [
        os.path.dirname(os.path.abspath(__file__)),
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "."
    ]] + ([os.path.join(os.environ.get('USERPROFILE', ''), "PycharmProjects/UFC/data")]):
        if os.path.exists(path): return os.path.abspath(path)
    return os.path.abspath("data")


def ensure_directory_exists(directory):
    """Ensure directory exists."""
    os.makedirs(os.path.abspath(directory), exist_ok=True)
    return os.path.abspath(directory)


class FightDataProcessor:
    """Loads and processes fight CSV files."""
    def __init__(self, data_dir):
        self.data_dir = data_dir
    def _load_csv(self, filepath):
        return pd.read_csv(filepath)
    def _save_csv(self, df, filepath):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        df.to_csv(filepath, index=False)


class FighterMatchupPredictor:
    """Creates matchup data for fighter pairs."""
    def __init__(self, data_dir: str = "data"):
        self.data_dir = os.path.abspath(data_dir)
        self.fight_processor = FightDataProcessor(self.data_dir)

    def create_fighter_matchup(self, fighter_a: str, fighter_b: str,
                              open_odds_a: Optional[float] = None, open_odds_b: Optional[float] = None,
                              closing_odds_a: Optional[float] = None, closing_odds_b: Optional[float] = None,
                              fight_date: Optional[Union[str, datetime]] = None,
                              n_past_fights: int = 3, save_individual_file: bool = True) -> pd.DataFrame:
        """Create matchup between two fighters. Odds are optional (can be None)."""
        current_date = datetime.strptime(fight_date, '%Y-%m-%d') if isinstance(fight_date, str) else (fight_date or datetime.now())
        if DEBUG: print(f"Creating matchup: {fighter_a} vs {fighter_b} on {current_date.strftime('%Y-%m-%d')}")

        fighter_stats_file = os.path.join(self.data_dir, PATHS['processed_fighter_stats'])
        if not os.path.exists(fighter_stats_file): raise FileNotFoundError(f"Fighter stats file not found: {fighter_stats_file}")

        df = self.fight_processor._load_csv(fighter_stats_file)
        df['fight_date'] = pd.to_datetime(df['fight_date'])
        df['fighter_lower'] = df['fighter'].str.lower()
        fighter_a_lower, fighter_b_lower = fighter_a.lower(), fighter_b.lower()

        if fighter_a_lower not in df['fighter_lower'].values: raise ValueError(f"Fighter '{fighter_a}' not found")
        if fighter_b_lower not in df['fighter_lower'].values: raise ValueError(f"Fighter '{fighter_b}' not found")

        # Get features
        features_to_include = [col for col in df.columns if col not in EXCLUDED_COLUMNS and col != 'age' and not col.endswith('_age')]
        numeric_features = [col for col in features_to_include if df[col].dtype in ['float64', 'float32', 'int64', 'int32']]

        # Get recent fights
        fighter_a_df = df[(df['fighter_lower'] == fighter_a_lower) & (df['fight_date'] < current_date)].sort_values('fight_date', ascending=False).head(n_past_fights)
        fighter_b_df = df[(df['fighter_lower'] == fighter_b_lower) & (df['fight_date'] < current_date)].sort_values('fight_date', ascending=False).head(n_past_fights)

        if len(fighter_a_df) == 0 or len(fighter_b_df) == 0: raise ValueError("Not enough fight data available")

        # Extract features
        fighter_a_features = fighter_a_df[numeric_features].mean().values
        fighter_b_features = fighter_b_df[numeric_features].mean().values

        # Get fighter stats
        tester = 6 - n_past_fights
        stats_a = self._calculate_fighter_stats(fighter_a_lower, df, current_date, n_past_fights)
        stats_b = self._calculate_fighter_stats(fighter_b_lower, df, current_date, n_past_fights)
        age_a, exp_a, days_a, win_streak_a, loss_streak_a = stats_a
        age_b, exp_b, days_b, win_streak_b, loss_streak_b = stats_b

        # Get ELO
        elo_a = fighter_a_df['fight_outcome_elo'].iloc[0] if 'fight_outcome_elo' in fighter_a_df.columns else (fighter_a_df['pre_fight_elo'].iloc[0] if 'pre_fight_elo' in fighter_a_df.columns else 1500)
        elo_b = fighter_b_df['fight_outcome_elo'].iloc[0] if 'fight_outcome_elo' in fighter_b_df.columns else (fighter_b_df['pre_fight_elo_b'].iloc[0] if 'pre_fight_elo_b' in fighter_b_df.columns else 1500)

        # Build matchup
        elo_diff = elo_a - elo_b
        elo_a_win_prob = 1 / (1 + 10 ** ((elo_b - elo_a) / 400))
        elo_b_win_prob = 1 / (1 + 10 ** ((elo_a - elo_b) / 400))

        # Results padding
        num_a_results = min(len(fighter_a_df), tester)
        num_b_results = min(len(fighter_b_df), tester)
        results_a = fighter_a_df[['result', 'winner', 'weight_class', 'scheduled_rounds']].head(num_a_results).values.flatten() if num_a_results > 0 else np.array([])
        results_b = fighter_b_df[['result_b', 'winner_b', 'weight_class_b', 'scheduled_rounds_b']].head(num_b_results).values.flatten() if num_b_results > 0 else np.array([])
        results_a = np.pad(results_a, (0, tester * 4 - len(results_a)), 'constant', constant_values=np.nan)
        results_b = np.pad(results_b, (0, tester * 4 - len(results_b)), 'constant', constant_values=np.nan)

        # Combine features
        odds_o, odds_o_diff, odds_o_ratio = process_odds_pair(open_odds_a, open_odds_b)
        odds_c, odds_c_diff, odds_c_ratio = process_odds_pair(closing_odds_a, closing_odds_b)

        closing_open_diff_a = (closing_odds_a - open_odds_a) if (closing_odds_a is not None and open_odds_a is not None) else np.nan
        closing_open_diff_b = (closing_odds_b - open_odds_b) if (closing_odds_b is not None and open_odds_b is not None) else np.nan
        combined = np.concatenate([
            fighter_a_features, fighter_b_features, results_a, results_b,
            odds_o, [odds_o_diff, odds_o_ratio],
            odds_c, [odds_c_diff, odds_c_ratio, closing_open_diff_a, closing_open_diff_b],
            [age_a, age_b, age_a - age_b, safe_divide(age_a, age_b)],
            [elo_a, elo_b, elo_diff, elo_a_win_prob, elo_b_win_prob, safe_divide(elo_a, elo_b)],
            [win_streak_a, win_streak_b, win_streak_a - win_streak_b, safe_divide(win_streak_a, win_streak_b),
             loss_streak_a, loss_streak_b, loss_streak_a - loss_streak_b, safe_divide(loss_streak_a, loss_streak_b),
             exp_a, exp_b, exp_a - exp_b, safe_divide(exp_a, exp_b),
             days_a, days_b, days_a - days_b, safe_divide(days_a, days_b)]
        ])

        # Column names
        col_names = ['fighter_a', 'fighter_b', 'fight_date_recent']
        col_names += [f"{f}_fighter_avg_last_{n_past_fights}" for f in numeric_features]
        col_names += [f"{f}_fighter_b_avg_last_{n_past_fights}" for f in numeric_features]
        for i in range(1, tester + 1):
            col_names += [f"result_fight_{i}", f"winner_fight_{i}", f"weight_class_fight_{i}", f"scheduled_rounds_fight_{i}",
                         f"result_b_fight_{i}", f"winner_b_fight_{i}", f"weight_class_b_fight_{i}", f"scheduled_rounds_b_fight_{i}"]
        col_names += ['current_fight_open_odds', 'current_fight_open_odds_b', 'current_fight_open_odds_diff', 'current_fight_open_odds_ratio',
                     'current_fight_closing_odds', 'current_fight_closing_odds_b', 'current_fight_closing_odds_diff', 'current_fight_closing_odds_ratio',
                     'current_fight_closing_open_diff_a', 'current_fight_closing_open_diff_b',
                     'current_fight_age', 'current_fight_age_b', 'current_fight_age_diff', 'current_fight_age_ratio',
                     'current_fight_pre_fight_elo_a', 'current_fight_pre_fight_elo_b', 'current_fight_pre_fight_elo_diff',
                     'current_fight_pre_fight_elo_a_win_chance', 'current_fight_pre_fight_elo_b_win_chance', 'current_fight_pre_fight_elo_ratio',
                     'current_fight_win_streak_a', 'current_fight_win_streak_b', 'current_fight_win_streak_diff', 'current_fight_win_streak_ratio',
                     'current_fight_loss_streak_a', 'current_fight_loss_streak_b', 'current_fight_loss_streak_diff', 'current_fight_loss_streak_ratio',
                     'current_fight_years_experience_a', 'current_fight_years_experience_b', 'current_fight_years_experience_diff', 'current_fight_years_experience_ratio',
                     'current_fight_days_since_last_a', 'current_fight_days_since_last_b', 'current_fight_days_since_last_diff', 'current_fight_days_since_last_ratio',
                     'winner', 'current_fight_date']

        matchup_df = pd.DataFrame([[fighter_a, fighter_b, fighter_a_df['fight_date'].max()] + combined.tolist() + [0, current_date]], columns=col_names)

        # Align with test data
        test_file = os.path.join(self.data_dir, PATHS['test_data'])
        if os.path.exists(test_file):
            test_df = self.fight_processor._load_csv(test_file)
            test_cols = test_df.columns.tolist()
            for col in IMPORTANT_COLUMNS:
                if col not in test_cols and col in matchup_df.columns:
                    test_cols.append(col)
            matchup_df = matchup_df[[c for c in test_cols if c in matchup_df.columns]]

        # Remove correlated features
        removed_file = os.path.join(self.data_dir, PATHS['removed_features'])
        if os.path.exists(removed_file):
            with open(removed_file, 'r') as f:
                removed = [line.strip() for line in f if line.strip()]
            matchup_df = matchup_df.drop(columns=[c for c in removed if c in matchup_df.columns])

        if 'fight_date' in matchup_df.columns: matchup_df = matchup_df.drop(columns=['fight_date'])

        # Save
        if save_individual_file:
            matchup_dir = ensure_directory_exists(os.path.join(self.data_dir, PATHS['live_data']))
            output_file = os.path.join(matchup_dir, f"{fighter_a.replace(' ', '_')}_vs_{fighter_b.replace(' ', '_')}_matchup.csv")
            self.fight_processor._save_csv(matchup_df, output_file)
            if DEBUG: print(f"Saved matchup to: {output_file}")

        return matchup_df

    def _calculate_fighter_stats(self, fighter_lower: str, df: pd.DataFrame, current_date: datetime, n_past_fights: int) -> tuple:
        """Calculate fighter age, experience, streaks."""
        all_fights = df[(df['fighter_lower'] == fighter_lower) & (df['fight_date'] < current_date)].sort_values('fight_date', ascending=False)
        recent = all_fights.head(n_past_fights)

        if recent.empty: return np.nan, np.nan, np.nan, np.nan, np.nan

        last_fight = recent['fight_date'].iloc[0]
        first_fight = all_fights['fight_date'].iloc[-1]
        days_since = (current_date - last_fight).days
        experience = (current_date - first_fight).days / 365.25

        age = np.ceil((recent['age'].iloc[0] if 'age' in recent.columns else 30) + days_since / 365.25)
        win_streak = recent['win_streak'].iloc[0] if 'win_streak' in recent.columns else 0
        loss_streak = recent['loss_streak'].iloc[0] if 'loss_streak' in recent.columns else 0

        if 'winner' in recent.columns:
            result = recent['winner'].iloc[0]
            if result == 1: win_streak += 1; loss_streak = 0
            elif result == 0: loss_streak += 1; win_streak = 0

        return age, experience, days_since, win_streak, loss_streak


class UFCMatchupCreator:
    """Creates multiple matchups."""
    def __init__(self, data_dir: str = "data"):
        self.data_dir = os.path.abspath(data_dir)
        self.matchup_predictor = FighterMatchupPredictor(self.data_dir)

    def create_matchup(self, fighter_a: str, fighter_b: str, open_odds_a: Optional[float] = None, open_odds_b: Optional[float] = None,
                      closing_odds_a: Optional[float] = None, closing_odds_b: Optional[float] = None, fight_date: str = None) -> pd.DataFrame:
        return self.matchup_predictor.create_fighter_matchup(fighter_a, fighter_b, open_odds_a, open_odds_b, closing_odds_a, closing_odds_b, fight_date)


class PredictionEvaluator:
    """Loads models and evaluates predictions with Kelly criterion."""
    def __init__(self, data_dir: str = "data"):
        self.data_dir = os.path.abspath(data_dir)
        self.models: List[xgb.XGBClassifier] = []
        self.use_ensemble = True
        self.X_val = None
        self.y_val = None
        self.calibrated_models = None

    def load_models(self, model_dir: Optional[str] = None) -> bool:
        """Load XGBoost models."""
        model_dir = os.path.abspath(model_dir or os.path.join(self.data_dir, "../saved_models/xgboost/single_split/"))

        if not os.path.isdir(model_dir):
            print(f"Model directory not found: {model_dir}")
            return False

        try:
            pattern = re.compile(r'ufc_xgb_single_(?:TRIAL\d{3}|FINAL).*\.json$')
            model_files = sorted([os.path.join(model_dir, f) for f in os.listdir(model_dir) if pattern.match(f)])

            if not model_files:
                print(f"No model files found in {model_dir}")
                return False

            self.models = []
            for path in model_files:
                try:
                    model = xgb.XGBClassifier(enable_categorical=True)
                    model.load_model(path)
                    self.models.append(model)
                except Exception as e:
                    print(f"Error loading {path}: {e}")

            if not self.models:
                print("No models loaded successfully")
                return False

            print(f"Successfully loaded {len(self.models)} models")
            return True
        except Exception as e:
            print(f"Error loading models: {e}")
            return False

    def load_calibration_data(self) -> bool:
        """Load validation data for calibration."""
        try:
            val_file = os.path.join(self.data_dir, PATHS['val_data'])
            if not os.path.exists(val_file):
                print(f"Validation data not found: {val_file}")
                return False

            val_df = pd.read_csv(val_file)
            self.y_val = val_df['winner']

            test_file = os.path.join(self.data_dir, PATHS['test_data'])
            test_df = pd.read_csv(test_file)
            feature_cols = [col for col in test_df.columns if col not in ['current_fight_date', 'fighter_a', 'fighter_b', 'winner']]

            self.X_val = val_df[feature_cols]
            return True
        except Exception as e:
            print(f"Error loading calibration data: {e}")
            return False

    def fit_calibration(self) -> bool:
        """Fit calibration on validation data."""
        if self.X_val is None or self.y_val is None:
            if not self.load_calibration_data():
                return False

        try:
            self.calibrated_models = []
            typ = CALIBRATION_TYPE.lower()

            for model in self.models:
                try:
                    # Align features to model
                    model_features = model.get_booster().feature_names or list(self.X_val.columns)
                    X_val_aligned = self.X_val.reindex(columns=model_features).fillna(0)

                    if CALIBRATION_BACKEND == 'cv':
                        cal = CalibratedClassifierCV(model, cv='prefit', method=typ)
                        cal.fit(X_val_aligned, self.y_val)
                        self.calibrated_models.append(cal)
                    elif CALIBRATION_BACKEND == 'simple':
                        if typ == 'isotonic':
                            Xv_proba = model.predict_proba(X_val_aligned)[:, 1]
                            iso = IsotonicRegression(out_of_bounds='clip')
                            iso.fit(Xv_proba, self.y_val)
                            self.calibrated_models.append(('isotonic', iso, model))
                        else:  # sigmoid
                            Xv_proba = model.predict_proba(X_val_aligned)[:, 1].reshape(-1, 1)
                            lr = LogisticRegression(solver='lbfgs', max_iter=1000)
                            lr.fit(Xv_proba, self.y_val)
                            self.calibrated_models.append(('sigmoid', lr, model))
                except Exception as e:
                    if DEBUG: print(f"Warning: Could not calibrate model: {e}")
                    continue

            if self.calibrated_models:
                if DEBUG: print(f"Calibration fitted {len(self.calibrated_models)}/{len(self.models)} models with {CALIBRATION_TYPE}/{CALIBRATION_BACKEND}")
                return True
            else:
                print("No models could be calibrated")
                return False
        except Exception as e:
            print(f"Error fitting calibration: {e}")
            return False

    def predict_matchup(self, matchup_df: pd.DataFrame) -> Optional[Tuple[float, float]]:
        """Get probability prediction for matchup."""
        if not self.models:
            print("No models loaded. Call load_models() first.")
            return None

        try:
            # Load test data for feature alignment
            test_file = os.path.join(self.data_dir, PATHS['test_data'])
            if not os.path.exists(test_file):
                print(f"Test data not found: {test_file}")
                return None

            test_df = pd.read_csv(test_file)
            feature_cols = [col for col in test_df.columns if col not in ['current_fight_date', 'fighter_a', 'fighter_b', 'winner']]

            # Ensure matchup has all features
            for col in feature_cols:
                if col not in matchup_df.columns: matchup_df[col] = 0.0

            # Get predictions from all models
            predictions = []

            if USE_CALIBRATION and self.calibrated_models:
                for cal_model in self.calibrated_models:
                    model_features = self.models[0].get_booster().feature_names or list(matchup_df.columns)
                    X_aligned = matchup_df[feature_cols].reindex(columns=model_features).fillna(0)

                    if CALIBRATION_BACKEND == 'cv':
                        proba = cal_model.predict_proba(X_aligned)
                    else:  # simple backend
                        method, calibrator, model = cal_model
                        raw_proba = model.predict_proba(X_aligned)[0, 1]
                        if method == 'isotonic':
                            cal_proba = calibrator.transform([raw_proba])[0]
                        else:  # sigmoid
                            cal_proba = calibrator.predict_proba([[raw_proba]])[0, 1]
                        proba = np.array([[1 - cal_proba, cal_proba]])

                    predictions.append(proba[0])
            else:
                for model in self.models:
                    model_features = model.get_booster().feature_names or list(matchup_df.columns)
                    X_aligned = matchup_df[feature_cols].reindex(columns=model_features).fillna(0)
                    proba = model.predict_proba(X_aligned)
                    predictions.append(proba[0])

            # Average across ensemble
            avg_proba = np.mean(predictions, axis=0) if self.use_ensemble and len(predictions) > 1 else predictions[0]
            return float(avg_proba[1]), float(avg_proba[0])

        except Exception as e:
            print(f"Error making predictions: {e}")
            if DEBUG: import traceback; traceback.print_exc()
            return None

    def print_results(self, fighter_a: str, fighter_b: str, prob_a: float, prob_b: float,
                     odds_a: Optional[float], odds_b: Optional[float]):
        """Print prediction results with Kelly analysis. Odds are optional."""
        pred_winner = fighter_a if prob_a > prob_b else fighter_b
        pred_prob = max(prob_a, prob_b)
        odds = odds_a if prob_a > prob_b else odds_b
        has_odds = odds is not None

        # Try rich output, fallback to plain text
        try:
            from rich.console import Console
            from rich.table import Table
            console = Console()
            table = Table(title="UFC Matchup Prediction")
            table.add_column("Metric", style="cyan")
            table.add_column("Value", justify="right", style="magenta")
            table.add_row("Fighter A", fighter_a)
            table.add_row("Fighter B", fighter_b)
            table.add_row("", "")
            table.add_row("Fighter A Win %", f"{prob_a:.2%}")
            table.add_row("Fighter B Win %", f"{prob_b:.2%}")
            table.add_row("", "")
            table.add_row("Predicted Winner", f"{pred_winner} ({pred_prob:.2%})")
            if has_odds:
                table.add_row(f"{pred_winner} Closing Odds", f"{odds:+.0f}")
                b = odds / 100 if odds > 0 else 100 / abs(odds)
                kelly_full = max(0, pred_prob - (1 - pred_prob) / b)
                kelly_half = kelly_full * 0.5
                kelly_quarter = kelly_full * 0.25
                table.add_row("", "")
                table.add_row("Kelly Full", f"{kelly_full:.2%}")
                table.add_row("Kelly Half", f"{kelly_half:.2%}")
                table.add_row("Kelly Quarter", f"{kelly_quarter:.2%}")
            console.print(table)
        except ImportError:
            print(f"\n{'='*50}\n{fighter_a} vs {fighter_b}\nA: {prob_a:.2%} | B: {prob_b:.2%}")
            if has_odds:
                print(f"Winner: {pred_winner} ({pred_prob:.2%}) @ {odds:+.0f}")
                b = odds / 100 if odds > 0 else 100 / abs(odds)
                kelly_full = max(0, pred_prob - (1 - pred_prob) / b)
                kelly_half = kelly_full * 0.5
                kelly_quarter = kelly_full * 0.25
                print(f"Kelly: Full {kelly_full:.2%} | Half {kelly_half:.2%} | Quarter {kelly_quarter:.2%}")
            else:
                print(f"Winner: {pred_winner} ({pred_prob:.2%})")

        # EV calculation (only if odds available)
        if has_odds:
            b = odds / 100 if odds > 0 else 100 / abs(odds)
            ev = (pred_prob * (b * 100)) - ((1 - pred_prob) * 100)
            print(f"\nExpected Value per $100: ${ev:.2f}")
            print(f"Status: {'POSITIVE [YES]' if ev > 0 else 'NEGATIVE [NO]'}")
        else:
            print(f"\n[NOTE: No odds provided - Kelly and EV calculations skipped]")
        print("="*50)


def main():
    """Process all matchups from EXAMPLE_MATCHUPS configuration."""
    data_dir = resolve_data_dir()
    if DEBUG: print(f"Using data directory: {data_dir}\n")

    creator = UFCMatchupCreator(data_dir)
    evaluator = PredictionEvaluator(data_dir)

    # Load models once
    if DEBUG: print("Loading models...")
    if not evaluator.load_models():
        print("Failed to load models")
        return

    # Fit calibration if enabled
    if USE_CALIBRATION:
        if DEBUG: print("Fitting calibration...")
        if not evaluator.fit_calibration():
            if DEBUG: print("Warning: Calibration failed, continuing without calibration")

    if DEBUG: print(f"\nProcessing {len(EXAMPLE_MATCHUPS)} matchups...\n")

    # Process each matchup
    for idx, matchup in enumerate(EXAMPLE_MATCHUPS, 1):
        try:
            fighter_a = matchup['fighter_a']
            fighter_b = matchup['fighter_b']
            open_odds_a = matchup['open_odds_a']
            open_odds_b = matchup['open_odds_b']
            closing_odds_a = matchup['closing_odds_a']
            closing_odds_b = matchup['closing_odds_b']
            fight_date = matchup.get('fight_date', datetime.now().strftime('%Y-%m-%d'))

            print(f"\n{'='*70}")
            print(f"MATCHUP {idx}/{len(EXAMPLE_MATCHUPS)}: {fighter_a} vs {fighter_b}")
            print(f"{'='*70}")

            # Create matchup
            if DEBUG: print(f"Creating matchup data...")
            matchup_data = creator.create_matchup(fighter_a, fighter_b, open_odds_a, open_odds_b, closing_odds_a, closing_odds_b, fight_date)

            if matchup_data is None or len(matchup_data) == 0:
                print("Failed to create matchup data\n")
                continue

            # Generate predictions
            if DEBUG: print(f"Generating predictions...")
            predictions = evaluator.predict_matchup(matchup_data)

            if predictions is None:
                print("Failed to generate predictions\n")
                continue

            # Print results
            evaluator.print_results(fighter_a, fighter_b, predictions[0], predictions[1], closing_odds_a, closing_odds_b)

        except Exception as e:
            print(f"Error processing {fighter_a} vs {fighter_b}: {e}")
            if DEBUG: import traceback; traceback.print_exc()
            continue

    print(f"\n{'='*70}")
    print(f"All {len(EXAMPLE_MATCHUPS)} matchups processed!")
    print(f"{'='*70}")
# ============================================================================
# CONFIGURATION (Bottom of file)
# ============================================================================

DEBUG = True

PATHS = {
    'processed_fighter_stats': "processed/combined_sorted_fighter_stats.csv",
    'test_data': "train_test/test_data.csv",
    'val_data': "train_test/val_data.csv",
    'removed_features': "train_test/removed_features.txt",
    'live_data': "live_data",
    'matchup_data': "matchup data"
}

IMPORTANT_COLUMNS = [
    'fighter_a', 'fighter_b',
    'current_fight_age', 'current_fight_age_b', 'current_fight_age_diff', 'current_fight_age_ratio',
    'current_fight_win_streak_a', 'current_fight_win_streak_b', 'current_fight_win_streak_diff', 'current_fight_win_streak_ratio',
    'current_fight_loss_streak_a', 'current_fight_loss_streak_b', 'current_fight_loss_streak_diff', 'current_fight_loss_streak_ratio',
    'current_fight_years_experience_a', 'current_fight_years_experience_b', 'current_fight_years_experience_diff', 'current_fight_years_experience_ratio',
    'current_fight_days_since_last_a', 'current_fight_days_since_last_b', 'current_fight_days_since_last_diff', 'current_fight_days_since_last_ratio',
    'current_fight_pre_fight_elo_a', 'current_fight_pre_fight_elo_b', 'current_fight_pre_fight_elo_diff',
    'current_fight_pre_fight_elo_a_win_chance', 'current_fight_pre_fight_elo_b_win_chance', 'current_fight_pre_fight_elo_ratio',
    'current_fight_closing_open_diff_a', 'current_fight_closing_open_diff_b'
]

EXCLUDED_COLUMNS = [
    'fighter', 'fighter_lower', 'id', 'fighter_b', 'fight_date', 'fight_date_b',
    'result', 'winner', 'weight_class', 'scheduled_rounds',
    'result_b', 'winner_b', 'weight_class_b', 'scheduled_rounds_b'
]

# ============================================================================
# CALIBRATION CONFIGURATION
# ============================================================================
USE_CALIBRATION = True                  # Enable/disable probability calibration
CALIBRATION_TYPE = "isotonic"           # 'isotonic' or 'sigmoid'
CALIBRATION_BACKEND = "cv"              # 'cv' (cross-validation) or 'simple'

# Calibration Type Guide:
#   isotonic  - More flexible, better for non-linear relationships
#   sigmoid   - Simpler, uses Platt scaling
#
# Calibration Backend Guide:
#   cv        - CalibratedClassifierCV with cv='prefit' (recommended)
#   simple    - Isotonic/Sigmoid fitted directly on validation data

EXAMPLE_MATCHUPS = [
    {
        'fighter_a': "Andre Fili",
        'fighter_b': "Christian Rodriguez",
        'open_odds_a': None,
        'open_odds_b': None,
        'closing_odds_a': None,
        'closing_odds_b': None,
        'fight_date': "2025-08-09"
    }
]


if __name__ == "__main__":
    start_time = datetime.now()
    main()
    end_time = datetime.now()
    print(f"Total runtime: {end_time - start_time}")
