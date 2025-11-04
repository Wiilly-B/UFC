"""
Create Individual Matchup - Prediction with Calibration
Creates a matchup between two fighters, runs it through trained models,
and displays calibrated probabilities like in predict_testset.
"""
import os
import sys
import pickle
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.calibration import CalibratedClassifierCV
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from typing import List, Any

# Add project root to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from src.data_processing.cleaning.data_cleaner import MatchupProcessor

# ============================== ENCODER ==============================

class CategoryEncoder:
    """Categorical column encoder with fit/transform/save/load."""

    def __init__(self):
        self.category_mappings = {}
        self.initialized = False

    @classmethod
    def load(cls, filepath: str):
        enc = cls()
        with open(filepath, 'rb') as f:
            enc.category_mappings = pickle.load(f)
        enc.initialized = True
        return enc

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        if not self.initialized:
            raise ValueError("CategoryEncoder must be fit before transform")
        df = data.copy()
        for col, mapping in self.category_mappings.items():
            if col in df.columns:
                df[col] = df[col].map(mapping).fillna(-1).astype('int32')
        return df


# ============================== MODEL UTILITIES ==============================

def load_models(model_dir: str, pattern: str = r'ufc_xgb_single_(?:TRIAL\d{3}|FINAL).*\.json$') -> List[Any]:
    """Load all models matching pattern from directory."""
    import re
    folder = os.path.abspath(model_dir)
    if not os.path.isdir(folder):
        raise FileNotFoundError(f"Model directory not found: {folder}")

    pattern_regex = re.compile(pattern)
    candidates = sorted([os.path.join(folder, f) for f in os.listdir(folder) if pattern_regex.match(f)])

    if not candidates:
        raise FileNotFoundError(f"No model files in {folder} matching pattern")

    models = []
    for path in candidates:
        model = xgb.XGBClassifier(enable_categorical=True)
        model.load_model(path)
        # XGBoost 3.x has classes_ as read-only property
        # Use object.__setattr__ to bypass the property setter for calibration compatibility
        object.__setattr__(model, '_classes', np.array([0, 1]))
        models.append(model)

    print(f"Loaded {len(models)} models from {folder}")
    return models


def align_features(model: Any, X: pd.DataFrame) -> pd.DataFrame:
    """Align X to model's expected columns."""
    feats = model.get_booster().feature_names or list(X.columns)
    X_aligned = X.reindex(columns=feats).fillna(0)
    return safe_downcast_float32(X_aligned)


def safe_downcast_float32(df: pd.DataFrame) -> pd.DataFrame:
    """Downcast float columns to float32 when safe."""
    max32 = np.finfo(np.float32).max
    for c in df.select_dtypes(include=[np.floating]).columns:
        col = df[c]
        with np.errstate(over="ignore", invalid="ignore"):
            max_abs = np.nanmax(np.abs(col.to_numpy(dtype=np.float64, copy=False)))
        if np.isfinite(max_abs) and max_abs <= max32:
            df[c] = col.astype(np.float32, copy=False)
    return df


def calibrate_predictions(models: List[Any], X_val: pd.DataFrame, y_val: pd.Series,
                         X_test: pd.DataFrame, calibration_type: str = 'isotonic',
                         calibration_backend: str = 'cv') -> List[np.ndarray]:
    """Apply calibration to model predictions."""
    backend = calibration_backend.lower()

    if backend == 'cv':
        return calibrate_cv(models, X_val, y_val, X_test, calibration_type)
    elif backend == 'simple':
        return calibrate_simple(models, X_val, y_val, X_test, calibration_type)
    else:
        raise ValueError(f"Unknown calibration_backend: {backend}")


def calibrate_cv(models: List[Any], X_val: pd.DataFrame, y_val: pd.Series,
                 X_test: pd.DataFrame, calibration_type: str) -> List[np.ndarray]:
    """Calibrate using CalibratedClassifierCV."""
    preds = []
    typ = calibration_type.lower()
    for model in models:
        Xv, Xt = align_features(model, X_val), align_features(model, X_test)
        cal = CalibratedClassifierCV(model, cv='prefit', method=typ)
        cal.fit(Xv, y_val)
        preds.append(cal.predict_proba(Xt))
    return preds


def calibrate_simple(models: List[Any], X_val: pd.DataFrame, y_val: pd.Series,
                    X_test: pd.DataFrame, calibration_type: str) -> List[np.ndarray]:
    """Calibrate using simple isotonic or sigmoid."""
    preds, typ = [], calibration_type.lower()
    if typ == 'isotonic':
        for model in models:
            Xv, Xt = align_features(model, X_val), align_features(model, X_test)
            val_p, tst_p = model.predict_proba(Xv)[:, 1], model.predict_proba(Xt)[:, 1]
            iso = IsotonicRegression(out_of_bounds='clip')
            iso.fit(val_p, y_val)
            p = iso.transform(tst_p)
            preds.append(np.vstack([1 - p, p]).T)
    elif typ == 'sigmoid':
        for model in models:
            Xv, Xt = align_features(model, X_val), align_features(model, X_test)
            val_p, tst_p = model.predict_proba(Xv)[:, 1].reshape(-1, 1), model.predict_proba(Xt)[:, 1].reshape(-1, 1)
            lr = LogisticRegression(solver='lbfgs')
            lr.fit(val_p, y_val)
            preds.append(lr.predict_proba(tst_p))
    else:
        raise ValueError(f"Unsupported calibration_type: {typ}")
    return preds


# ============================== MAIN ==============================

def main():
    """Create and predict an individual matchup using trained models and calibration."""
    # Initialize the processor
    mp = MatchupProcessor(data_dir="../../../data")

    # Create matchup with odds and date
    print("=" * 80)
    print("CREATING INDIVIDUAL MATCHUP WITH FEATURE SELECTION")
    print("=" * 80)

    matchup = mp.create_individual_matchup(
        fighter_a_name=FIGHTER_A_NAME,
        fighter_b_name=FIGHTER_B_NAME,
        reference_date=REFERENCE_DATE,
        fighter_a_odds=FIGHTER_A_ODDS,
        fighter_b_odds=FIGHTER_B_ODDS,
        fighter_a_closing_odds=FIGHTER_A_CLOSING_ODDS,
        fighter_b_closing_odds=FIGHTER_B_CLOSING_ODDS,
        n_past_fights=N_PAST_FIGHTS,
        n_detailed_results=N_DETAILED_RESULTS,
        apply_feature_selection=APPLY_FEATURE_SELECTION,
    )

    print("\n" + "=" * 80)
    print("MATCHUP SUMMARY")
    print("=" * 80)

    # Display key information
    fighter_a = matchup['fighter_a'].iloc[0]
    fighter_b = matchup['fighter_b'].iloc[0]
    print(f"\nFighters: {fighter_a} vs {fighter_b}")
    print(f"Total Features: {len(matchup.columns)}")

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
    print("LOADING MODELS AND RUNNING PREDICTION")
    print("=" * 80)

    # Load models
    print("\n1. Loading models...")
    models = load_models(MODEL_DIR)

    # Load encoder
    print("\n2. Loading encoder...")
    encoder = CategoryEncoder.load(ENCODER_PATH)

    # Load validation data for calibration
    print("\n3. Loading validation data for calibration...")
    val_data = pd.read_csv(VAL_DATA_PATH)
    y_val = val_data['winner']
    display_columns = ['current_fight_date', 'fighter_a', 'fighter_b']
    feature_drop = ['winner'] + display_columns
    X_val = encoder.transform(val_data.drop(feature_drop, axis=1))

    # Prepare test data (individual matchup)
    print("\n4. Preparing matchup features...")
    X_test = encoder.transform(matchup.drop(display_columns + ['winner'], axis=1))

    # Apply calibration
    print(f"\n5. Applying calibration ({CALIBRATION_TYPE}/{CALIBRATION_BACKEND})...")
    y_pred_proba_list = calibrate_predictions(
        models, X_val, y_val, X_test,
        calibration_type=CALIBRATION_TYPE,
        calibration_backend=CALIBRATION_BACKEND
    )

    # Calculate ensemble average if using ensemble
    if USE_ENSEMBLE:
        y_pred_proba_avg = np.mean(y_pred_proba_list, axis=0)
    else:
        y_pred_proba_avg = y_pred_proba_list[0]

    # Extract probabilities
    prob_fighter_b_wins = y_pred_proba_avg[0, 0]  # Probability fighter B wins (winner=0)
    prob_fighter_a_wins = y_pred_proba_avg[0, 1]  # Probability fighter A wins (winner=1)

    # Display results
    print("\n" + "=" * 80)
    print("PREDICTION RESULTS")
    print("=" * 80)

    print(f"\nFight: {fighter_a} vs {fighter_b}")
    print(f"Date: {matchup['current_fight_date'].iloc[0]}")
    print(f"\n--- CALIBRATED PROBABILITIES ---")
    print(f"{fighter_a} wins: {prob_fighter_a_wins:.2%}")
    print(f"{fighter_b} wins: {prob_fighter_b_wins:.2%}")

    # Determine predicted winner
    if prob_fighter_a_wins > prob_fighter_b_wins:
        predicted_winner = fighter_a
        confidence = prob_fighter_a_wins
    else:
        predicted_winner = fighter_b
        confidence = prob_fighter_b_wins

    print(f"\n--- PREDICTION ---")
    print(f"Predicted Winner: {predicted_winner}")
    print(f"Confidence: {confidence:.2%}")

    # Model agreement (if ensemble)
    if USE_ENSEMBLE and len(y_pred_proba_list) > 1:
        agreements = sum([
            1 for proba in y_pred_proba_list
            if (proba[0, 1] > proba[0, 0]) == (prob_fighter_a_wins > prob_fighter_b_wins)
        ])
        print(f"Models Agreeing: {agreements}/{len(y_pred_proba_list)}")

    print("\n" + "=" * 80)
    print("Analysis Complete")
    print("=" * 80)


# ============================== CONFIG ==============================

# Model and data paths
MODEL_DIR = '../../../saved_models/xgboost/single_split/'
ENCODER_PATH = '../../../saved_models/encoders/category_encoder.pkl'
VAL_DATA_PATH = '../../../data/train_test/val_data.csv'

# Calibration settings
USE_CALIBRATION = True
CALIBRATION_TYPE = 'isotonic'  # 'isotonic' or 'sigmoid'
CALIBRATION_BACKEND = 'cv'  # 'cv' or 'simple'
USE_ENSEMBLE = True

# Matchup configuration
FIGHTER_A_NAME = "Jeremiah Wells"
FIGHTER_B_NAME = "Themba Gorimbo"
REFERENCE_DATE = "2025-11-01"
FIGHTER_A_ODDS = -550
FIGHTER_B_ODDS = 400
FIGHTER_A_CLOSING_ODDS = -650
FIGHTER_B_CLOSING_ODDS = 450
N_PAST_FIGHTS = 3
N_DETAILED_RESULTS = 3
APPLY_FEATURE_SELECTION = True


if __name__ == "__main__":
    main()
