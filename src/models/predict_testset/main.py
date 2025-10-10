"""
MMA Betting Analysis System
Main module containing core logic, configuration, and entry point
(Rewritten to support BOTH calibration backends: 'cv' and 'simple',
and to align features per-model safely for ensembles of different models.)
"""
import os
import sys
import re
import pickle
from io import StringIO
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)
from rich.console import Console
from rich.panel import Panel
from rich.table import Table


# ============================== Config ==============================

@dataclass
class Config:
    """Configuration for MMA betting analysis"""
    # Decision threshold
    manual_threshold: float = 0.5

    # Calibration controls
    use_calibration: bool = True
    calibration_type: str = "isotonic"     # 'isotonic' or 'sigmoid'
    calibration_backend: str = "cv"        # 'cv' (CalibratedClassifierCV) or 'simple'
    require_trained_encoder: bool = False  # set True to force loading pre-trained encoder

    # Betting strategy (used by betting_module)
    initial_bankroll: float = 10000
    kelly_fraction: float = 0.5
    fixed_bet_fraction: float = 0.1
    max_bet_percentage: float = 0.1

    # Odds filters (used by betting_module)
    min_odds: int = -300
    max_underdog_odds: int = 200
    odds_type: str = "close"               # 'open' | 'close' | 'average'

    # Model loading
    use_ensemble: bool = True              # if False, use first matched model only
    model_files: List[str] = field(default_factory=list)  # legacy explicit list
    model_dir: Optional[str] = None        # prefer this: folder auto-discovery
    # Matches your autosaved TRIALs and FINAL from the trainer:
    model_filename_pattern: str = r'ufc_xgb_single_(?:TRIAL\d{3}|FINAL).*\.json$'
    # legacy base path if using model_files
    model_base_path: str = '../../../saved_models/xgboost/new_features_15y2/'

    # Data paths
    val_data_path: str = '../../../data/train_test/val_data.csv'
    test_data_path: str = '../../../data/train_test/test_data.csv'
    encoder_path: str = '../../../saved_models/encoders/category_encoder.pkl'

    # Display / output
    display_columns: List[str] = field(default_factory=lambda: ['current_fight_date', 'fighter_a', 'fighter_b'])
    output_dir: str = '../../outputs/calibration_plots'


# ========================== Category Encoder ==========================

class CategoryEncoder:
    """
    Simple mapping encoder for categorical columns ending with 'fight_1/2/3'.
    Use the SAME mappings as training to avoid leakage / drift.
    """

    def __init__(self):
        self.category_mappings: Dict[str, Dict[Any, int]] = {}
        self.initialized = False

    def fit(self, data: pd.DataFrame) -> 'CategoryEncoder':
        category_columns = [col for col in data.columns if col.endswith(('fight_1', 'fight_2', 'fight_3'))]
        for col in category_columns:
            unique_values = data[col].dropna().unique()
            self.category_mappings[col] = {val: i for i, val in enumerate(unique_values)}
        self.initialized = True
        return self

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        if not self.initialized:
            raise ValueError("CategoryEncoder must be fit before transform")
        df = data.copy()
        for col, mapping in self.category_mappings.items():
            if col in df.columns:
                df[col] = df[col].map(mapping).fillna(-1).astype('int32')
        return df

    def fit_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        return self.fit(data).transform(data)

    def save(self, filepath: str):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(self.category_mappings, f)

    @classmethod
    def load(cls, filepath: str) -> 'CategoryEncoder':
        enc = cls()
        with open(filepath, 'rb') as f:
            enc.category_mappings = pickle.load(f)
        enc.initialized = True
        return enc


# ============================= Model I/O ==============================

class ModelManager:
    """Manages model loading, strict per-model feature alignment, and calibrated predictions."""
    def __init__(self, config: Config):
        self.config = config
        self.models: List[xgb.XGBClassifier] = []
        self.encoder: Optional[CategoryEncoder] = None

    # ---------- load models ----------
    def load_models(self) -> List[Any]:
        self.models = []
        # Folder auto-discovery
        if self.config.model_dir:
            folder = os.path.abspath(self.config.model_dir)
            if not os.path.isdir(folder):
                raise FileNotFoundError(f"Model directory not found: {folder}")
            pattern = re.compile(self.config.model_filename_pattern)
            candidates = [
                os.path.join(folder, f)
                for f in os.listdir(folder)
                if pattern.match(f)
            ]
            if not candidates:
                raise FileNotFoundError(
                    f"No model files in {folder} matching pattern {self.config.model_filename_pattern}"
                )
            for path in sorted(candidates):
                self.models.append(self._load_single_model(path))
            if not self.config.use_ensemble and self.models:
                self.models = [self.models[0]]
            return self.models

        # Legacy explicit list
        if not self.config.model_files:
            raise ValueError("No models specified. Set model_dir or model_files.")
        paths = [os.path.abspath(os.path.join(self.config.model_base_path, mf)) for mf in self.config.model_files]
        if self.config.use_ensemble:
            for p in paths:
                self.models.append(self._load_single_model(p))
        else:
            self.models.append(self._load_single_model(paths[0]))
        return self.models

    def _load_single_model(self, model_path: str) -> Any:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        model = xgb.XGBClassifier(enable_categorical=True)
        model.load_model(model_path)
        return model

    # ---------- data prep ----------
    def prepare_datasets(self) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame]:
        val_data = pd.read_csv(self.config.val_data_path)
        test_data = pd.read_csv(self.config.test_data_path)

        y_val, y_test = val_data['winner'], test_data['winner']

        feature_drop = ['winner'] + self.config.display_columns
        if os.path.exists(self.config.encoder_path):
            self.encoder = CategoryEncoder.load(self.config.encoder_path)
            X_val = self.encoder.transform(val_data.drop(feature_drop, axis=1))
        else:
            if self.config.require_trained_encoder:
                raise FileNotFoundError(
                    f"Encoder not found at {self.config.encoder_path}. "
                    "Set require_trained_encoder=False to fit on VAL (not recommended)."
                )
            # fallback (not recommended): fit on VAL then save
            self.encoder = CategoryEncoder()
            X_val = self.encoder.fit_transform(val_data.drop(feature_drop, axis=1))
            self.encoder.save(self.config.encoder_path)

        X_test = self.encoder.transform(test_data.drop(feature_drop, axis=1))

        # Keep display columns with test copy for reporting
        test_data_with_display = pd.concat([X_test, test_data[self.config.display_columns], y_test], axis=1)
        X_test = test_data_with_display.drop(self.config.display_columns + ['winner'], axis=1)
        y_test = test_data_with_display['winner']
        return X_val, y_val, X_test, y_test, test_data_with_display

    # ---------- alignment + calibration ----------
    def _align_to_model_features(self, model: Any, X: pd.DataFrame) -> pd.DataFrame:
        """Return X with columns exactly as the model expects (missing -> 0)."""
        feats = model.get_booster().feature_names or list(X.columns)
        X_aligned = X.reindex(columns=feats)
        X_aligned = X_aligned.fillna(0)
        return _safe_downcast_float32(X_aligned)

    def _validate_calibration_type(self):
        typ = (self.config.calibration_type or '').lower()
        if typ not in ("isotonic", "sigmoid"):
            raise ValueError(f"Unsupported calibration_type: {self.config.calibration_type}")

    def _get_uncalibrated_predictions(self, X_test: pd.DataFrame) -> List[np.ndarray]:
        preds: List[np.ndarray] = []
        for model in self.models:
            Xt = self._align_to_model_features(model, X_test)
            preds.append(model.predict_proba(Xt))
        return preds

    def calibrate_predictions(self, X_val: pd.DataFrame, y_val: pd.Series, X_test: pd.DataFrame) -> List[np.ndarray]:
        """Apply calibration to model predictions based on backend + type."""
        if not self.config.use_calibration:
            return self._get_uncalibrated_predictions(X_test)

        backend = (self.config.calibration_backend or 'cv').lower()
        if backend == 'cv':
            return self._apply_standard_calibration(X_val, y_val, X_test)
        elif backend == 'simple':
            return self._apply_simple_calibration(X_val, y_val, X_test)
        else:
            raise ValueError(f"Unknown calibration_backend: {self.config.calibration_backend}")

    # ---- CV backend (CalibratedClassifierCV, your original behavior) ----
    def _apply_standard_calibration(self, X_val: pd.DataFrame, y_val: pd.Series, X_test: pd.DataFrame) -> List[np.ndarray]:
        self._validate_calibration_type()
        calibrated_models = []
        for model in self.models:
            Xv = self._align_to_model_features(model, X_val)
            Xt = self._align_to_model_features(model, X_test)
            cal = CalibratedClassifierCV(model, cv='prefit', method=self.config.calibration_type)
            cal.fit(Xv, y_val)
            calibrated_models.append((cal, Xt))

        y_pred_proba_list = []
        for cal, Xt in calibrated_models:
            y_pred_proba_list.append(cal.predict_proba(Xt))
        return y_pred_proba_list

    # ---- Simple backend (direct isotonic or Platt/logistic; faster) ----
    def _apply_simple_calibration(self, X_val: pd.DataFrame, y_val: pd.Series, X_test: pd.DataFrame) -> List[np.ndarray]:
        typ = (self.config.calibration_type or 'isotonic').lower()
        out: List[np.ndarray] = []

        if typ == 'isotonic':
            from sklearn.isotonic import IsotonicRegression
            for model in self.models:
                Xv = self._align_to_model_features(model, X_val)
                Xt = self._align_to_model_features(model, X_test)
                val_p = model.predict_proba(Xv)[:, 1]
                tst_p = model.predict_proba(Xt)[:, 1]
                iso = IsotonicRegression(out_of_bounds='clip')
                iso.fit(val_p, y_val)
                p = iso.transform(tst_p)
                out.append(np.vstack([1 - p, p]).T)
            return out

        if typ == 'sigmoid':
            from sklearn.linear_model import LogisticRegression
            for model in self.models:
                Xv = self._align_to_model_features(model, X_val)
                Xt = self._align_to_model_features(model, X_test)
                val_p = model.predict_proba(Xv)[:, 1].reshape(-1, 1)
                tst_p = model.predict_proba(Xt)[:, 1].reshape(-1, 1)
                lr = LogisticRegression(solver='lbfgs')
                lr.fit(val_p, y_val)
                out.append(lr.predict_proba(tst_p))
            return out

        raise ValueError(f"Unsupported calibration_type for simple backend: {self.config.calibration_type}")


# ============================= Main logic =============================

def main(config: Optional[Config] = None):
    """Main execution function for MMA betting analysis."""
    if config is None:
        config = Config()

    # capture stdout to render with rich
    old_stdout = sys.stdout
    sys.stdout = mystdout = StringIO()
    console = Console()

    try:
        model_manager = ModelManager(config)

        print("Loading models...")
        model_manager.load_models()

        print("Preparing datasets...")
        X_val, y_val, X_test, y_test, test_data_with_display = model_manager.prepare_datasets()

        if len(test_data_with_display) == 0:
            print("[bold red]Error: No test data available for evaluation[/bold red]")
            return

        # Calibrated predictions (per-model alignment handled internally)
        if config.use_calibration:
            print(f"Applying calibration: type={config.calibration_type}, backend={config.calibration_backend}")
        else:
            print("Using uncalibrated models...")
        y_pred_proba_list = model_manager.calibrate_predictions(X_val, y_val, X_test)

        # Betting evaluation
        from betting_module import BettingEvaluator
        evaluator = BettingEvaluator(config)
        bet_results = evaluator.evaluate_bets(y_test, y_pred_proba_list, test_data_with_display)
        evaluator.print_results(bet_results, test_data_with_display)

        # Aggregate metrics
        if config.use_ensemble:
            y_pred_proba_avg = np.mean(y_pred_proba_list, axis=0)
        else:
            y_pred_proba_avg = y_pred_proba_list[0]

        y_pred = (y_pred_proba_avg[:, 1] > y_pred_proba_avg[:, 0]).astype(int)
        print_overall_metrics(y_test, y_pred, y_pred_proba_avg)

        # Optional calibration plots
        try:
            from visualization import create_calibration_plots
            create_calibration_plots(y_test, y_pred_proba_list, config)
        except Exception as e:
            print(f"\n[Warning] Error generating calibration plots: {str(e)}")
            print("Continuing with analysis without calibration plots.")

    finally:
        # restore stdout and show panel
        sys.stdout = old_stdout
        output = mystdout.getvalue()

        console = Console(width=93)
        main_panel = Panel(
            output,
            title=f"MMA Betting Analysis (Odds: {config.odds_type.capitalize()}, "
                  f"Calibration: {config.calibration_type.capitalize()} / {config.calibration_backend.upper()})",
            border_style="bold magenta",
            expand=True,
        )
        console.print(main_panel)


# ============================= Utilities =============================

def print_overall_metrics(y_test: pd.Series, y_pred: np.ndarray, y_pred_proba: np.ndarray):
    """Print overall model performance metrics"""
    console = Console()
    table = Table(title="Overall Model Metrics (all predictions)")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", justify="right", style="magenta")

    metrics = {
        "Accuracy": lambda: accuracy_score(y_test, y_pred),
        "Precision": lambda: precision_score(y_test, y_pred),
        "Recall": lambda: recall_score(y_test, y_pred),
        "F1 Score": lambda: f1_score(y_test, y_pred),
        "AUC": lambda: roc_auc_score(y_test, y_pred_proba[:, 1]) if len(np.unique(y_test)) > 1 else None,
    }

    for name, fn in metrics.items():
        try:
            val = fn()
            table.add_row(name, f"{val:.4f}" if val is not None else "Not available (one class)")
        except Exception as e:
            table.add_row(name, "Not available")
            print(f"Warning: Could not calculate {name} - {str(e)}")

    console.print(table)


def _safe_downcast_float32(df: pd.DataFrame) -> pd.DataFrame:
    """Downcast ONLY float columns to float32 when safe (avoids overflow)."""
    max32 = np.finfo(np.float32).max
    float_cols = df.select_dtypes(include=[np.floating]).columns
    for c in float_cols:
        col = df[c]
        with np.errstate(over="ignore", invalid="ignore"):
            max_abs = np.nanmax(np.abs(col.to_numpy(dtype=np.float64, copy=False)))
        if np.isfinite(max_abs) and max_abs <= max32:
            df[c] = col.astype(np.float32, copy=False)
    return df


# ============================== Entrypoint ==============================

if __name__ == "__main__":
    # Example custom configuration
    custom_config = Config(
        manual_threshold=0.50,
        use_calibration=True,
        calibration_type='isotonic',            # or 'sigmoid'
        calibration_backend='cv',               # 'cv' for CalibratedClassifierCV, 'simple' for direct mapping
        initial_bankroll=10000,
        kelly_fraction=0.5,
        fixed_bet_fraction=0.1,
        model_dir='../../../saved_models/xgboost/single_split/',  # autosaved TRIAL/FINAL models
        model_filename_pattern=r'ufc_xgb_single_(?:TRIAL\d{3}|FINAL).*\.json$',
        use_ensemble=False,
        require_trained_encoder=True           # set True if you have the training-time encoder saved
    )
    main(custom_config)
