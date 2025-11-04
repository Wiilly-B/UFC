"""
MMA Betting Analysis System
Supports multiple calibration backends (cv/simple) and ensemble/single models.
"""
import os, sys, re, pickle, json
from io import StringIO
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.calibration import CalibratedClassifierCV
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from rich.console import Console
from rich.panel import Panel
from rich.table import Table


# ============================== ENCODER ==============================

class CategoryEncoder:
    """Categorical column encoder with fit/transform/save/load."""

    def __init__(self):
        self.category_mappings: Dict[str, Dict[Any, int]] = {}
        self.initialized = False

    def fit(self, data: pd.DataFrame):
        cats = [col for col in data.columns if col.endswith(('fight_1', 'fight_2', 'fight_3'))]
        for col in cats:
            self.category_mappings[col] = {val: i for i, val in enumerate(data[col].dropna().unique())}
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
    def load(cls, filepath: str):
        enc = cls()
        with open(filepath, 'rb') as f:
            enc.category_mappings = pickle.load(f)
        enc.initialized = True
        return enc


# ============================== MODEL MANAGER ==============================

class ModelManager:
    """Handles model loading, feature alignment, and calibration."""

    def __init__(self, config):
        self.config = config
        self.models: List[xgb.XGBClassifier] = []
        self.encoder: Optional[CategoryEncoder] = None

    def load_models(self) -> List[Any]:
        self.models = []
        if self.config.model_dir:
            folder = os.path.abspath(self.config.model_dir)
            if not os.path.isdir(folder):
                raise FileNotFoundError(f"Model directory not found: {folder}")
            pattern = re.compile(self.config.model_filename_pattern)
            candidates = sorted([os.path.join(folder, f) for f in os.listdir(folder) if pattern.match(f)])
            if not candidates:
                raise FileNotFoundError(f"No model files in {folder} matching pattern")
            for path in candidates:
                self.models.append(self._load_single_model(path))
            if not self.config.use_ensemble and self.models:
                self.models = [self.models[0]]
            return self.models

        if not self.config.model_files:
            raise ValueError("No models specified. Set model_dir or model_files.")
        paths = [os.path.abspath(os.path.join(self.config.model_base_path, mf)) for mf in self.config.model_files]
        for p in paths if self.config.use_ensemble else paths[:1]:
            self.models.append(self._load_single_model(p))
        return self.models

    def _load_single_model(self, model_path: str) -> xgb.XGBClassifier:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
        model = xgb.XGBClassifier(enable_categorical=True)
        model.load_model(model_path)
        # XGBoost 3.x has classes_ as read-only property
        # Use object.__setattr__ to bypass the property setter for calibration compatibility
        object.__setattr__(model, '_classes', np.array([0, 1]))
        return model

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
                raise FileNotFoundError(f"Encoder not found. Set require_trained_encoder=False to fit on VAL.")
            self.encoder = CategoryEncoder()
            X_val = self.encoder.fit_transform(val_data.drop(feature_drop, axis=1))
            self.encoder.save(self.config.encoder_path)

        X_test = self.encoder.transform(test_data.drop(feature_drop, axis=1))
        test_data_with_display = pd.concat([X_test, test_data[self.config.display_columns], y_test], axis=1)
        X_test = test_data_with_display.drop(self.config.display_columns + ['winner'], axis=1)
        return X_val, y_val, X_test, test_data_with_display['winner'], test_data_with_display

    def _align_features(self, model: Any, X: pd.DataFrame) -> pd.DataFrame:
        """Align X to model's expected columns."""
        feats = model.get_booster().feature_names or list(X.columns)
        X_aligned = X.reindex(columns=feats).fillna(0)
        return _safe_downcast_float32(X_aligned)

    def calibrate_predictions(self, X_val: pd.DataFrame, y_val: pd.Series, X_test: pd.DataFrame) -> List[np.ndarray]:
        if not self.config.use_calibration:
            return [m.predict_proba(self._align_features(m, X_test)) for m in self.models]

        backend = (self.config.calibration_backend or 'cv').lower()
        if backend == 'cv':
            return self._calibrate_cv(X_val, y_val, X_test)
        elif backend == 'simple':
            return self._calibrate_simple(X_val, y_val, X_test)
        else:
            raise ValueError(f"Unknown calibration_backend: {backend}")

    def _calibrate_cv(self, X_val: pd.DataFrame, y_val: pd.Series, X_test: pd.DataFrame) -> List[np.ndarray]:
        preds = []
        typ = (self.config.calibration_type or 'isotonic').lower()
        for model in self.models:
            Xv, Xt = self._align_features(model, X_val), self._align_features(model, X_test)
            cal = CalibratedClassifierCV(model, cv='prefit', method=typ)
            cal.fit(Xv, y_val)
            preds.append(cal.predict_proba(Xt))
        return preds

    def _calibrate_simple(self, X_val: pd.DataFrame, y_val: pd.Series, X_test: pd.DataFrame) -> List[np.ndarray]:
        preds, typ = [], (self.config.calibration_type or 'isotonic').lower()
        if typ == 'isotonic':
            for model in self.models:
                Xv, Xt = self._align_features(model, X_val), self._align_features(model, X_test)
                val_p, tst_p = model.predict_proba(Xv)[:, 1], model.predict_proba(Xt)[:, 1]
                iso = IsotonicRegression(out_of_bounds='clip')
                iso.fit(val_p, y_val)
                p = iso.transform(tst_p)
                preds.append(np.vstack([1 - p, p]).T)
        elif typ == 'sigmoid':
            for model in self.models:
                Xv, Xt = self._align_features(model, X_val), self._align_features(model, X_test)
                val_p, tst_p = model.predict_proba(Xv)[:, 1].reshape(-1, 1), model.predict_proba(Xt)[:, 1].reshape(-1, 1)
                lr = LogisticRegression(solver='lbfgs')
                lr.fit(val_p, y_val)
                preds.append(lr.predict_proba(tst_p))
        else:
            raise ValueError(f"Unsupported calibration_type: {typ}")
        return preds


# ============================== UTILITIES ==============================

def _safe_downcast_float32(df: pd.DataFrame) -> pd.DataFrame:
    """Downcast float columns to float32 when safe."""
    max32 = np.finfo(np.float32).max
    for c in df.select_dtypes(include=[np.floating]).columns:
        col = df[c]
        with np.errstate(over="ignore", invalid="ignore"):
            max_abs = np.nanmax(np.abs(col.to_numpy(dtype=np.float64, copy=False)))
        if np.isfinite(max_abs) and max_abs <= max32:
            df[c] = col.astype(np.float32, copy=False)
    return df


def _print_metrics(y_test: pd.Series, y_pred: np.ndarray, y_pred_proba: np.ndarray):
    """Print overall model performance metrics."""
    console = Console()
    table = Table(title="Overall Model Metrics")
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
            table.add_row(name, f"{val:.4f}" if val is not None else "N/A (one class)")
        except Exception as e:
            table.add_row(name, "N/A")
    console.print(table)


def main(config=None):
    """Main execution function."""
    if config is None:
        config = CONFIG

    old_stdout = sys.stdout
    sys.stdout = mystdout = StringIO()

    try:
        mgr = ModelManager(config)
        print("Loading models...")
        mgr.load_models()

        print("Preparing datasets...")
        X_val, y_val, X_test, y_test, test_display = mgr.prepare_datasets()

        if len(test_display) == 0:
            print("[bold red]Error: No test data[/bold red]")
            return

        if config.use_calibration:
            print(f"Applying calibration: {config.calibration_type}/{config.calibration_backend}")
        else:
            print("Using uncalibrated models...")

        y_pred_proba_list = mgr.calibrate_predictions(X_val, y_val, X_test)

        from betting_module import BettingEvaluator
        evaluator = BettingEvaluator(config)
        bet_results = evaluator.evaluate_bets(y_test, y_pred_proba_list, test_display)
        evaluator.print_results(bet_results, test_display)

        y_pred_proba_avg = np.mean(y_pred_proba_list, axis=0) if config.use_ensemble else y_pred_proba_list[0]
        y_pred = (y_pred_proba_avg[:, 1] > y_pred_proba_avg[:, 0]).astype(int)
        _print_metrics(y_test, y_pred, y_pred_proba_avg)
    finally:
        sys.stdout = old_stdout
        output = mystdout.getvalue()
        console = Console(width=93)
        main_panel = Panel(
            output,
            title=f"MMA Betting Analysis (Odds: {config.odds_type}, "
                  f"Cal: {config.calibration_type}/{config.calibration_backend})",
            border_style="bold magenta",
            expand=True,
        )
        console.print(main_panel)


# ============================== CONFIG ==============================

@dataclass
class Config:
    """Configuration for MMA betting analysis."""

    # === CALIBRATION ===
    use_calibration: bool = True
    calibration_type: str = "isotonic"     # 'isotonic' | 'sigmoid'
    calibration_backend: str = "cv"        # 'cv' | 'simple'
    require_trained_encoder: bool = True

    # === BETTING ===
    manual_threshold: float = 0.5
    initial_bankroll: float = 10000
    kelly_fraction: float = 0.5
    fixed_bet_fraction: float = 0.1
    max_bet_percentage: float = 0.1
    min_odds: int = -300
    max_underdog_odds: int = 200
    odds_type: str = "close"               # 'open' | 'close' | 'average'

    # === MODELS ===
    use_ensemble: bool = True
    model_dir: Optional[str] = None
    model_files: List[str] = field(default_factory=list)
    model_filename_pattern: str = r'ufc_xgb_single_(?:TRIAL\d{3}|FINAL).*\.json$'
    model_base_path: str = '../../../saved_models/xgboost/single_split/'

    # === DATA ===
    val_data_path: str = '../../../data/train_test/val_data.csv'
    test_data_path: str = '../../../data/train_test/test_data.csv'
    encoder_path: str = '../../../saved_models/encoders/category_encoder.pkl'
    display_columns: List[str] = field(default_factory=lambda: ['current_fight_date', 'fighter_a', 'fighter_b'])


# ============================== ENTRYPOINT ==============================

CONFIG = Config(
    use_calibration=True,
    calibration_type='isotonic',
    calibration_backend='cv',
    initial_bankroll=10000,
    kelly_fraction=0.25,
    fixed_bet_fraction=0.1,
    model_dir='../../../saved_models/xgboost/single_split/',
    model_filename_pattern=r'ufc_xgb_single_(?:TRIAL\d{3}|FINAL).*\.json$',
    use_ensemble=True,
    require_trained_encoder=True,
)

if __name__ == "__main__":
    main(CONFIG)
