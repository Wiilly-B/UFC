"""
MMA Betting Analysis System
Main module containing core logic, configuration, and entry point
"""
import os
import sys
import re  # <-- added
import pickle
import numpy as np
import pandas as pd
import xgboost as xgb
from io import StringIO
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from sklearn.calibration import CalibratedClassifierCV
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from rich.console import Console
from rich.panel import Panel
from rich.table import Table


@dataclass
class Config:
    """Configuration for MMA betting analysis"""
    # Threshold settings
    manual_threshold: float = 0.5

    # Calibration settings
    use_calibration: bool = True
    calibration_type: str = 'isotonic'

    # Betting strategy parameters
    initial_bankroll: float = 10000
    kelly_fraction: float = 0.5  # Fraction of Kelly criterion to use
    fixed_bet_fraction: float = 0.1  # Fraction of bankroll for fixed bets
    max_bet_percentage: float = 0.1  # Maximum percentage of bankroll to bet

    # Odds settings
    min_odds: int = -300  # Minimum odds to place a bet
    max_underdog_odds: int = 200  # Maximum underdog odds to place a bet
    odds_type: str = 'close'  # Options: 'open', 'close', 'average'

    # Model settings
    use_ensemble: bool = True  # Whether to use ensemble of models
    model_files: List[str] = field(default_factory=lambda: [
        # 'run1_final_model_20251007_022504.json',
        # 'run2_final_model_20251007_023852.json',
        # 'run3_final_model_20251007_025103.json',
        # 'run4_final_model_20251007_030437.json',
        # 'run5_final_model_20251007_031843.json'
    ])
    # optional folder-based auto-discovery (all files named like runX*.json)
    model_dir: Optional[str] = None
    model_filename_pattern: str = r'^run\d.*\.json$'

    # Data paths
    val_data_path: str = '../../../data/train_test/val_data.csv'
    test_data_path: str = '../../../data/train_test/test_data.csv'
    # test_data_path: str = '../../../data/matchup data/all_matchups.csv'
    encoder_path: str = '../../../saved_models/encoders/category_encoder.pkl'
    model_base_path: str = '../../../saved_models/xgboost/new_features_15y2/'

    # Display settings
    display_columns: List[str] = field(default_factory=lambda: ['current_fight_date', 'fighter_a', 'fighter_b'])
    output_dir: str = '../../outputs/calibration_plots'


class CategoryEncoder:
    """Ensures consistent categorical encoding across different datasets"""

    def __init__(self):
        self.category_mappings = {}
        self.initialized = False

    def fit(self, data: pd.DataFrame) -> 'CategoryEncoder':
        """Learn category mappings from reference data"""
        category_columns = [col for col in data.columns if col.endswith(('fight_1', 'fight_2', 'fight_3'))]

        for col in category_columns:
            unique_values = data[col].dropna().unique()
            self.category_mappings[col] = {val: i for i, val in enumerate(unique_values)}

        self.initialized = True
        return self

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply consistent categorical mappings"""
        if not self.initialized:
            raise ValueError("CategoryEncoder must be fit before transform")

        data_copy = data.copy()
        for col, mapping in self.category_mappings.items():
            if col in data_copy.columns:
                data_copy[col] = data_copy[col].map(mapping).fillna(-1).astype('int32')

        return data_copy

    def fit_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Fit and transform in one step"""
        return self.fit(data).transform(data)

    def save(self, filepath: str):
        """Save encoder to disk"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(self.category_mappings, f)

    @classmethod
    def load(cls, filepath: str) -> 'CategoryEncoder':
        """Load encoder from disk"""
        encoder = cls()
        with open(filepath, 'rb') as f:
            encoder.category_mappings = pickle.load(f)
        encoder.initialized = True
        return encoder


class ModelManager:
    """Manages model loading, calibration, and predictions"""

    def __init__(self, config: Config):
        self.config = config
        self.models = []
        self.encoder = None

    def load_models(self) -> List[Any]:
        """Load trained models from disk

        Behavior:
        - If config.model_dir is set: load all files in that folder whose names
          match config.model_filename_pattern (e.g., '^run\\d.*\\.json$').
        - Else: fall back to prior behavior using model_files (and use_ensemble flag).
        """
        self.models = []

        # --- New folder auto-discovery path ---
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

            # Load all discovered models (ensemble)
            for path in sorted(candidates):
                self.models.append(self._load_single_model(path))
            return self.models

        # --- Original behavior (explicit list) ---
        if self.config.use_ensemble:
            for model_file in self.config.model_files:
                model_path = os.path.abspath(f"{self.config.model_base_path}/{model_file}")
                self.models.append(self._load_single_model(model_path))
        else:
            model_path = os.path.abspath(f"{self.config.model_base_path}/{self.config.model_files[0]}")
            self.models.append(self._load_single_model(model_path))
        return self.models

    def _load_single_model(self, model_path: str) -> Any:
        """Load a single XGBoost model"""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        model = xgb.XGBClassifier(enable_categorical=True)
        model.load_model(model_path)
        return model

    def prepare_datasets(self) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame]:
        """Load and prepare validation and test datasets"""
        val_data = pd.read_csv(self.config.val_data_path)
        test_data = pd.read_csv(self.config.test_data_path)

        y_val, y_test = val_data['winner'], test_data['winner']

        # Handle encoder
        if os.path.exists(self.config.encoder_path):
            self.encoder = CategoryEncoder.load(self.config.encoder_path)
            X_val = self.encoder.transform(val_data.drop(['winner'] + self.config.display_columns, axis=1))
        else:
            self.encoder = CategoryEncoder()
            X_val = self.encoder.fit_transform(val_data.drop(['winner'] + self.config.display_columns, axis=1))
            self.encoder.save(self.config.encoder_path)

        X_test = self.encoder.transform(test_data.drop(['winner'] + self.config.display_columns, axis=1))

        test_data_with_display = pd.concat([X_test, test_data[self.config.display_columns], y_test], axis=1)
        X_test = test_data_with_display.drop(self.config.display_columns + ['winner'], axis=1)
        y_test = test_data_with_display['winner']

        return X_val, y_val, X_test, y_test, test_data_with_display

    def calibrate_predictions(self, X_val: pd.DataFrame, y_val: pd.Series, X_test: pd.DataFrame) -> List[np.ndarray]:
        """Apply calibration to model predictions"""
        if not self.config.use_calibration:
            return self._get_uncalibrated_predictions(X_test)

        # Only standard calibration remains (isotonic or sigmoid via CalibratedClassifierCV)
        return self._apply_standard_calibration(X_val, y_val, X_test)

    def _get_uncalibrated_predictions(self, X_test: pd.DataFrame) -> List[np.ndarray]:
        y_pred_proba_list = []
        for model in self.models:
            # single vectorized call per model
            y_pred_proba_list.append(model.predict_proba(X_test))
        return y_pred_proba_list

    def _apply_standard_calibration(self, X_val: pd.DataFrame, y_val: pd.Series, X_test: pd.DataFrame) -> List[
        np.ndarray]:
        calibrated_models = []
        for model in self.models:
            calibrated_model = CalibratedClassifierCV(model, cv='prefit', method=self.config.calibration_type)
            calibrated_model.fit(X_val, y_val)
            calibrated_models.append(calibrated_model)

        y_pred_proba_list = []
        for model in calibrated_models:
            # single vectorized call per model
            y_pred_proba_list.append(model.predict_proba(X_test))
        return y_pred_proba_list


def main(config: Optional[Config] = None):
    """Main execution function for MMA betting analysis"""
    if config is None:
        config = Config()

    # Redirect stdout to capture output
    old_stdout = sys.stdout
    sys.stdout = mystdout = StringIO()

    console = Console()

    try:
        # Initialize model manager
        model_manager = ModelManager(config)

        # Load models
        print("Loading models...")
        model_manager.load_models()

        # Prepare datasets
        print("Preparing datasets...")
        X_val, y_val, X_test, y_test, test_data_with_display = model_manager.prepare_datasets()

        # Ensure consistent feature ordering
        expected_features = model_manager.models[0].get_booster().feature_names
        X_val = X_val.reindex(columns=expected_features)
        X_test = X_test.reindex(columns=expected_features)

        # CAST to float32 for faster predict (keeps structure unchanged)
        X_val = _safe_downcast_float32(X_val)
        X_test = _safe_downcast_float32(X_test)

        # Apply calibration and get predictions
        print(
            f"Applying {config.calibration_type} calibration..." if config.use_calibration else "Using uncalibrated models...")
        y_pred_proba_list = model_manager.calibrate_predictions(X_val, y_val, X_test)

        # Check if there are enough samples
        if len(test_data_with_display) == 0:
            print("[bold red]Error: No test data available for evaluation[/bold red]")
            return

        # Import betting module (we'll create this next)
        from betting_module import BettingEvaluator

        # Evaluate bets
        evaluator = BettingEvaluator(config)
        bet_results = evaluator.evaluate_bets(y_test, y_pred_proba_list, test_data_with_display)

        # Print results
        evaluator.print_results(bet_results, test_data_with_display)

        # Calculate overall metrics
        if config.use_ensemble:
            y_pred_proba_avg = np.mean(y_pred_proba_list, axis=0)
            y_pred = np.array([1 if proba[1] > proba[0] else 0 for proba in y_pred_proba_avg])
        else:
            y_pred_proba_avg = y_pred_proba_list[0]
            y_pred = np.array([1 if proba[1] > proba[0] else 0 for proba in y_pred_proba_avg])

        # Print overall metrics
        print_overall_metrics(y_test, y_pred, y_pred_proba_avg)

        # Generate calibration plots
        try:
            from visualization import create_calibration_plots
            create_calibration_plots(y_test, y_pred_proba_list, config)
        except Exception as e:
            print(f"\n[Warning] Error generating calibration plots: {str(e)}")
            print("Continuing with analysis without calibration plots.")

    finally:
        # Restore stdout and print final output
        sys.stdout = old_stdout
        output = mystdout.getvalue()

        console = Console(width=93)
        main_panel = Panel(
            output,
            title=f"MMA Betting Analysis (Odds: {config.odds_type.capitalize()}, Calibration: {config.calibration_type.capitalize()})",
            border_style="bold magenta",
            expand=True,
        )
        console.print(main_panel)


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
        "AUC": lambda: roc_auc_score(y_test, y_pred_proba[:, 1]) if len(np.unique(y_test)) > 1 else None
    }

    for metric_name, metric_func in metrics.items():
        try:
            value = metric_func()
            if value is not None:
                table.add_row(metric_name, f"{value:.4f}")
            else:
                table.add_row(metric_name, "Not available (only one class)")
        except Exception as e:
            table.add_row(metric_name, "Not available")
            print(f"Warning: Could not calculate {metric_name} - {str(e)}")

    console.print(table)


def _safe_downcast_float32(df: pd.DataFrame) -> pd.DataFrame:
    """Downcast ONLY float columns to float32 when safe (avoids overflow warnings)."""
    max32 = np.finfo(np.float32).max
    float_cols = df.select_dtypes(include=[np.floating]).columns
    for c in float_cols:
        col = df[c]
        with np.errstate(over="ignore", invalid="ignore"):
            max_abs = np.nanmax(np.abs(col.to_numpy(dtype=np.float64, copy=False)))
        if np.isfinite(max_abs) and max_abs <= max32:
            df[c] = col.astype(np.float32, copy=False)
    return df


if __name__ == "__main__":
    # Run with default configuration
    # main()

    # Or customize configuration:
    custom_config = Config(
        manual_threshold=0.50,
        use_calibration=True,
        calibration_type='isotonic',
        initial_bankroll=10000,
        kelly_fraction=0.5,
        fixed_bet_fraction=0.1,
        model_dir='../../../saved_models/xgboost/no_odds_high_reg_120f/',
        use_ensemble=True
    )
    main(custom_config)
