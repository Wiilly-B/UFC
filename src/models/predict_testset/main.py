"""
MMA Betting Analysis System
Main module containing core logic, configuration, and entry point
"""
import os
import sys
import pickle
import numpy as np
import pandas as pd
import xgboost as xgb
from datetime import datetime
from io import StringIO
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.columns import Columns


@dataclass
class Config:
    """Configuration for MMA betting analysis"""
    # Threshold settings
    manual_threshold: float = 0.5

    # Calibration settings
    use_calibration: bool = True
    calibration_type: str = 'isotonic'
    range_calibration_ranges: List[float] = field(default_factory=lambda: [0.25, 0.45, 0.65, 0.85])

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
        'run1_final_model_20251007_022504.json',
        'run2_final_model_20251007_023852.json',
        'run3_final_model_20251007_025103.json',
        'run4_final_model_20251007_030437.json',
        'run5_final_model_20251007_031843.json'
    ])

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


class RangeBasedCalibrator:
    """Custom calibrator that applies different calibration strategies to different probability ranges"""

    def __init__(self, ranges: Optional[List[float]] = None, method: str = 'isotonic', out_of_bounds: str = 'clip'):
        self.ranges = [0.33, 0.67] if ranges is None else ranges
        self.method = method
        self.out_of_bounds = out_of_bounds
        self.calibrators = {}
        self.min_probs = {}
        self.max_probs = {}

    def _get_region(self, prob: float) -> str:
        """Determine which region a probability belongs to"""
        if prob < self.ranges[0]:
            return 'low'
        elif prob > self.ranges[-1]:
            return 'high'
        else:
            for i in range(len(self.ranges) - 1):
                if self.ranges[i] <= prob <= self.ranges[i + 1]:
                    return f'mid_{i}'
            return 'mid_0'

    def fit(self, probs: np.ndarray, y_true: np.ndarray) -> 'RangeBasedCalibrator':
        """Fit calibration models for each probability region"""
        all_regions = ['low'] + [f'mid_{i}' for i in range(len(self.ranges) - 1)] + ['high']

        # Create masks for each region
        region_masks = {
            'low': probs < self.ranges[0],
            'high': probs > self.ranges[-1]
        }

        for i in range(len(self.ranges) - 1):
            region_masks[f'mid_{i}'] = (probs >= self.ranges[i]) & (probs <= self.ranges[i + 1])

        # Fit calibrators for each region
        for region in all_regions:
            mask = region_masks[region]

            if np.sum(mask) < 10:  # Skip regions with too few samples
                continue

            region_probs = probs[mask]
            region_y = y_true[mask]

            self.min_probs[region] = np.min(region_probs) if len(region_probs) > 0 else 0
            self.max_probs[region] = np.max(region_probs) if len(region_probs) > 0 else 1

            if self.method == 'isotonic':
                self.calibrators[region] = IsotonicRegression(out_of_bounds=self.out_of_bounds)
                self.calibrators[region].fit(region_probs, region_y)
            elif self.method == 'sigmoid':
                lr = LogisticRegression(C=1.0)
                lr.fit(region_probs.reshape(-1, 1), region_y)
                self.calibrators[region] = lr
            else:
                raise ValueError(f"Unknown calibration method: {self.method}")

        return self

    def transform(self, probs: np.ndarray) -> np.ndarray:
        """Apply calibration transformation to new probabilities"""
        calibrated_probs = np.zeros_like(probs)

        for i, prob in enumerate(probs):
            region = self._get_region(prob)

            # If calibrator for this region doesn't exist, use the closest one
            if region not in self.calibrators:
                available_regions = list(self.calibrators.keys())
                if len(available_regions) == 0:
                    calibrated_probs[i] = prob
                    continue

                # Find closest region
                region_values = {
                    'low': self.ranges[0] / 2,
                    'high': (1 + self.ranges[-1]) / 2
                }
                for j in range(len(self.ranges) - 1):
                    region_values[f'mid_{j}'] = (self.ranges[j] + self.ranges[j + 1]) / 2

                current_region_value = region_values.get(region, 0.5)
                region = min(available_regions, key=lambda r: abs(region_values.get(r, 0.5) - current_region_value))

            # Apply calibration
            if self.method == 'isotonic':
                calibrated_probs[i] = self.calibrators[region].predict([prob])[0]
            elif self.method == 'sigmoid':
                if self.out_of_bounds == 'clip':
                    prob = np.clip(prob, self.min_probs[region], self.max_probs[region])
                calibrated_probs[i] = self.calibrators[region].predict_proba([[prob]])[0, 1]

        return calibrated_probs


class ModelManager:
    """Manages model loading, calibration, and predictions"""

    def __init__(self, config: Config):
        self.config = config
        self.models = []
        self.encoder = None

    def load_models(self) -> List[Any]:
        """Load trained models from disk"""
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

        if self.config.calibration_type == 'range_based':
            return self._apply_range_calibration(X_val, y_val, X_test)
        else:
            return self._apply_standard_calibration(X_val, y_val, X_test)

    def _get_uncalibrated_predictions(self, X_test: pd.DataFrame) -> List[np.ndarray]:
        """Get uncalibrated predictions from models"""
        y_pred_proba_list = []
        for model in self.models:
            predictions = []
            for i in range(len(X_test)):
                pred = model.predict_proba(X_test.iloc[[i]])
                predictions.append(pred[0])
            y_pred_proba_list.append(np.array(predictions))
        return y_pred_proba_list

    def _apply_range_calibration(self, X_val: pd.DataFrame, y_val: pd.Series, X_test: pd.DataFrame) -> List[np.ndarray]:
        """Apply range-based calibration"""
        y_pred_proba_list = []

        for model in self.models:
            val_probs = model.predict_proba(X_val)[:, 1]
            calibrator = RangeBasedCalibrator(ranges=self.config.range_calibration_ranges, method='isotonic')
            calibrator.fit(val_probs, y_val)

            test_probs_raw = model.predict_proba(X_test)
            calibrated_probs_class1 = calibrator.transform(test_probs_raw[:, 1])

            calibrated_probs = np.zeros_like(test_probs_raw)
            calibrated_probs[:, 1] = calibrated_probs_class1
            calibrated_probs[:, 0] = 1 - calibrated_probs_class1

            y_pred_proba_list.append(calibrated_probs)

        return y_pred_proba_list

    def _apply_standard_calibration(self, X_val: pd.DataFrame, y_val: pd.Series, X_test: pd.DataFrame) -> List[
        np.ndarray]:
        """Apply standard isotonic or sigmoid calibration"""
        calibrated_models = []
        for model in self.models:
            calibrated_model = CalibratedClassifierCV(model, cv='prefit', method=self.config.calibration_type)
            calibrated_model.fit(X_val, y_val)
            calibrated_models.append(calibrated_model)

        y_pred_proba_list = []
        for model in calibrated_models:
            predictions = []
            for i in range(len(X_test)):
                pred = model.predict_proba(X_test.iloc[[i]])
                predictions.append(pred[0])
            y_pred_proba_list.append(np.array(predictions))

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


if __name__ == "__main__":
    # Run with default configuration
    main()

    # Or customize configuration:
    # custom_config = Config(
    #     manual_threshold=0.55,
    #     use_calibration=True,
    #     calibration_type='range_based',
    #     initial_bankroll=5000,
    #     kelly_fraction=0.25,
    #     fixed_bet_fraction=0.05
    # )
    # main(custom_config)