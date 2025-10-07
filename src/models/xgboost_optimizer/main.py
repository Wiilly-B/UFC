"""UFC Fight Prediction - XGBoost training with walk-forward nested cross-validation and pause/resume."""

from __future__ import annotations

import json
import warnings
from datetime import datetime
from pathlib import Path
import random
import threading
import time

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, roc_auc_score

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = PROJECT_ROOT / "data" / "train_test"
SAVE_DIR = PROJECT_ROOT / "saved_models" / "xgboost" / "trials"
FINAL_MODEL_DIR = PROJECT_ROOT / "saved_models" / "xgboost" / "new_features_15y_100"
TRIAL_PLOTS_DIR = SAVE_DIR / "trial_plots"

ACC_THRESHOLD = 0.50
VAL_ACC_SAVE_THRESHOLD = 0.50
LOSS_GAP_THRESHOLD = 1

INCLUDE_ODDS_COLUMNS = False

SAVE_DIR.mkdir(parents=True, exist_ok=True)
FINAL_MODEL_DIR.mkdir(parents=True, exist_ok=True)
TRIAL_PLOTS_DIR.mkdir(parents=True, exist_ok=True)


# ==================== PAUSE/RESUME CONTROL ====================

class TrainingController:
    """Controls pause/resume functionality for training."""

    def __init__(self):
        self.paused = False
        self.should_stop = False
        self.pause_lock = threading.Lock()
        self.listener_thread = None
        self.running = False

    def start_listener(self):
        """Start control listener in background thread."""
        if self.listener_thread is not None and self.listener_thread.is_alive():
            return

        print("\n" + "="*70)
        print("  TRAINING CONTROLS ACTIVE")
        print("  Type 'p' and press ENTER to PAUSE")
        print("  Type 'r' and press ENTER to RESUME")
        print("  Type 'q' and press ENTER to QUIT after current operation")
        print("="*70 + "\n")

        self.running = True
        self.listener_thread = threading.Thread(target=self._keyboard_listener, daemon=True)
        self.listener_thread.start()

    def _keyboard_listener(self):
        """Listen for keyboard input."""
        while self.running:
            try:
                command = input().strip().lower()

                if command == 'p':
                    self._handle_pause()
                elif command == 'r':
                    self._handle_resume()
                elif command == 'q':
                    self._handle_quit()
                elif command:
                    print(f"[Unknown: '{command}'] Valid commands: p, r, q")

            except (EOFError, KeyboardInterrupt):
                break
            except Exception:
                pass

    def _handle_pause(self):
        """Handle pause signal."""
        with self.pause_lock:
            if not self.paused:
                self.paused = True
                print("\n" + "="*70)
                print("  ⏸️  TRAINING PAUSED")
                print("  Type 'r' and press ENTER to RESUME")
                print("  Type 'q' and press ENTER to QUIT")
                print("="*70 + "\n")

    def _handle_resume(self):
        """Handle resume signal."""
        with self.pause_lock:
            if self.paused:
                self.paused = False
                print("\n" + "="*70)
                print("  ▶️  TRAINING RESUMED")
                print("="*70 + "\n")
            else:
                print("[Info] Training is not paused")

    def _handle_quit(self):
        """Handle quit signal."""
        with self.pause_lock:
            if not self.should_stop:
                self.should_stop = True
                print("\n" + "="*70)
                print("  🛑 QUIT REQUESTED - Will stop after current operation")
                print("="*70 + "\n")

    def check_pause(self):
        """Check if training should pause. Blocks until resumed."""
        while True:
            with self.pause_lock:
                if self.should_stop:
                    raise KeyboardInterrupt("Training stopped by user")
                if not self.paused:
                    break
            time.sleep(0.5)

    def stop(self):
        """Stop the listener thread."""
        self.running = False
        if self.listener_thread is not None:
            self.listener_thread.join(timeout=1.0)


# Global training controller
training_controller = TrainingController()


# ==================== ORIGINAL FUNCTIONS WITH PAUSE POINTS ====================

def load_data_for_cv(train_path: str | Path | None = None,
                     val_path: str | Path | None = None,
                     include_odds: bool = True,
                     date_column: str = 'current_fight_date'):
    """Combine train and validation data for walk-forward nested cross-validation."""
    from pandas.api.types import CategoricalDtype

    train_path = Path(train_path) if train_path is not None else DATA_DIR / "train_data.csv"
    val_path = Path(val_path) if val_path is not None else DATA_DIR / "val_data.csv"

    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    df = pd.concat([train_df, val_df], ignore_index=True)

    if date_column not in df.columns:
        raise ValueError(f"Date column '{date_column}' not found in data. Available columns: {df.columns.tolist()}")

    df[date_column] = pd.to_datetime(df[date_column])
    df = df.sort_values(date_column).reset_index(drop=True)

    drop_cols = ["winner", "fighter_a", "fighter_b", "date"]

    if not include_odds:
        odds_cols = [c for c in df.columns if "odd" in c.lower()]
        drop_cols.extend(odds_cols)
        print(f"Dropping {len(odds_cols)} odds columns: {odds_cols}")

    feature_cols = [c for c in df.columns if c not in drop_cols and c != date_column]

    X = df[feature_cols].copy()
    y = df["winner"].copy()
    dates = df[date_column].copy()

    obj_cols = list(X.select_dtypes(include="object").columns)
    for column in obj_cols:
        values = X[column].astype("string").fillna("<NA>")
        categories = sorted(values.unique())
        cat_dtype = CategoricalDtype(categories=categories, ordered=False)
        X[column] = values.astype(cat_dtype)

    print(f"Combined data for walk-forward nested CV: {X.shape}")
    print(f"Date range: {dates.min()} to {dates.max()}")
    if obj_cols:
        print(f"Categorical cols: {obj_cols}")

    X, y = clean_data_for_xgboost(X, y)

    return X, y, dates


def clean_data_for_xgboost(X: pd.DataFrame, y: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
    """Clean data to remove inf, NaN, and extremely large values."""
    print("\n" + "=" * 70)
    print("DATA CLEANING FOR XGBOOST")
    print("=" * 70)

    initial_shape = X.shape

    inf_mask = np.isinf(X.select_dtypes(include=[np.number]).values).any(axis=1)
    n_inf = inf_mask.sum()
    if n_inf > 0:
        print(f"  Found {n_inf} rows with inf values")
        X = X.replace([np.inf, -np.inf], np.nan)

    nan_counts = X.isna().sum()
    cols_with_nan = nan_counts[nan_counts > 0]
    if len(cols_with_nan) > 0:
        print(f"  Found NaN values in {len(cols_with_nan)} columns")
        print(f"  Total NaN values: {nan_counts.sum()}")

        numeric_cols = X.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if X[col].isna().any():
                median_val = X[col].median()
                if pd.isna(median_val):
                    median_val = 0
                X[col].fillna(median_val, inplace=True)

        cat_cols = X.select_dtypes(include=['category']).columns
        for col in cat_cols:
            if X[col].isna().any():
                X[col].fillna('<NA>', inplace=True)

    numeric_cols = X.select_dtypes(include=[np.number]).columns
    max_val = np.finfo(np.float32).max / 10
    min_val = np.finfo(np.float32).min / 10

    clipped_cols = []
    for col in numeric_cols:
        col_max = X[col].max()
        col_min = X[col].min()
        if col_max > max_val or col_min < min_val:
            clipped_cols.append(col)
            X[col] = X[col].clip(lower=min_val, upper=max_val)

    if clipped_cols:
        print(f"  Clipped {len(clipped_cols)} columns with extreme values")

    assert not X.isna().any().any(), "NaN values still present after cleaning"
    assert not np.isinf(X.select_dtypes(include=[np.number]).values).any(), "Inf values still present after cleaning"

    print(f"  Data shape: {initial_shape} -> {X.shape}")
    print(f"  Data cleaned and ready for XGBoost")
    print("=" * 70)

    return X, y


def get_walk_forward_splits(n_samples: int, n_splits: int = 5, min_train_ratio: float = 0.5):
    """Generate walk-forward (expanding window) splits."""
    min_train_size = int(n_samples * min_train_ratio)
    remaining_samples = n_samples - min_train_size
    test_size = remaining_samples // n_splits

    splits = []
    for i in range(n_splits):
        train_end = min_train_size + (i * test_size)
        test_start = train_end
        test_end = min(test_start + test_size, n_samples)

        if test_end <= test_start:
            continue

        train_idx = np.arange(0, train_end)
        test_idx = np.arange(test_start, test_end)

        splits.append((train_idx, test_idx))

    return splits


def plot_trial_metrics(train_logloss_curves, val_logloss_curves,
                       train_error_curves, val_error_curves,
                       plot_path: Path, trial_number: int, outer_fold: int) -> None:
    """Plot aggregated loss and accuracy curves."""
    if not train_logloss_curves or not val_logloss_curves:
        return

    max_len = max(len(curve) for curve in train_logloss_curves + val_logloss_curves)

    def _mean_curve(curves, transform=None):
        arr = np.full((len(curves), max_len), np.nan, dtype=float)
        for idx, curve in enumerate(curves):
            values = np.asarray(curve, dtype=float)
            if transform is not None:
                values = transform(values)
            arr[idx, :len(values)] = values
        return np.nanmean(arr, axis=0)

    mean_train_loss = _mean_curve(train_logloss_curves)
    mean_val_loss = _mean_curve(val_logloss_curves)
    mean_train_acc = _mean_curve(train_error_curves, transform=lambda x: 1.0 - x)
    mean_val_acc = _mean_curve(val_error_curves, transform=lambda x: 1.0 - x)

    iterations = np.arange(1, max_len + 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"Optuna Trial {trial_number} - Outer Fold {outer_fold}")

    for idx, curve in enumerate(train_logloss_curves):
        rounds = np.arange(1, len(curve) + 1)
        axes[0].plot(rounds, curve, color="tab:blue", alpha=0.3,
                     label="Train Logloss" if idx == 0 else None)
    for idx, curve in enumerate(val_logloss_curves):
        rounds = np.arange(1, len(curve) + 1)
        axes[0].plot(rounds, curve, color="tab:orange", alpha=0.5,
                     label="Validation Logloss" if idx == 0 else None)
    axes[0].plot(iterations, mean_train_loss, color="tab:blue", linewidth=2,
                 label="Train Logloss (mean)")
    axes[0].plot(iterations, mean_val_loss, color="tab:orange", linewidth=2,
                 label="Validation Logloss (mean)")
    axes[0].set_xlabel("Boosting Rounds")
    axes[0].set_ylabel("Log Loss")
    axes[0].set_title("Loss Curves")
    axes[0].legend()

    for idx, curve in enumerate(train_error_curves):
        rounds = np.arange(1, len(curve) + 1)
        axes[1].plot(rounds, 1.0 - np.asarray(curve), color="tab:green", alpha=0.3,
                     label="Train Accuracy" if idx == 0 else None)
    for idx, curve in enumerate(val_error_curves):
        rounds = np.arange(1, len(curve) + 1)
        axes[1].plot(rounds, 1.0 - np.asarray(curve), color="tab:red", alpha=0.5,
                     label="Validation Accuracy" if idx == 0 else None)
    axes[1].plot(iterations, mean_train_acc, color="tab:green", linewidth=2,
                 label="Train Accuracy (mean)")
    axes[1].plot(iterations, mean_val_acc, color="tab:red", linewidth=2,
                 label="Validation Accuracy (mean)")
    axes[1].set_xlabel("Boosting Rounds")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Accuracy Curves")
    axes[1].legend()

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)


def aggregate_best_params(best_params_per_fold):
    """Aggregate hyperparameters from all folds."""
    aggregated = {}

    param_names = set()
    for params in best_params_per_fold:
        param_names.update(params.keys())

    for param_name in param_names:
        values = [params[param_name] for params in best_params_per_fold]

        if isinstance(values[0], (int, float)):
            aggregated[param_name] = type(values[0])(np.median(values))
        else:
            from collections import Counter
            aggregated[param_name] = Counter(values).most_common(1)[0][0]

    return aggregated


def walk_forward_nested_cv(X, y, dates, outer_cv: int = 5, inner_cv: int = 3,
                           optuna_trials: int = 100, save_models: bool = True,
                           run_number: int = 1, include_odds: bool = True):
    """Run walk-forward nested cross-validation with pause/resume."""
    print("\n" + "=" * 70)
    print(f"RUN {run_number} | Walk-Forward Nested CV")
    print(f"Outer folds: {outer_cv} | Inner folds: {inner_cv} | Optuna trials: {optuna_trials}")
    print("=" * 70)

    outer_splits = get_walk_forward_splits(len(X), n_splits=outer_cv, min_train_ratio=0.5)

    outer_accs, outer_aucs = [], []
    train_accs, train_aucs = [], []
    best_params_per_fold = []
    best_n_per_fold = []

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    for fold_idx, (tr_idx, te_idx) in enumerate(outer_splits, start=1):
        training_controller.check_pause()

        print(f"\n{'='*70}")
        print(f"  RUN {run_number} | OUTER FOLD {fold_idx}/{len(outer_splits)}")
        print(f"{'='*70}")

        X_tr, X_te = X.iloc[tr_idx], X.iloc[te_idx]
        y_tr, y_te = y.iloc[tr_idx], y.iloc[te_idx]
        dates_tr, dates_te = dates.iloc[tr_idx], dates.iloc[te_idx]

        print(f"  Training period:   {dates_tr.min().date()} to {dates_tr.max().date()} ({len(X_tr)} samples)")
        print(f"  Test period:       {dates_te.min().date()} to {dates_te.max().date()} ({len(X_te)} samples)")

        inner_splits = get_walk_forward_splits(len(X_tr), n_splits=inner_cv, min_train_ratio=0.6)

        fold_plot_dir = TRIAL_PLOTS_DIR / f"run{run_number}_outer_fold_{fold_idx:02d}"
        fold_plot_dir.mkdir(parents=True, exist_ok=True)

        trial_count = [0]

        def inner_objective(trial: optuna.Trial) -> float:
            """Objective function with pause check."""
            trial_count[0] += 1
            if trial_count[0] % 5 == 0:
                training_controller.check_pause()

            params = {
                "objective": "binary:logistic",
                "tree_method": "hist",
                "device": "cuda",
                "enable_categorical": True,
                'n_estimators': trial.suggest_int('n_estimators', 100, 2000),
                "eval_metric": ["logloss", "error"],
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "gamma": trial.suggest_float("gamma", 0.01, 1.0, log=True),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
                "early_stopping_rounds": 50,
            }

            fold_aucs = []
            train_logloss_curves, val_logloss_curves = [], []
            train_error_curves, val_error_curves = [], []

            for in_tr_idx, in_va_idx in inner_splits:
                X_in_tr, X_in_va = X_tr.iloc[in_tr_idx], X_tr.iloc[in_va_idx]
                y_in_tr, y_in_va = y_tr.iloc[in_tr_idx], y_tr.iloc[in_va_idx]

                model = xgb.XGBClassifier(**params)
                model.fit(
                    X_in_tr,
                    y_in_tr,
                    eval_set=[(X_in_tr, y_in_tr), (X_in_va, y_in_va)],
                    verbose=False,
                )

                proba = model.predict_proba(X_in_va)[:, 1]
                fold_aucs.append(roc_auc_score(y_in_va, proba))

                evals_result = model.evals_result()
                train_logloss_curves.append(evals_result["validation_0"]["logloss"])
                val_logloss_curves.append(evals_result["validation_1"]["logloss"])
                train_error_curves.append(evals_result["validation_0"]["error"])
                val_error_curves.append(evals_result["validation_1"]["error"])

            plot_path = fold_plot_dir / f"trial_{trial.number:03d}_metrics.png"
            plot_trial_metrics(
                train_logloss_curves,
                val_logloss_curves,
                train_error_curves,
                val_error_curves,
                plot_path,
                trial.number,
                fold_idx,
            )

            return float(np.mean(fold_aucs))

        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=random.randint(0, 100000))
        )
        study.optimize(inner_objective, n_trials=optuna_trials)

        best_params = dict(study.best_params)
        best_params_per_fold.append(best_params)
        print(f"\n  Inner CV Best AUC: {study.best_value:.4f}")

        training_controller.check_pause()

        val_split_point = int(len(X_tr) * 0.85)
        X_tr2, X_va2 = X_tr.iloc[:val_split_point], X_tr.iloc[val_split_point:]
        y_tr2, y_va2 = y_tr.iloc[:val_split_point], y_tr.iloc[val_split_point:]

        final_params = {
            "objective": "binary:logistic",
            "tree_method": "hist",
            "device": "cpu",
            "enable_categorical": True,
            "eval_metric": ["logloss", "error"],
            "early_stopping_rounds": 50,
            **best_params,
        }

        final_model = xgb.XGBClassifier(**final_params)
        final_model.fit(
            X_tr2,
            y_tr2,
            eval_set=[(X_tr2, y_tr2), (X_va2, y_va2)],
            verbose=False,
        )

        evals_result = final_model.evals_result()
        train_loss_curve = evals_result["validation_0"]["logloss"]
        val_loss_curve = evals_result["validation_1"]["logloss"]
        train_error_curve = evals_result["validation_0"]["error"]
        val_error_curve = evals_result["validation_1"]["error"]

        best_iteration = getattr(final_model, "best_iteration", None)
        metric_index = int(best_iteration) if best_iteration is not None else len(val_loss_curve) - 1
        train_loss_at_best = float(train_loss_curve[metric_index])
        val_loss_at_best = float(val_loss_curve[metric_index])
        train_acc_at_best = 1.0 - float(train_error_curve[metric_index])
        val_acc_at_best = 1.0 - float(val_error_curve[metric_index])
        loss_gap = train_loss_at_best - val_loss_at_best
        loss_gap_abs = abs(loss_gap)

        best_n_estimators = (
            int(best_iteration) + 1
            if best_iteration is not None
            else final_model.get_params().get("n_estimators", 800)
        )
        best_n_per_fold.append(int(best_n_estimators))

        final_model = xgb.XGBClassifier(
            **{**final_params, "n_estimators": int(best_n_estimators), "early_stopping_rounds": None}
        )
        final_model.fit(X_tr, y_tr, verbose=False)

        te_proba = final_model.predict_proba(X_te)[:, 1]
        te_acc = accuracy_score(y_te, (te_proba > 0.5).astype(int))
        te_auc = roc_auc_score(y_te, te_proba)

        tr_proba = final_model.predict_proba(X_tr)[:, 1]
        tr_acc = accuracy_score(y_tr, (tr_proba > 0.5).astype(int))
        tr_auc = roc_auc_score(y_tr, tr_proba)

        outer_accs.append(te_acc)
        outer_aucs.append(te_auc)
        train_accs.append(tr_acc)
        train_aucs.append(tr_auc)

        print(f"\n  FOLD {fold_idx} | Test ACC: {te_acc:.4f} | Test AUC: {te_auc:.4f}")

        if (
            save_models
            and val_acc_at_best >= ACC_THRESHOLD
            and val_acc_at_best >= VAL_ACC_SAVE_THRESHOLD
            and loss_gap_abs <= LOSS_GAP_THRESHOLD
        ):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_path = SAVE_DIR / f"run{run_number}_walkforward_fold{fold_idx}_acc{te_acc:.3f}_{timestamp}.json"
            final_model.save_model(str(model_path))

            metadata_path = SAVE_DIR / f"run{run_number}_walkforward_fold{fold_idx}_metadata_{timestamp}.json"
            metadata = {
                "run_number": run_number,
                "fold": fold_idx,
                "validation_method": "walk_forward",
                "include_odds": include_odds,
                "metrics": {
                    "test_acc": float(te_acc),
                    "test_auc": float(te_auc),
                },
            }
            metadata_path.write_text(json.dumps(metadata, indent=2))

    print("\n" + "=" * 70)
    print(f"  RUN {run_number} | RESULTS")
    print(f"  Test Accuracy:  {np.mean(outer_accs):.4f} ± {np.std(outer_accs):.4f}")
    print(f"  Test AUC:       {np.mean(outer_aucs):.4f} ± {np.std(outer_aucs):.4f}")
    print("=" * 70)

    return {
        "outer_accs": outer_accs,
        "outer_aucs": outer_aucs,
        "train_accs": train_accs,
        "train_aucs": train_aucs,
        "best_params_per_fold": best_params_per_fold,
        "best_n_per_fold": best_n_per_fold,
    }


def train_final_model_on_all_data(X, y, dates, aggregated_params, median_n_estimators,
                                  run_number, include_odds):
    """Train final production model."""
    training_controller.check_pause()

    print("\n" + "=" * 70)
    print(f"  RUN {run_number} | TRAINING FINAL MODEL")
    print("=" * 70)

    final_params = {
        "objective": "binary:logistic",
        "tree_method": "hist",
        "device": "cpu",
        "enable_categorical": True,
        "n_estimators": int(median_n_estimators),
        "eval_metric": ["logloss", "error"],
        "early_stopping_rounds": None,
        **aggregated_params,
    }

    final_model = xgb.XGBClassifier(**final_params)
    final_model.fit(X, y, verbose=False)

    train_proba = final_model.predict_proba(X)[:, 1]
    train_acc = accuracy_score(y, (train_proba > 0.5).astype(int))
    train_auc = roc_auc_score(y, train_proba)

    print(f"  Train ACC: {train_acc:.4f} | Train AUC: {train_auc:.4f}")
    print("=" * 70)

    return final_model, final_params, train_acc, train_auc


def train_xgboost_walkforward(optuna_trials: int = 100, outer_cv: int = 5,
                              inner_cv: int = 3, save_models: bool = True,
                              run_number: int = 1, include_odds: bool = True) -> dict:
    """Entry point for walk-forward training."""
    print("=" * 70)
    print(f"  RUN {run_number} | XGBoost Walk-Forward Training")
    print("=" * 70)

    X, y, dates = load_data_for_cv(include_odds=include_odds)

    results = walk_forward_nested_cv(
        X,
        y,
        dates,
        outer_cv=outer_cv,
        inner_cv=inner_cv,
        optuna_trials=optuna_trials,
        save_models=save_models,
        run_number=run_number,
        include_odds=include_odds,
    )

    aggregated_params = aggregate_best_params(results["best_params_per_fold"])
    median_n_estimators = int(np.median(results["best_n_per_fold"]))

    final_model, final_params, train_acc, train_auc = train_final_model_on_all_data(
        X, y, dates, aggregated_params, median_n_estimators, run_number, include_odds
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    final_model_path = FINAL_MODEL_DIR / f"run{run_number}_final_model_{timestamp}.json"
    final_model.save_model(str(final_model_path))

    print(f"  ✓ Model saved: {final_model_path.name}")

    return {
        "run_number": run_number,
        "model_path": str(final_model_path),
        "mean_test_acc": float(np.mean(results["outer_accs"])),
        "std_test_acc": float(np.std(results["outer_accs"])),
        "mean_test_auc": float(np.mean(results["outer_aucs"])),
        "std_test_auc": float(np.std(results["outer_aucs"])),
    }


def run_multiple_training_sessions(n_runs: int = 5, optuna_trials: int = 100,
                                   outer_cv: int = 5, inner_cv: int = 3,
                                   save_models: bool = True, include_odds: bool = True):
    """Run multiple training sessions with pause/resume."""
    print("\n" + "█" * 70)
    print(f"█  WALK-FORWARD TRAINING - {n_runs} RUNS")
    print("█" * 70 + "\n")

    training_controller.start_listener()

    all_run_results = []

    try:
        for run_idx in range(1, n_runs + 1):
            training_controller.check_pause()

            print(f"\n{'█' * 70}")
            print(f"█  RUN {run_idx}/{n_runs}")
            print(f"{'█' * 70}\n")

            run_results = train_xgboost_walkforward(
                optuna_trials=optuna_trials,
                outer_cv=outer_cv,
                inner_cv=inner_cv,
                save_models=save_models,
                run_number=run_idx,
                include_odds=include_odds,
            )

            all_run_results.append(run_results)

    except KeyboardInterrupt:
        print("\n" + "=" * 70)
        print("  TRAINING STOPPED BY USER")
        print(f"  Completed {len(all_run_results)}/{n_runs} runs")
        print("=" * 70)
    finally:
        training_controller.stop()

    if not all_run_results:
        print("\nNo runs completed.")
        return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_path = FINAL_MODEL_DIR / f"all_runs_summary_{timestamp}.json"

    summary = {
        "total_runs": n_runs,
        "completed_runs": len(all_run_results),
        "timestamp": timestamp,
        "all_runs": all_run_results,
        "aggregate_statistics": {
            "mean_test_acc": float(np.mean([r["mean_test_acc"] for r in all_run_results])),
            "std_test_acc": float(np.std([r["mean_test_acc"] for r in all_run_results])),
            "mean_test_auc": float(np.mean([r["mean_test_auc"] for r in all_run_results])),
            "std_test_auc": float(np.std([r["mean_test_auc"] for r in all_run_results])),
        },
    }

    summary_path.write_text(json.dumps(summary, indent=2))

    print("\n" + "█" * 70)
    print("█  TRAINING COMPLETE")
    print(f"█  Completed: {len(all_run_results)}/{n_runs} runs")
    print(f"█  Mean Acc: {summary['aggregate_statistics']['mean_test_acc']:.4f}")
    print(f"█  Mean AUC: {summary['aggregate_statistics']['mean_test_auc']:.4f}")
    print("█" * 70 + "\n")


if __name__ == "__main__":
    try:
        run_multiple_training_sessions(
            n_runs=100,
            optuna_trials=20,
            outer_cv=5,
            inner_cv=3,
            save_models=True,
            include_odds=INCLUDE_ODDS_COLUMNS
        )
    except KeyboardInterrupt:
        print("\nTraining interrupted. Exiting...")
    finally:
        training_controller.stop()