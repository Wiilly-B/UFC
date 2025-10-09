"""UFC Fight Prediction - XGBoost (walk-forward nested CV, leakage-safe)

Implements three key fixes:
1) Walk-forward splitter covers the entire tail (no leftover samples).
2) Preprocessing is *fit on outer-train only* and applied to outer-test (no leakage).
   - numeric: median impute (from outer-train), float32-safe clipping
   - categorical: categories learned on outer-train, aligned on outer-test
3) Feature selection is *nested properly*:
   - During inner CV, Top-K selection is re-run on each inner-train split.
   - Before final outer refit/test, Top-K is re-run once on the full outer-train.

Other notes:
- Optuna still minimizes validation logloss with pruning.
- Best iteration chosen on a small holdout from the outer-train, then refit on full outer-train.
"""

from __future__ import annotations

import json
import warnings
from datetime import datetime
from pathlib import Path
import random
import threading
import time
import os
import sys

from optuna.pruners import MedianPruner
from optuna.integration import XGBoostPruningCallback

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss

# ==================== TOGGLES / PATHS ====================

# ---- Plotting / feature preview toggles ----
SHOW_PLOTS = True             # True -> show charts via plt.show(); False -> headless
SHOW_TRIAL_PLOTS = False      # Per-trial mean curves (keep False to avoid spam)
FEATURE_PREVIEW_N = 10        # how many of the selected features to print (preview)
if not SHOW_PLOTS:
    matplotlib.use("Agg")     # headless only when not showing
# --------------------------------------------

# ---- Feature selection toggle ----
USE_TOP_K_FEATURES = True     # True -> select top K features
TOP_K_FEATURES = 120
INCLUDE_ODDS_COLUMNS = False   # when False, drop odds/open/closing-related columns

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = PROJECT_ROOT / "data" / "train_test"
SAVE_DIR = PROJECT_ROOT / "saved_models" / "xgboost" / "trials"
FINAL_MODEL_DIR = PROJECT_ROOT / "saved_models" / "xgboost" / "test"
TRIAL_PLOTS_DIR = SAVE_DIR / "trial_plots"

# ---- Save criteria (logloss-based) ----
VAL_LOGLOSS_SAVE_MAX = 0.69    # ~ random baseline for balanced labels
TEST_LOGLOSS_SAVE_MAX = 0.70

# Alignment cap used by constraints + outer save gate
GAP_MAX = 0.04  # tighten (0.03) or relax (0.05–0.06) as needed

SAVE_DIR.mkdir(parents=True, exist_ok=True)
FINAL_MODEL_DIR.mkdir(parents=True, exist_ok=True)
TRIAL_PLOTS_DIR.mkdir(parents=True, exist_ok=True)


# ==================== PAUSE/RESUME CONTROL ====================

class TrainingController:
    """Controls pause/resume functionality for training with a non-blocking listener."""

    def __init__(self):
        self.paused = False
        self.should_stop = False
        self.pause_lock = threading.Lock()
        self.listener_thread = None
        self.running = False

    def start_listener(self):
        """Start control listener in background thread (non-blocking)."""
        if self.listener_thread is not None and self.listener_thread.is_alive():
            return

        print("\n" + "=" * 70)
        print("  TRAINING CONTROLS ACTIVE")
        print("  Type 'p' and press ENTER to PAUSE")
        print("  Type 'r' and press ENTER to RESUME")
        print("  Type 'q' and press ENTER to QUIT after current operation")
        print("=" * 70 + "\n")

        self.running = True
        self.listener_thread = threading.Thread(target=self._keyboard_listener, daemon=True)
        self.listener_thread.start()

    def _keyboard_listener(self):
        """Non-blocking keyboard listener (Windows + POSIX)."""
        if os.name == "nt":
            try:
                import msvcrt
            except ImportError:
                while self.running:
                    time.sleep(0.2)
                return

            buf = []
            while self.running:
                try:
                    if msvcrt.kbhit():
                        ch = msvcrt.getwch()
                        if ch in ("\r", "\n"):
                            command = "".join(buf).strip().lower()
                            buf.clear()
                            self._dispatch(command)
                        elif ch == "\x03":  # Ctrl+C
                            break
                        else:
                            buf.append(ch)
                    else:
                        time.sleep(0.1)
                except Exception:
                    break
        else:
            import select
            line = []
            while self.running:
                try:
                    rlist, _, _ = select.select([sys.stdin], [], [], 0.2)
                    if rlist:
                        ch = sys.stdin.read(1)
                        if ch in ("\n", "\r"):
                            command = "".join(line).strip().lower()
                            line.clear()
                            self._dispatch(command)
                        else:
                            line.append(ch)
                except Exception:
                    break

    def _dispatch(self, command: str):
        if command == "p":
            self._handle_pause()
        elif command == "r":
            self._handle_resume()
        elif command == "q":
            self._handle_quit()
        elif command:
            print(f"[Unknown: '{command}'] Valid commands: p, r, q")

    def _handle_pause(self):
        with self.pause_lock:
            if not self.paused:
                self.paused = True
                print("\n" + "=" * 70)
                print("  ⏸️  TRAINING PAUSED")
                print("  Type 'r' and press ENTER to RESUME")
                print("  Type 'q' and press ENTER to QUIT")
                print("=" * 70 + "\n")

    def _handle_resume(self):
        with self.pause_lock:
            if self.paused:
                self.paused = False
                print("\n" + "=" * 70)
                print("  ▶️  TRAINING RESUMED")
                print("=" * 70 + "\n")
            else:
                print("[Info] Training is not paused")

    def _handle_quit(self):
        with self.pause_lock:
            if not self.should_stop:
                self.should_stop = True
                print("\n" + "=" * 70)
                print("  🛑 QUIT REQUESTED - Will stop after current operation")
                print("=" * 70 + "\n")

    def check_pause(self):
        """Block while paused; raise if quit requested."""
        while True:
            with self.pause_lock:
                if self.should_stop:
                    raise KeyboardInterrupt("Training stopped by user")
                if not self.paused:
                    break
            time.sleep(0.5)

    def stop(self):
        self.running = False
        if self.listener_thread is not None:
            self.listener_thread.join(timeout=1.0)


training_controller = TrainingController()


# ==================== DATA LOADING (NO PREPROCESSING HERE) ====================

def load_data_for_cv(train_path: str | Path | None = None,
                     val_path: str | Path | None = None,
                     include_odds: bool = True,
                     date_column: str = "current_fight_date"):
    """Load and concatenate train/val. Sort by date. Do *not* impute/clip here (avoid leakage)."""
    train_path = Path(train_path) if train_path is not None else DATA_DIR / "train_data.csv"
    val_path = Path(val_path) if val_path is not None else DATA_DIR / "val_data.csv"

    df_tr = pd.read_csv(train_path)
    df_va = pd.read_csv(val_path)
    df = pd.concat([df_tr, df_va], ignore_index=True)

    if date_column not in df.columns:
        raise ValueError(f"Date column '{date_column}' not found. Available: {df.columns.tolist()}")

    df[date_column] = pd.to_datetime(df[date_column], errors="coerce")
    df = df.sort_values(date_column).reset_index(drop=True)

    drop_cols = ["winner", "fighter_a", "fighter_b", "date"]
    if not include_odds:
        odds_terms = ("odd", "open", "opening", "close", "closing")
        odds_cols = [c for c in df.columns if any(t in c.lower() for t in odds_terms)]
        drop_cols.extend(odds_cols)
        print(f"Dropping {len(odds_cols)} odds-related columns: {odds_cols}")

    feature_cols = [c for c in df.columns if c not in drop_cols and c != date_column]

    X_raw = df[feature_cols].copy()   # raw (may contain NaNs, inf, objects)
    y = df["winner"].copy()
    dates = df[date_column].copy()

    print(f"Loaded: X={X_raw.shape}, y={y.shape}, dates={dates.min()} → {dates.max()}")
    return X_raw, y, dates


# ==================== PER-FOLD PREPROCESSING (NO LEAKAGE) ====================

def fit_transform_fold(X_tr_raw: pd.DataFrame, X_te_raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Fit preprocessing on OUTER-TRAIN ONLY; apply to OUTER-TEST.
    - numeric: median impute (fallback 0) + float32-safe clipping
    - categorical: categories from train; unseen in test -> '<NA>'
    """
    X_tr = X_tr_raw.copy()
    X_te = X_te_raw.copy()

    # Numeric
    num_cols = X_tr.select_dtypes(include=[np.number]).columns.tolist()
    if num_cols:
        med = X_tr[num_cols].median().fillna(0)
        X_tr[num_cols] = X_tr[num_cols].fillna(med)
        X_te[num_cols] = X_te[num_cols].fillna(med)

        max_val = np.finfo(np.float32).max / 10
        min_val = np.finfo(np.float32).min / 10
        X_tr[num_cols] = X_tr[num_cols].clip(lower=min_val, upper=max_val)
        X_te[num_cols] = X_te[num_cols].clip(lower=min_val, upper=max_val)

    # Categorical (object/string/category) → pandas Categorical with TRAIN categories
    obj_cols = X_tr.select_dtypes(include=["object", "string", "category"]).columns.tolist()
    for col in obj_cols:
        tr_vals = X_tr[col].astype("string").fillna("<NA>")
        cats = pd.Index(sorted(pd.unique(pd.concat([tr_vals, pd.Series(["<NA>"])]))))
        X_tr[col] = pd.Categorical(tr_vals, categories=cats)

        te_vals = X_te[col].astype("string").fillna("<NA>")
        te_vals = te_vals.where(te_vals.isin(cats), "<NA>")
        X_te[col] = pd.Categorical(te_vals, categories=cats)

    # Safety assertions (no inf; NaNs should only appear if truly missing categories)
    assert not np.isinf(X_tr.select_dtypes(include=[np.number]).to_numpy()).any(), "Inf in train after preprocess"
    assert not np.isinf(X_te.select_dtypes(include=[np.number]).to_numpy()).any(), "Inf in test after preprocess"
    # Allow NaNs for categoricals (XGB handles categorical missing); numeric should be clean
    assert not X_tr.select_dtypes(include=[np.number]).isna().any().any(), "Numeric NaNs in train"
    assert not X_te.select_dtypes(include=[np.number]).isna().any().any(), "Numeric NaNs in test"

    return X_tr, X_te


# ==================== FEATURE SELECTION (NESTED) ====================

def select_top_features_by_xgb(X_tr: pd.DataFrame, y_tr: pd.Series,
                               k: int = TOP_K_FEATURES,
                               enabled: bool = USE_TOP_K_FEATURES) -> tuple[list[str], list[str]]:
    """
    If enabled:
      - Train a quick XGB on the given TRAIN split only and select top-k by gain.
    Else:
      - Return all columns unchanged.

    Returns (selected_cols, full_ranking_or_all_cols).
    """
    if not enabled:
        cols = list(X_tr.columns)
        print(f"[FS] DISABLED → using ALL {len(cols)} features.")
        if FEATURE_PREVIEW_N > 0:
            preview = cols[:min(FEATURE_PREVIEW_N, len(cols))]
            print("[FS] Preview:", ", ".join(preview))
        return cols, cols

    print(f"[FS] ENABLED → selecting top {k} features on this TRAIN split...")
    params = {
        "objective": "binary:logistic",
        "tree_method": "hist",
        "device": "cuda",
        "enable_categorical": True,
        "n_estimators": 200,
        "max_depth": 6,
        "learning_rate": 0.1,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "eval_metric": "logloss",
        "verbosity": 0,
    }
    fs_model = xgb.XGBClassifier(**params)
    fs_model.fit(X_tr, y_tr, verbose=False)

    booster = fs_model.get_booster()
    score = booster.get_score(importance_type="gain")  # dict: name -> gain

    # Map f{i} -> names if needed
    if score and all(k.startswith("f") for k in score.keys()):
        feat_names = booster.feature_names
        mapped = {}
        for key, val in score.items():
            idx = int(key[1:])
            if idx < len(feat_names):
                mapped[feat_names[idx]] = val
        score = mapped

    if not score:
        print("[FS] Warning: no importance scores. Falling back to first K.")
        full_rank = list(X_tr.columns)
        selected_cols = full_rank[:k]
    else:
        full_rank = [name for name, _ in sorted(score.items(), key=lambda kv: kv[1], reverse=True)]
        top_set = set(full_rank[:k])
        selected_cols = [c for c in X_tr.columns if c in top_set]

    print(f"[FS] Selected {len(selected_cols)} features.")
    if FEATURE_PREVIEW_N > 0 and score:
        preview = full_rank[:min(FEATURE_PREVIEW_N, len(full_rank))]
        print("[FS] Top (by gain):", ", ".join(preview))
    return selected_cols, full_rank


# ==================== CV SPLITS & PLOTTING ====================

def get_walk_forward_splits(n_samples: int, n_splits: int = 5, min_train_ratio: float = 0.5):
    """
    Expanding-window walk-forward splits that COVER THE TAIL.
    - Training indices: [0, train_end)
    - Test indices:     [train_end, next_edge)
    Ensures the last fold ends exactly at n_samples.
    """
    min_train_size = int(n_samples * min_train_ratio)
    if min_train_size < 1:
        raise ValueError("min_train_ratio too small for the dataset size.")

    # Evenly partition the remaining segment into n_splits contiguous chunks
    edges = np.linspace(min_train_size, n_samples, n_splits + 1, dtype=int)
    splits = []
    for i in range(n_splits):
        train_end = edges[i]
        test_start = edges[i]
        test_end = edges[i + 1]
        if test_end > test_start:
            train_idx = np.arange(0, train_end)
            test_idx = np.arange(test_start, test_end)
            splits.append((train_idx, test_idx))
    return splits


def plot_trial_metrics(train_logloss_curves, val_logloss_curves,
                       train_error_curves, val_error_curves,
                       trial_number: int, outer_fold: int) -> None:
    if not SHOW_PLOTS or not SHOW_TRIAL_PLOTS:
        return
    if not train_logloss_curves or not val_logloss_curves:
        return

    max_len = max(len(c) for c in train_logloss_curves + val_logloss_curves)

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

    axes[0].plot(iterations, mean_train_loss, linewidth=2, label="Train Logloss (mean)")
    axes[0].plot(iterations, mean_val_loss, linewidth=2, label="Validation Logloss (mean)")
    axes[0].set_xlabel("Boosting Rounds")
    axes[0].set_ylabel("Log Loss")
    axes[0].set_title("Loss Curves")
    axes[0].legend()

    axes[1].plot(iterations, mean_train_acc, linewidth=2, label="Train Accuracy (mean)")
    axes[1].plot(iterations, mean_val_acc, linewidth=2, label="Validation Accuracy (mean)")
    axes[1].set_xlabel("Boosting Rounds")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Accuracy Curves")
    axes[1].legend()

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show(block=False)
    plt.pause(0.001)


def plot_final_fold_curves(evals_result: dict, outer_fold: int):
    if not SHOW_PLOTS or not evals_result:
        return
    tr_ll = evals_result.get("validation_0", {}).get("logloss", None)
    va_ll = evals_result.get("validation_1", {}).get("logloss", None)
    tr_er = evals_result.get("validation_0", {}).get("error", None)
    va_er = evals_result.get("validation_1", {}).get("error", None)
    if tr_ll is None or va_ll is None or tr_er is None or va_er is None:
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"Final Model Curves - Outer Fold {outer_fold}")

    axes[0].plot(np.arange(1, len(tr_ll) + 1), tr_ll, label="Train Logloss")
    axes[0].plot(np.arange(1, len(va_ll) + 1), va_ll, label="Val Logloss")
    axes[0].set_xlabel("Boosting Rounds")
    axes[0].set_ylabel("Log Loss")
    axes[0].legend()
    axes[0].set_title("Loss")

    axes[1].plot(np.arange(1, len(tr_er) + 1), 1.0 - np.asarray(tr_er), label="Train Acc")
    axes[1].plot(np.arange(1, len(va_er) + 1), 1.0 - np.asarray(va_er), label="Val Acc")
    axes[1].set_xlabel("Boosting Rounds")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    axes[1].set_title("Accuracy")

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show(block=False)
    plt.pause(0.001)


def aggregate_best_params(best_params_per_fold):
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


# ==================== OPTUNA CONSTRAINTS (OPTIONAL) ====================

def _constraints(trial: optuna.Trial):
    """
    Constrained optimization hook (optional).
    We keep mean GAP ≤ GAP_MAX via user_attrs if you later compute & store it.
    For now we pass through; kept for compatibility.
    """
    mean_gap = trial.user_attrs.get("mean_loss_gap", 0.0)
    return (max(0.0, mean_gap - GAP_MAX),)


# ==================== TRAINING LOOPS ====================

def walk_forward_nested_cv(X_raw, y, dates, outer_cv: int = 5, inner_cv: int = 3,
                           optuna_trials: int = 100, save_models: bool = True,
                           run_number: int = 1, include_odds: bool = True):
    """Run walk-forward nested CV with leakage-safe preprocessing and nested FS."""
    print("\n" + "=" * 70)
    print(f"RUN {run_number} | Walk-Forward Nested CV (logloss-min; nested FS; leakage-safe)")
    print(f"Outer folds: {outer_cv} | Inner folds: {inner_cv} | Optuna trials: {optuna_trials}")
    print("=" * 70)

    outer_splits = get_walk_forward_splits(len(X_raw), n_splits=outer_cv, min_train_ratio=0.5)

    outer_test_logloss, outer_test_auc = [], []
    outer_train_logloss, outer_train_auc = [], []
    best_params_per_fold = []
    best_n_per_fold = []

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    for fold_idx, (tr_idx, te_idx) in enumerate(outer_splits, start=1):
        training_controller.check_pause()

        print(f"\n{'=' * 70}")
        print(f"  RUN {run_number} | OUTER FOLD {fold_idx}/{len(outer_splits)}")
        print(f"{'=' * 70}")

        X_tr_raw, X_te_raw = X_raw.iloc[tr_idx], X_raw.iloc[te_idx]
        y_tr, y_te = y.iloc[tr_idx], y.iloc[te_idx]
        dates_tr, dates_te = dates.iloc[tr_idx], dates.iloc[te_idx]

        print(f"  Train: {dates_tr.min().date()} → {dates_tr.max().date()} ({len(X_tr_raw)})")
        print(f"  Test : {dates_te.min().date()} → {dates_te.max().date()} ({len(X_te_raw)})")

        # -------- FIX #2: leakage-safe preprocessing (fit on outer-train; apply to outer-test)
        X_tr_proc, X_te_proc = fit_transform_fold(X_tr_raw, X_te_raw)

        # -------- INNER CV (with nested FS *per inner split*)
        inner_splits = get_walk_forward_splits(len(X_tr_proc), n_splits=inner_cv, min_train_ratio=0.6)
        trial_count = [0]

        def inner_objective(trial: optuna.Trial) -> float:
            trial_count[0] += 1
            if trial_count[0] % 5 == 0:
                training_controller.check_pause()

            # Conservative, stable hyperparameters; optimized for logloss
            params = {
                "objective": "binary:logistic",
                "tree_method": "hist",
                "device": "cuda",
                "enable_categorical": True,
                "n_estimators": trial.suggest_int("n_estimators", 100, 2000),
                "eval_metric": ["logloss", "error"],
                "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.03, log=True),
                "max_depth": trial.suggest_int("max_depth", 3, 4),
                "min_child_weight": trial.suggest_int("min_child_weight", 20, 120, step=10),
                "subsample": trial.suggest_float("subsample", 0.55, 0.85),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.55, 0.85),
                "gamma": trial.suggest_float("gamma", 0.001, 10.0, log=True),
                "reg_alpha": trial.suggest_float("reg_alpha", 50.0, 150.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 50.0, 150.0, log=True),
                "max_delta_step": trial.suggest_int("max_delta_step", 3, 10),
                "early_stopping_rounds": 50,
                "sampling_method": "gradient_based",
                "grow_policy": trial.suggest_categorical("grow_policy", ["depthwise", "lossguide"]),
            }

            fold_val_logloss = []
            train_logloss_curves, val_logloss_curves = [], []
            train_error_curves, val_error_curves = [], []

            for in_tr_idx, in_va_idx in inner_splits:
                X_in_tr = X_tr_proc.iloc[in_tr_idx]
                X_in_va = X_tr_proc.iloc[in_va_idx]
                y_in_tr = y_tr.iloc[in_tr_idx]
                y_in_va = y_tr.iloc[in_va_idx]

                # -------- FIX #3: nested FS per inner TRAIN split
                sel_cols, _ = select_top_features_by_xgb(
                    X_in_tr, y_in_tr, k=TOP_K_FEATURES, enabled=USE_TOP_K_FEATURES
                )
                X_in_tr_sel = X_in_tr[sel_cols]
                X_in_va_sel = X_in_va[sel_cols]

                model = xgb.XGBClassifier(**params)
                prune_cb = XGBoostPruningCallback(trial, "validation_1-logloss")
                model.fit(
                    X_in_tr_sel, y_in_tr,
                    eval_set=[(X_in_tr_sel, y_in_tr), (X_in_va_sel, y_in_va)],
                    verbose=False,
                    callbacks=[prune_cb],
                )

                evals = model.evals_result()
                tr_curve = np.asarray(evals["validation_0"]["logloss"], dtype=float)
                va_curve = np.asarray(evals["validation_1"]["logloss"], dtype=float)

                best_idx = int(np.argmin(va_curve))
                fold_val_logloss.append(float(va_curve[best_idx]))

                # collect for optional plotting
                train_logloss_curves.append(tr_curve.tolist())
                val_logloss_curves.append(va_curve.tolist())
                train_error_curves.append(evals["validation_0"]["error"])
                val_error_curves.append(evals["validation_1"]["error"])

            plot_trial_metrics(
                train_logloss_curves, val_logloss_curves,
                train_error_curves, val_error_curves,
                trial.number, fold_idx,
            )

            mean_val_ll = float(np.mean(fold_val_logloss))
            # Store a nominal "gap" attribute if you compute one (kept for constraints API symmetry)
            trial.set_user_attr("mean_loss_gap", 0.0)
            return mean_val_ll

        study = optuna.create_study(
            direction="minimize",
            sampler=optuna.samplers.TPESampler(
                seed=random.randint(0, 100000),
                constraints_func=_constraints,
            ),
            pruner=MedianPruner(n_warmup_steps=10),
        )
        study.optimize(inner_objective, n_trials=optuna_trials)

        best_params = dict(study.best_params)
        best_params_per_fold.append(best_params)
        print(f"\n  Inner CV Best Val Logloss: {study.best_value:.4f}")

        training_controller.check_pause()

        # -------- Finalize feature set for the OUTER fold (FS on full outer-train)
        sel_cols_outer, _ = select_top_features_by_xgb(
            X_tr_proc, y_tr, k=TOP_K_FEATURES, enabled=USE_TOP_K_FEATURES
        )
        X_tr_sel = X_tr_proc[sel_cols_outer]
        X_te_sel = X_te_proc[sel_cols_outer]

        # Holdout from outer-train to lock best_iteration, then refit on full outer-train
        val_split_point = int(len(X_tr_sel) * 0.85)
        X_tr2, X_va2 = X_tr_sel.iloc[:val_split_point], X_tr_sel.iloc[val_split_point:]
        y_tr2, y_va2 = y_tr.iloc[:val_split_point], y_tr.iloc[val_split_point:]

        final_params = {
            "objective": "binary:logistic",
            "tree_method": "hist",
            "device": "cuda",
            "enable_categorical": True,
            "eval_metric": ["logloss", "error"],
            "early_stopping_rounds": 100,
            **best_params,
        }

        # Stage 1: determine best_iteration using the small holdout
        final_model = xgb.XGBClassifier(**final_params)
        final_model.fit(
            X_tr2, y_tr2,
            eval_set=[(X_tr2, y_tr2), (X_va2, y_va2)],
            verbose=False,
        )

        evals_result = final_model.evals_result()
        plot_final_fold_curves(evals_result, fold_idx)

        val_loss_curve = evals_result["validation_1"]["logloss"]
        best_iteration = getattr(final_model, "best_iteration", None)
        metric_index = int(best_iteration) if best_iteration is not None else int(np.argmin(val_loss_curve))
        best_n_estimators = metric_index + 1
        best_n_per_fold.append(int(best_n_estimators))

        # Stage 2: refit on ALL outer-train with fixed n_estimators
        final_params_refit = {
            **final_params,
            "n_estimators": int(best_n_estimators),
            "early_stopping_rounds": None,
        }
        refit_model = xgb.XGBClassifier(**final_params_refit)
        refit_model.fit(X_tr_sel, y_tr, verbose=False)

        # Evaluate on outer-test
        te_proba = refit_model.predict_proba(X_te_sel)[:, 1]
        te_logloss = log_loss(y_te, te_proba)
        te_auc = roc_auc_score(y_te, te_proba)

        # Also record outer-train metrics (for diagnostics)
        tr_proba = refit_model.predict_proba(X_tr_sel)[:, 1]
        tr_logloss = log_loss(y_tr, tr_proba)
        tr_auc = roc_auc_score(y_tr, tr_proba)

        outer_test_logloss.append(te_logloss)
        outer_test_auc.append(te_auc)
        outer_train_logloss.append(tr_logloss)
        outer_train_auc.append(tr_auc)

        print(f"\n  FOLD {fold_idx} | Test LOGLOSS: {te_logloss:.4f} | Test AUC: {te_auc:.4f}")

        # Optional conservative saving gate
        if save_models and (te_logloss <= TEST_LOGLOSS_SAVE_MAX):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_path = SAVE_DIR / f"run{run_number}_fold{fold_idx}_tll{te_logloss:.3f}_{timestamp}.json"
            refit_model.save_model(str(model_path))

            metadata_path = SAVE_DIR / f"run{run_number}_fold{fold_idx}_metadata_{timestamp}.json"
            metadata = {
                "run_number": run_number,
                "fold": fold_idx,
                "validation_method": "walk_forward",
                "include_odds": include_odds,
                "best_n_estimators": int(best_n_estimators),
                "metrics": {
                    "test_logloss": float(te_logloss),
                    "test_auc": float(te_auc),
                    "train_logloss": float(tr_logloss),
                    "train_auc": float(tr_auc),
                },
                "selected_feature_count": len(sel_cols_outer),
            }
            metadata_path.write_text(json.dumps(metadata, indent=2))

    print("\n" + "=" * 70)
    print(f"  RUN {run_number} | RESULTS (lower logloss is better)")
    print(f"  Test Logloss: {np.mean(outer_test_logloss):.4f} ± {np.std(outer_test_logloss):.4f}")
    print(f"  Test AUC:     {np.mean(outer_test_auc):.4f} ± {np.std(outer_test_auc):.4f}")
    print("=" * 70)

    return {
        "outer_test_logloss": outer_test_logloss,
        "outer_test_auc": outer_test_auc,
        "outer_train_logloss": outer_train_logloss,
        "outer_train_auc": outer_train_auc,
        "best_params_per_fold": best_params_per_fold,
        "best_n_per_fold": best_n_per_fold,
    }


def train_final_model_on_all_data(X_raw, y, dates, aggregated_params, median_n_estimators,
                                  run_number, include_odds):
    """Train final production model (fit on ALL data, with FS and no leakage in prep)."""
    training_controller.check_pause()

    print("\n" + "=" * 70)
    print(f"  RUN {run_number} | TRAINING FINAL MODEL (fit on ALL data)")
    print("=" * 70)

    # For the final model, do a single leakage-safe preprocessing step by treating
    # ALL data as "train" (since this is the production fit after CV is complete).
    # Here we just impute/clip globally and set categories globally.
    X_all = X_raw.copy()

    # Numeric
    num_cols = X_all.select_dtypes(include=[np.number]).columns.tolist()
    if num_cols:
        med = X_all[num_cols].median().fillna(0)
        X_all[num_cols] = X_all[num_cols].fillna(med)
        max_val = np.finfo(np.float32).max / 10
        min_val = np.finfo(np.float32).min / 10
        X_all[num_cols] = X_all[num_cols].clip(lower=min_val, upper=max_val)

    # Categorical
    obj_cols = X_all.select_dtypes(include=["object", "string", "category"]).columns.tolist()
    for col in obj_cols:
        vals = X_all[col].astype("string").fillna("<NA>")
        cats = pd.Index(sorted(pd.unique(pd.concat([vals, pd.Series(["<NA>"])]))))
        X_all[col] = pd.Categorical(vals, categories=cats)

    # Final feature selection on ALL data (for the production model)
    sel_cols_all, _ = select_top_features_by_xgb(
        X_all, y, k=TOP_K_FEATURES, enabled=USE_TOP_K_FEATURES
    )
    X_all_sel = X_all[sel_cols_all]

    final_params = {
        "objective": "binary:logistic",
        "tree_method": "hist",
        "device": "cuda",
        "predictor": "gpu_predictor",
        "enable_categorical": True,
        "n_estimators": int(median_n_estimators),
        "eval_metric": ["logloss", "error"],
        "early_stopping_rounds": None,
        **aggregated_params,
    }

    final_model = xgb.XGBClassifier(**final_params)
    final_model.fit(X_all_sel, y, verbose=False)

    train_proba = final_model.predict_proba(X_all_sel)[:, 1]
    train_logloss = log_loss(y, train_proba)
    train_auc = roc_auc_score(y, train_proba)

    print(f"  Train LOGLOSS: {train_logloss:.4f} | Train AUC: {train_auc:.4f}")
    print("=" * 70)

    return final_model, final_params, train_logloss, train_auc, sel_cols_all


def train_xgboost_walkforward(optuna_trials: int = 100, outer_cv: int = 5,
                              inner_cv: int = 3, save_models: bool = True,
                              run_number: int = 1, include_odds: bool = True) -> dict:
    """Entry point for walk-forward training with the three fixes applied."""
    print("=" * 70)
    print(f"  RUN {run_number} | XGBoost Walk-Forward Training (nested FS; leakage-safe)")
    print("=" * 70)

    X_raw, y, dates = load_data_for_cv(include_odds=include_odds)

    results = walk_forward_nested_cv(
        X_raw, y, dates,
        outer_cv=outer_cv,
        inner_cv=inner_cv,
        optuna_trials=optuna_trials,
        save_models=save_models,
        run_number=run_number,
        include_odds=include_odds,
    )

    aggregated_params = aggregate_best_params(results["best_params_per_fold"])
    median_n_estimators = int(np.median(results["best_n_per_fold"]))

    final_model, final_params, train_logloss, train_auc, sel_cols_all = train_final_model_on_all_data(
        X_raw, y, dates, aggregated_params, median_n_estimators, run_number, include_odds
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    final_model_path = FINAL_MODEL_DIR / f"run{run_number}_final_model_{timestamp}.json"
    final_model.save_model(str(final_model_path))
    print(f"  ✓ Model saved: {final_model_path.name}")

    meta_path = FINAL_MODEL_DIR / f"run{run_number}_final_metadata_{timestamp}.json"
    meta = {
        "run_number": run_number,
        "model_path": str(final_model_path),
        "selected_feature_count": len(sel_cols_all),
        "selected_features_sample": sel_cols_all[:min(20, len(sel_cols_all))],
        "train_logloss": float(train_logloss),
        "train_auc": float(train_auc),
        "cv_summary": {
            "mean_test_logloss": float(np.mean(results["outer_test_logloss"])),
            "std_test_logloss": float(np.std(results["outer_test_logloss"])),
            "mean_test_auc": float(np.mean(results["outer_test_auc"])),
            "std_test_auc": float(np.std(results["outer_test_auc"])),
        },
    }
    meta_path.write_text(json.dumps(meta, indent=2))

    return {
        "run_number": run_number,
        "model_path": str(final_model_path),
        "mean_test_logloss": float(np.mean(results["outer_test_logloss"])),
        "std_test_logloss": float(np.std(results["outer_test_logloss"])),
        "mean_test_auc": float(np.mean(results["outer_test_auc"])),
        "std_test_auc": float(np.std(results["outer_test_auc"])),
    }


def run_multiple_training_sessions(n_runs: int = 5, optuna_trials: int = 100,
                                   outer_cv: int = 5, inner_cv: int = 3,
                                   save_models: bool = True, include_odds: bool = True):
    """Run multiple training sessions with pause/resume."""
    print("\n" + "█" * 70)
    print(f"█  WALK-FORWARD TRAINING - {n_runs} RUNS (nested FS; leakage-safe)")
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
            "mean_test_logloss": float(np.mean([r["mean_test_logloss"] for r in all_run_results])),
            "std_test_logloss": float(np.std([r["mean_test_logloss"] for r in all_run_results])),
            "mean_test_auc": float(np.mean([r["mean_test_auc"] for r in all_run_results])),
            "std_test_auc": float(np.std([r["mean_test_auc"] for r in all_run_results])),
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2))

    print("\n" + "█" * 70)
    print("█  TRAINING COMPLETE")
    print(f"█  Completed: {len(all_run_results)}/{n_runs} runs")
    print(f"█  Mean Test Logloss: {summary['aggregate_statistics']['mean_test_logloss']:.4f}")
    print(f"█  Mean Test AUC:     {summary['aggregate_statistics']['mean_test_auc']:.4f}")
    print("█" * 70 + "\n")


if __name__ == "__main__":
    try:
        run_multiple_training_sessions(
            n_runs=25,
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
