"""
UFC Fight Prediction - XGBoost (single split, leakage-safe, autosave passing trials)

What it does (chronological, no shuffle):
- Train on TRAIN, validate on VAL, report on TEST (no shuffling, by date).
- Preprocessing fit ONLY on TRAIN, applied to VAL/TEST (no leakage):
  * numeric: median impute (from TRAIN), float32-safe clipping
  * categorical: categories learned on TRAIN; unseen in VAL/TEST -> '<NA>'
- Feature selection (Top-K by gain) fit ONLY on TRAIN; same cols applied to VAL/TEST.
- Optuna objective is configurable:
  * "logloss"  -> minimize VAL logloss
  * "accuracy" -> maximize VAL accuracy (equivalently minimize VAL error)
- For EACH trial:
  * If (VAL logloss ≤ VAL_LOGLOSS_SAVE_MAX) and (gap := |train-VAL| ≤ GAP_MAX),
    then we refit with fixed n_estimators (at the *chosen metric's* best iteration)
    and AUTOSAVE the model immediately.
  * For each autosaved model, we also save a PNG plot (in TRIAL_PLOTS_DIR) annotated with
    best iteration & gap.
- After tuning completes:
  * Refit the final model (TRAIN+VAL by default) with best params & n_estimators
    chosen by the configured objective.
  * Evaluate on TEST and save a final model file with VAL+TEST metrics in filename.

No metadata sidecars — filenames contain the metrics.
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

# ---- Plotting / preview ----
SHOW_PLOTS = True
SHOW_TRIAL_PLOTS = False
FEATURE_PREVIEW_N = 12
if not SHOW_PLOTS:
    matplotlib.use("Agg")

# ---- Feature selection ----
USE_TOP_K_FEATURES = True
TOP_K_FEATURES = 300

# ---- Columns / odds ----
INCLUDE_ODDS_COLUMNS = False  # False -> drop odds/open/close-related columns

# ---- Refit / Autosave / Plots ----
REFIT_ON_TRAIN_PLUS_VAL = True
AUTOSAVE_INTERMEDIATE = True
AUTOSAVE_INCLUDE_TEST = False  # keep False to avoid test peeking
SAVE_PLOTS_AS_PNG = True

# ---- Objective toggle (NEW) ----
# Choose Optuna objective: "logloss" (minimize) or "accuracy" (maximize)
OPTUNA_OBJECTIVE = "accuracy"  # or "accuracy"

# ---- Save gates ----
VAL_LOGLOSS_SAVE_MAX = 0.69  # ~random baseline for balanced classes
GAP_MAX = 0.06  # |train_logloss - val_logloss| at chosen best iteration

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = PROJECT_ROOT / "data" / "train_test"
SAVE_DIR = PROJECT_ROOT / "saved_models" / "xgboost" / "single_split"
TRIAL_PLOTS_DIR = SAVE_DIR / "trial_plots"
for p in (SAVE_DIR, TRIAL_PLOTS_DIR):
    p.mkdir(parents=True, exist_ok=True)


# ==================== PAUSE/RESUME CONTROL ====================

class TrainingController:
    def __init__(self):
        self.paused = False
        self.should_stop = False
        self.pause_lock = threading.Lock()
        self.listener_thread = None
        self.running = False

    def start_listener(self):
        if self.listener_thread is not None and self.listener_thread.is_alive():
            return
        print("\n" + "=" * 70)
        print("  TRAINING CONTROLS ACTIVE")
        print("  Type 'p' + ENTER to PAUSE | 'r' to RESUME | 'q' to QUIT after current op")
        print("=" * 70 + "\n")
        self.running = True
        self.listener_thread = threading.Thread(target=self._keyboard_listener, daemon=True)
        self.listener_thread.start()

    def _keyboard_listener(self):
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
                            cmd = "".join(buf).strip().lower()
                            buf.clear()
                            self._dispatch(cmd)
                        elif ch == "\x03":
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
                            cmd = "".join(line).strip().lower()
                            line.clear()
                            self._dispatch(cmd)
                        else:
                            line.append(ch)
                except Exception:
                    break

    def _dispatch(self, cmd: str):
        if cmd == "p":
            with self.pause_lock:
                if not self.paused:
                    self.paused = True
                    print("\n=== ⏸️  TRAINING PAUSED — 'r' to resume, 'q' to quit ===\n")
        elif cmd == "r":
            with self.pause_lock:
                if self.paused:
                    self.paused = False
                    print("\n=== ▶️  TRAINING RESUMED ===\n")
        elif cmd == "q":
            with self.pause_lock:
                if not self.should_stop:
                    self.should_stop = True
                    print("\n=== 🛑 QUIT REQUESTED — will stop after current operation ===\n")
        elif cmd:
            print(f"[Unknown '{cmd}'] Valid: p, r, q")

    def check_pause(self):
        while True:
            with self.pause_lock:
                if self.should_stop:
                    raise KeyboardInterrupt("Training stopped by user")
                if not self.paused:
                    break
            time.sleep(0.3)

    def stop(self):
        self.running = False
        if self.listener_thread is not None:
            self.listener_thread.join(timeout=1.0)


training_controller = TrainingController()


# ==================== DATA LOADING ====================

def _drop_odds_columns(df: pd.DataFrame, include_odds: bool) -> pd.DataFrame:
    if include_odds:
        return df
    terms = ("odd", "open", "opening", "close", "closing")
    drop = [c for c in df.columns if any(t in c.lower() for t in terms)]
    if drop:
        print(f"[Cols] Dropping {len(drop)} odds-related columns.")
    return df.drop(columns=drop, errors="ignore")


def load_datasets(
        train_path: str | Path | None = None,
        val_path: str | Path | None = None,
        test_path: str | Path | None = None,
        date_column: str = "current_fight_date",
        include_odds: bool = True,
):
    train_path = Path(train_path) if train_path else DATA_DIR / "train_data.csv"
    val_path = Path(val_path) if val_path else DATA_DIR / "val_data.csv"
    test_path = Path(test_path) if test_path else DATA_DIR / "test_data.csv"

    tr = pd.read_csv(train_path)
    va = pd.read_csv(val_path)
    te = pd.read_csv(test_path)

    # Enforce chronological order
    for df, name in [(tr, "TRAIN"), (va, "VAL"), (te, "TEST")]:
        if date_column not in df.columns:
            raise ValueError(f"{name}: Date column '{date_column}' not found.")
        df[date_column] = pd.to_datetime(df[date_column], errors="coerce")
        df.sort_values(date_column, inplace=True)
        df.reset_index(drop=True, inplace=True)

    tr = _drop_odds_columns(tr, include_odds)
    va = _drop_odds_columns(va, include_odds)
    te = _drop_odds_columns(te, include_odds)

    drop_cols = ["winner", "fighter_a", "fighter_b", "date"]
    feature_cols = [c for c in tr.columns if c not in drop_cols and c != date_column]

    X_tr_raw, y_tr = tr[feature_cols].copy(), tr["winner"].copy()
    X_va_raw, y_va = va[feature_cols].copy(), va["winner"].copy()
    X_te_raw, y_te = te[feature_cols].copy(), te["winner"].copy()

    print(f"TRAIN: {tr[date_column].min().date()} → {tr[date_column].max().date()} | n={len(tr)}")
    print(f"VAL  : {va[date_column].min().date()} → {va[date_column].max().date()} | n={len(va)}")
    print(f"TEST : {te[date_column].min().date()} → {te[date_column].max().date()} | n={len(te)}")
    print(f"Features: {len(feature_cols)}")
    return X_tr_raw, y_tr, X_va_raw, y_va, X_te_raw, y_te, feature_cols


# ==================== LEAKAGE-SAFE PREPROCESS ====================

def fit_transform_preprocess(X_tr_raw: pd.DataFrame, X_va_raw: pd.DataFrame, X_te_raw: pd.DataFrame):
    X_tr = X_tr_raw.copy()
    X_va = X_va_raw.copy()
    X_te = X_te_raw.copy()

    # numeric
    num_cols = X_tr.select_dtypes(include=[np.number]).columns.tolist()
    if num_cols:
        med = X_tr[num_cols].median().fillna(0)
        X_tr[num_cols] = X_tr[num_cols].fillna(med)
        X_va[num_cols] = X_va[num_cols].fillna(med)
        X_te[num_cols] = X_te[num_cols].fillna(med)

        max_val = np.finfo(np.float32).max / 10
        min_val = np.finfo(np.float32).min / 10
        X_tr[num_cols] = X_tr[num_cols].clip(min_val, max_val)
        X_va[num_cols] = X_va[num_cols].clip(min_val, max_val)
        X_te[num_cols] = X_te[num_cols].clip(min_val, max_val)

    # categorical (train categories; unseen -> '<NA>')
    obj_cols = X_tr.select_dtypes(include=["object", "string", "category"]).columns.tolist()
    for col in obj_cols:
        tr_vals = X_tr[col].astype("string").fillna("<NA>")
        cats = pd.Index(sorted(pd.unique(pd.concat([tr_vals, pd.Series(["<NA>"])]))))
        X_tr[col] = pd.Categorical(tr_vals, categories=cats)

        def _align(s):
            s2 = s.astype("string").fillna("<NA>")
            s2 = s2.where(s2.isin(cats), "<NA>")
            return pd.Categorical(s2, categories=cats)

        X_va[col] = _align(X_va[col])
        X_te[col] = _align(X_te[col])

    # numeric safety
    for df, nm in [(X_tr, "TRAIN"), (X_va, "VAL"), (X_te, "TEST")]:
        assert not np.isinf(df.select_dtypes(include=[np.number]).to_numpy()).any(), f"Inf in {nm}"
        assert not df.select_dtypes(include=[np.number]).isna().any().any(), f"Numeric NaN in {nm}"

    return X_tr, X_va, X_te


# ==================== FEATURE SELECTION ====================

def select_top_features_by_xgb(X_tr: pd.DataFrame, y_tr: pd.Series, k: int, enabled: bool):
    if not enabled:
        cols = list(X_tr.columns)
        print(f"[FS] DISABLED → using ALL {len(cols)} features.")
        if FEATURE_PREVIEW_N > 0:
            print("[FS] Preview:", ", ".join(cols[:min(FEATURE_PREVIEW_N, len(cols))]))
        return cols

    print(f"[FS] ENABLED → selecting top {k} features on TRAIN only...")
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
    score = booster.get_score(importance_type="gain")

    # map f{i} -> names if needed
    if score and all(k.startswith("f") for k in score.keys()):
        feat_names = booster.feature_names
        score = {feat_names[int(k[1:])]: v for k, v in score.items() if int(k[1:]) < len(feat_names)}

    if not score:
        print("[FS] Warning: empty gain map; falling back to first K.")
        selected = list(X_tr.columns)[:k]
    else:
        ranked = [n for n, _ in sorted(score.items(), key=lambda kv: kv[1], reverse=True)]
        keep = set(ranked[:k])
        selected = [c for c in X_tr.columns if c in keep]

    print(f"[FS] Selected {len(selected)} features.")
    if FEATURE_PREVIEW_N > 0 and score:
        topn = [n for n, _ in sorted(score.items(), key=lambda kv: kv[1], reverse=True)][
               :min(FEATURE_PREVIEW_N, len(score))]
        print("[FS] Top by gain:", ", ".join(topn))
    return selected


# ==================== PLOTTING ====================

def _annotated_trial_plot(evals_result: dict, title: str, best_idx: int, gap: float, save_path_png: Path | None):
    if not evals_result:
        return
    tr_ll = evals_result.get("validation_0", {}).get("logloss", None)
    va_ll = evals_result.get("validation_1", {}).get("logloss", None)
    tr_er = evals_result.get("validation_0", {}).get("error", None)
    va_er = evals_result.get("validation_1", {}).get("error", None)
    if tr_ll is None or va_ll is None or tr_er is None or va_er is None:
        return

    iters = np.arange(1, len(tr_ll) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(title)

    # Loss
    axes[0].plot(iters, tr_ll, linewidth=2, label="Train Logloss")
    axes[0].plot(iters, va_ll, linewidth=2, label="Val Logloss")
    axes[0].axvline(best_idx + 1, linestyle="--", linewidth=1)
    axes[0].annotate(f"best@{best_idx + 1}\ngap={gap:.3f}",
                     xy=(best_idx + 1, va_ll[best_idx]),
                     xytext=(best_idx + 1, max(va_ll) * 0.9),
                     arrowprops=dict(arrowstyle="->", lw=1), fontsize=9)
    axes[0].set_xlabel("Boosting Rounds")
    axes[0].set_ylabel("Log Loss")
    axes[0].legend()
    axes[0].set_title("Loss")

    # Accuracy
    axes[1].plot(iters, 1.0 - np.asarray(tr_er), linewidth=2, label="Train Accuracy")
    axes[1].plot(iters, 1.0 - np.asarray(va_er), linewidth=2, label="Val Accuracy")
    axes[1].axvline(best_idx + 1, linestyle="--", linewidth=1)
    axes[1].set_xlabel("Boosting Rounds")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    axes[1].set_title("Accuracy")

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    if SAVE_PLOTS_AS_PNG and save_path_png is not None:
        fig.savefig(save_path_png, dpi=120)

    if SHOW_PLOTS:
        plt.show(block=False)
        plt.pause(0.001)
    else:
        plt.close(fig)


# ==================== SAVE HELPERS ====================

def _save_candidate_model(
        trial_num: int,
        best_n: int,
        base_params: dict,
        X_tr_sel: pd.DataFrame, y_tr: pd.Series,
        X_va_sel: pd.DataFrame, y_va: pd.Series,
        X_te_sel: pd.DataFrame | None = None, y_te: pd.Series | None = None,
        run_tag: str = "ufc_xgb_single",
        evals_result: dict | None = None,
        best_idx: int | None = None,
        gap_at_best: float | None = None,
):
    """Refit with fixed n_estimators (TRAIN+VAL if REFIT_ON_TRAIN_PLUS_VAL) and save immediately.
       Also saves an annotated PNG plot if available.
    """
    fixed = {**base_params, "n_estimators": int(best_n), "early_stopping_rounds": None}
    if REFIT_ON_TRAIN_PLUS_VAL:
        X_refit = pd.concat([X_tr_sel, X_va_sel], axis=0)
        y_refit = pd.concat([y_tr, y_va], axis=0)
    else:
        X_refit, y_refit = X_tr_sel, y_tr

    model = xgb.XGBClassifier(**fixed)
    model.fit(X_refit, y_refit, verbose=False)

    # Compute VAL metrics for filename + gating
    va_proba = model.predict_proba(X_va_sel)[:, 1]
    va_pred = (va_proba >= 0.5).astype(int)
    va_ll = log_loss(y_va, va_proba)
    va_acc = accuracy_score(y_va, va_pred)

    # Train logloss for gap
    tr_proba = model.predict_proba(X_tr_sel)[:, 1]
    tr_ll = log_loss(y_tr, tr_proba)
    gap = abs(tr_ll - va_ll) if gap_at_best is None else gap_at_best

    # Optional TEST peek (kept off by default)
    test_part = ""
    if AUTOSAVE_INCLUDE_TEST and X_te_sel is not None and y_te is not None:
        te_proba = model.predict_proba(X_te_sel)[:, 1]
        te_pred = (te_proba >= 0.5).astype(int)
        te_ll = log_loss(y_te, te_proba)
        te_acc = accuracy_score(y_te, te_pred)
        test_part = f"_TESTacc{te_acc:.3f}_TESTll{te_ll:.3f}"

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    fname = f"{run_tag}_TRIAL{trial_num:03d}_VALacc{va_acc:.3f}_GAP{gap:.3f}_VALll{va_ll:.3f}{test_part}_{ts}.json"
    path = SAVE_DIR / fname
    model.save_model(str(path))
    print(f"  ↳ Autosaved: {path.name}")

    # Save plot PNG with the same stem — in TRIAL_PLOTS_DIR (separate folder)
    if evals_result is not None and best_idx is not None:
        png_path = TRIAL_PLOTS_DIR / (path.stem + ".png")
        _annotated_trial_plot(
            evals_result,
            title=f"Autosaved Trial {trial_num} (best@{best_n}, gap={gap:.3f})",
            best_idx=best_idx,
            gap=gap,
            save_path_png=png_path,
        )


# ==================== TRAINER ====================

def _choose_best_index(evals_result: dict) -> tuple[int, float, float, float]:
    """
    Return (best_idx, va_ll_best, tr_ll_best, val_error_best) based on OPTUNA_OBJECTIVE.
    - For "logloss": pick idx with minimal validation logloss.
    - For "accuracy": pick idx with maximal validation accuracy == minimal validation error.
    """
    va_ll = np.asarray(evals_result["validation_1"]["logloss"], dtype=float)
    tr_ll = np.asarray(evals_result["validation_0"]["logloss"], dtype=float)
    va_er = np.asarray(evals_result["validation_1"]["error"], dtype=float)

    if OPTUNA_OBJECTIVE.lower() == "accuracy":
        # maximize accuracy == minimize error
        best_idx = int(np.argmin(va_er))
    else:
        # default: minimize logloss
        best_idx = int(np.argmin(va_ll))

    return best_idx, float(va_ll[best_idx]), float(tr_ll[best_idx]), float(va_er[best_idx])


def train_single_split(
        optuna_trials: int = 80,
        include_odds: bool = True,
        run_tag: str = "ufc_xgb_single",
        use_gpu: bool = True,
) -> dict:
    print("=" * 70)
    print("  SINGLE-SPLIT TRAINER (leakage-safe, autosave passing trials)")
    print(f"  Optuna objective: {OPTUNA_OBJECTIVE.upper()}")
    print("=" * 70)

    training_controller.start_listener()

    X_tr_raw, y_tr, X_va_raw, y_va, X_te_raw, y_te, _ = load_datasets(include_odds=include_odds)
    X_tr, X_va, X_te = fit_transform_preprocess(X_tr_raw, X_va_raw, X_te_raw)

    # Feature selection on TRAIN only
    sel_cols = select_top_features_by_xgb(X_tr, y_tr, TOP_K_FEATURES, USE_TOP_K_FEATURES)
    X_tr_sel, X_va_sel, X_te_sel = X_tr[sel_cols], X_va[sel_cols], X_te[sel_cols]

    # Optuna search
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    trial_counter = [0]

    # Configure pruning metric & study direction based on objective
    if OPTUNA_OBJECTIVE.lower() == "accuracy":
        prune_metric = "validation_1-error"  # lower = better
        study_direction = "minimize"  # we minimize error
    else:
        prune_metric = "validation_1-logloss"
        study_direction = "minimize"

    def objective(trial: optuna.Trial) -> float:
        trial_counter[0] += 1
        if trial_counter[0] % 5 == 0:
            training_controller.check_pause()

        params = {
            "objective": "binary:logistic",
            "tree_method": "hist",
            "device": "cuda" if use_gpu else "cpu",
            "enable_categorical": True,
            "n_estimators": trial.suggest_int("n_estimators", 100, 3000),
            "eval_metric": ["logloss", "error"],
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.08, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 64.0, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "gamma": trial.suggest_float("gamma", 1e-3, 10.0, log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-2, 200.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-2, 200.0, log=True),
            "max_delta_step": trial.suggest_int("max_delta_step", 0, 3),
            "grow_policy": trial.suggest_categorical("grow_policy", ["depthwise", "lossguide"]),
            "max_leaves": trial.suggest_int("max_leaves", 16, 256) if trial.params.get(
                "grow_policy") == "lossguide" else 0,
            "max_bin": trial.suggest_int("max_bin", 64, 512),
            "sampling_method": "gradient_based",
            "early_stopping_rounds": 50,
        }

        model = xgb.XGBClassifier(**params)
        model.fit(
            X_tr_sel, y_tr,
            eval_set=[(X_tr_sel, y_tr), (X_va_sel, y_va)],
            verbose=False,
            callbacks=[XGBoostPruningCallback(trial, prune_metric)],
        )

        ev = model.evals_result()
        if SHOW_TRIAL_PLOTS:
            _annotated_trial_plot(
                ev,
                title=f"Trial {trial.number}",
                best_idx=_choose_best_index(ev)[0],
                gap=0.0,
                save_path_png=None
            )

        # Choose best by configured objective
        best_idx, va_ll_at_best, tr_ll_at_best, va_err_at_best = _choose_best_index(ev)
        gap_at_best = abs(tr_ll_at_best - va_ll_at_best)
        trial.set_user_attr("loss_gap_at_best", float(gap_at_best))

        # ✅ AUTOSAVE ANY PASSING TRIAL (mid-run) — gating still by logloss + gap
        if AUTOSAVE_INTERMEDIATE and (va_ll_at_best <= VAL_LOGLOSS_SAVE_MAX) and (gap_at_best <= GAP_MAX):
            best_n = best_idx + 1
            _save_candidate_model(
                trial_num=trial.number,
                best_n=best_n,
                base_params={
                    "objective": "binary:logistic",
                    "tree_method": "hist",
                    "device": "cuda" if use_gpu else "cpu",
                    "enable_categorical": True,
                    "eval_metric": ["logloss", "error"],
                    **{k: v for k, v in trial.params.items()},
                },
                X_tr_sel=X_tr_sel, y_tr=y_tr,
                X_va_sel=X_va_sel, y_va=y_va,
                X_te_sel=X_te_sel if AUTOSAVE_INCLUDE_TEST else None,
                y_te=y_te if AUTOSAVE_INCLUDE_TEST else None,
                run_tag=run_tag,
                evals_result=ev,
                best_idx=best_idx,
                gap_at_best=gap_at_best,
            )

        # Return objective value to minimize
        if OPTUNA_OBJECTIVE.lower() == "accuracy":
            return float(va_err_at_best)  # minimize error == maximize accuracy
        else:
            return float(va_ll_at_best)  # minimize logloss

    study = optuna.create_study(direction=study_direction,
                                sampler=optuna.samplers.TPESampler(),
                                pruner=MedianPruner(n_warmup_steps=10))
    study.optimize(objective, n_trials=optuna_trials)

    best_params = study.best_params
    print(f"\nBest VAL objective ({OPTUNA_OBJECTIVE}): {study.best_value:.4f}")
    print("Best params:", json.dumps(best_params, indent=2))

    # === Final stage: lock best_iteration on TRAIN vs VAL using configured objective
    final_stage_params = {
        "objective": "binary:logistic",
        "tree_method": "hist",
        "device": "cuda" if use_gpu else "cpu",
        "enable_categorical": True,
        "eval_metric": ["logloss", "error"],
        "early_stopping_rounds": 100,
        **best_params,
    }

    stage_model = xgb.XGBClassifier(**final_stage_params)
    stage_model.fit(
        X_tr_sel, y_tr,
        eval_set=[(X_tr_sel, y_tr), (X_va_sel, y_va)],
        verbose=False,
    )
    ev = stage_model.evals_result()

    best_idx, va_ll_at_best, tr_ll_at_best, _va_err_at_best = _choose_best_index(ev)
    best_n = best_idx + 1
    loss_gap = abs(tr_ll_at_best - va_ll_at_best)

    # Optionally show/save annotated plot for the final stage
    final_png = TRIAL_PLOTS_DIR / f"{run_tag}_FINAL_stage.png" if SAVE_PLOTS_AS_PNG else None
    _annotated_trial_plot(ev, title=f"Final Stage (best@{best_n}, gap={loss_gap:.3f})",
                          best_idx=best_idx, gap=loss_gap, save_path_png=final_png)

    # Refit production model with fixed n_estimators (TRAIN+VAL default)
    fixed_params = {**final_stage_params, "n_estimators": int(best_n), "early_stopping_rounds": None}
    if REFIT_ON_TRAIN_PLUS_VAL:
        X_refit = pd.concat([X_tr_sel, X_va_sel], axis=0)
        y_refit = pd.concat([y_tr, y_va], axis=0)
    else:
        X_refit, y_refit = X_tr_sel, y_tr

    refit_model = xgb.XGBClassifier(**fixed_params)
    refit_model.fit(X_refit, y_refit, verbose=False)

    # Evaluate on VAL (for filename) and TEST (final reporting)
    va_proba = refit_model.predict_proba(X_va_sel)[:, 1]
    va_pred = (va_proba >= 0.5).astype(int)
    va_logloss = log_loss(y_va, va_proba)
    va_acc = accuracy_score(y_va, va_pred)

    te_proba = refit_model.predict_proba(X_te_sel)[:, 1]
    te_pred = (te_proba >= 0.5).astype(int)
    te_logloss = log_loss(y_te, te_proba)
    te_acc = accuracy_score(y_te, te_pred)
    te_auc = roc_auc_score(y_te, te_proba)

    print(f"\nVAL  -> Logloss {va_logloss:.4f} | Acc {va_acc:.3f} | Gap |train-VAL| {loss_gap:.4f}")
    print(f"TEST -> Logloss {te_logloss:.4f} | Acc {te_acc:.3f} | AUC {te_auc:.4f}")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    final_name = f"{run_tag}_FINAL_VALacc{va_acc:.3f}_GAP{loss_gap:.3f}_VALll{va_logloss:.3f}_TESTacc{te_acc:.3f}_TESTll{te_logloss:.3f}_{ts}.json"
    final_path = SAVE_DIR / final_name
    refit_model.save_model(str(final_path))
    print(f"✓ Saved FINAL model: {final_path.name}")

    return {
        "best_n_estimators": int(best_n),
        "val_logloss": float(va_logloss),
        "val_accuracy": float(va_acc),
        "loss_gap": float(loss_gap),
        "test_logloss": float(te_logloss),
        "test_accuracy": float(te_acc),
        "test_auc": float(te_auc),
        "selected_features": sel_cols,
        "refit_on_train_plus_val": REFIT_ON_TRAIN_PLUS_VAL,
        "optuna_objective": OPTUNA_OBJECTIVE,
    }


# ==================== MAIN ====================

if __name__ == "__main__":
    try:
        training_controller.start_listener()
        res = train_single_split(
            optuna_trials=250,  # tune as needed
            include_odds=INCLUDE_ODDS_COLUMNS,
            run_tag="ufc_xgb_single",
            use_gpu=True,
        )
        print("\nRESULTS:", json.dumps(res, indent=2))
    except KeyboardInterrupt:
        print("\nTraining interrupted. Exiting...")
    finally:
        training_controller.stop()
