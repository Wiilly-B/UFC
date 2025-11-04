"""
UFC Fight Prediction - XGBoost (single split, leakage-safe, autosave)
Supports Optuna hyperparameter optimization, feature selection, and dual calibration backends.
"""
from __future__ import annotations
import json, warnings
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass
from optuna.pruners import MedianPruner
import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss
from helper import TrainingController, annotated_trial_plot, set_matplotlib_backend

warnings.filterwarnings("ignore")


# ==================== DATA UTILITIES ====================

def _drop_odds_columns(df: pd.DataFrame, include_odds: bool) -> pd.DataFrame:
    """Drop odds-related columns if requested."""
    if include_odds:
        return df
    terms = ("odd", "open", "opening", "close", "closing")
    drop = [c for c in df.columns if any(t in c.lower() for t in terms)]
    if drop:
        print(f"[Cols] Dropping {len(drop)} odds-related columns.")
    return df.drop(columns=drop, errors="ignore")


def load_datasets(train_path=None, val_path=None, test_path=None, date_column="current_fight_date", include_odds=True):
    """Load train/val/test data with chronological ordering."""
    cfg = CONFIG
    train_path = Path(train_path or cfg.data_dir / "train_data.csv")
    val_path = Path(val_path or cfg.data_dir / "val_data.csv")
    test_path = Path(test_path or cfg.data_dir / "test_data.csv")

    dfs = {}
    for label, path in [("TRAIN", train_path), ("VAL", val_path), ("TEST", test_path)]:
        df = pd.read_csv(path)
        if date_column not in df.columns:
            raise ValueError(f"{label}: Date column '{date_column}' not found.")
        df[date_column] = pd.to_datetime(df[date_column], errors="coerce")
        df.sort_values(date_column, inplace=True)
        df.reset_index(drop=True, inplace=True)
        df = _drop_odds_columns(df, include_odds)
        dfs[label] = df

    drop_cols = ["winner", "fighter_a", "fighter_b", "date"]
    feature_cols = [c for c in dfs["TRAIN"].columns if c not in drop_cols and c != date_column]

    for label in ["TRAIN", "VAL", "TEST"]:
        df = dfs[label]
        print(f"{label}: {df[date_column].min().date()} → {df[date_column].max().date()} | n={len(df)}")
    print(f"Features: {len(feature_cols)}")

    return (
        dfs["TRAIN"][feature_cols].copy(), dfs["TRAIN"]["winner"].copy(),
        dfs["VAL"][feature_cols].copy(), dfs["VAL"]["winner"].copy(),
        dfs["TEST"][feature_cols].copy(), dfs["TEST"]["winner"].copy(),
        feature_cols
    )


# ==================== PREPROCESSING ====================

def fit_transform_preprocess(X_tr_raw: pd.DataFrame, X_va_raw: pd.DataFrame, X_te_raw: pd.DataFrame):
    """Apply leakage-safe preprocessing (fit on TRAIN only)."""
    X_tr, X_va, X_te = X_tr_raw.copy(), X_va_raw.copy(), X_te_raw.copy()

    # Numeric: fill with TRAIN median, clip to safe float32 range
    num_cols = X_tr.select_dtypes(include=[np.number]).columns.tolist()
    if num_cols:
        med = X_tr[num_cols].median().fillna(0)
        for X in [X_tr, X_va, X_te]:
            X[num_cols] = X[num_cols].fillna(med).clip(
                np.finfo(np.float32).min / 10, np.finfo(np.float32).max / 10
            )

    # Categorical: fit categories from TRAIN, unseen -> '<NA>'
    obj_cols = X_tr.select_dtypes(include=["object", "string", "category"]).columns.tolist()
    for col in obj_cols:
        tr_vals = X_tr[col].astype("string").fillna("<NA>")
        cats = pd.Index(sorted(pd.unique(pd.concat([tr_vals, pd.Series(["<NA>"])]))))
        X_tr[col] = pd.Categorical(tr_vals, categories=cats)
        for X in [X_va, X_te]:
            s = X[col].astype("string").fillna("<NA>")
            s = s.where(s.isin(cats), "<NA>")
            X[col] = pd.Categorical(s, categories=cats)

    # Validate numeric safety
    for df, nm in [(X_tr, "TRAIN"), (X_va, "VAL"), (X_te, "TEST")]:
        assert not np.isinf(df.select_dtypes(include=[np.number]).to_numpy()).any(), f"Inf in {nm}"
        assert not df.select_dtypes(include=[np.number]).isna().any().any(), f"Numeric NaN in {nm}"

    return X_tr, X_va, X_te


# ==================== FEATURE SELECTION ====================

def select_top_features_by_xgb(X_tr: pd.DataFrame, y_tr: pd.Series, k: int, enabled: bool):
    """Select top-K features by XGBoost gain importance."""
    if not enabled:
        cols = list(X_tr.columns)
        print(f"[FS] DISABLED → using ALL {len(cols)} features.")
        if CONFIG.feature_preview_n > 0:
            print("[FS] Preview:", ", ".join(cols[:min(CONFIG.feature_preview_n, len(cols))]))
        return cols

    print(f"[FS] ENABLED → selecting top {k} features on TRAIN only...")
    fs_model = xgb.XGBClassifier(
        objective="binary:logistic", tree_method="hist", device="cuda", enable_categorical=True,
        n_estimators=200, max_depth=6, learning_rate=0.1, subsample=0.9, colsample_bytree=0.9,
        eval_metric="logloss", verbosity=0
    )
    fs_model.fit(X_tr, y_tr, verbose=False)

    score = fs_model.get_booster().get_score(importance_type="gain")
    if score and all(k.startswith("f") for k in score.keys()):
        feat_names = fs_model.get_booster().feature_names
        score = {feat_names[int(k[1:])]: v for k, v in score.items() if int(k[1:]) < len(feat_names)}

    if not score:
        print("[FS] Warning: empty gain map; falling back to first K.")
        selected = list(X_tr.columns)[:k]
    else:
        ranked = [n for n, _ in sorted(score.items(), key=lambda kv: kv[1], reverse=True)]
        selected = [c for c in X_tr.columns if c in set(ranked[:k])]

    print(f"[FS] Selected {len(selected)} features.")
    if CONFIG.feature_preview_n > 0 and score:
        topn = [n for n, _ in sorted(score.items(), key=lambda kv: kv[1], reverse=True)][:min(CONFIG.feature_preview_n, len(score))]
        print("[FS] Top by gain:", ", ".join(topn))
    return selected


# ==================== MODEL SAVING ====================

def _save_candidate_model(trial_num, best_n, base_params, X_tr_sel, y_tr, X_va_sel, y_va,
                          X_te_sel=None, y_te=None, run_tag="ufc_xgb_single",
                          evals_result=None, best_idx=None, gap_at_best=None):
    """Refit with fixed n_estimators and save with metrics in filename."""
    cfg = CONFIG
    fixed = {**base_params, "n_estimators": int(best_n), "early_stopping_rounds": None}
    X_refit, y_refit = (pd.concat([X_tr_sel, X_va_sel], axis=0), pd.concat([y_tr, y_va], axis=0)) if cfg.refit_on_train_plus_val else (X_tr_sel, y_tr)

    model = xgb.XGBClassifier(**fixed)
    model.fit(X_refit, y_refit, verbose=False)

    va_proba, va_pred = model.predict_proba(X_va_sel)[:, 1], (model.predict_proba(X_va_sel)[:, 1] >= 0.5).astype(int)
    va_ll, va_acc = log_loss(y_va, va_proba), accuracy_score(y_va, va_pred)
    tr_proba, tr_ll = model.predict_proba(X_tr_sel)[:, 1], log_loss(y_tr, model.predict_proba(X_tr_sel)[:, 1])
    gap = abs(tr_ll - va_ll) if gap_at_best is None else gap_at_best

    test_part = ""
    if cfg.autosave_include_test and X_te_sel is not None and y_te is not None:
        te_proba, te_pred = model.predict_proba(X_te_sel)[:, 1], (model.predict_proba(X_te_sel)[:, 1] >= 0.5).astype(int)
        te_ll, te_acc = log_loss(y_te, te_proba), accuracy_score(y_te, te_pred)
        test_part = f"_TESTacc{te_acc:.3f}_TESTll{te_ll:.3f}"

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    fname = f"{run_tag}_TRIAL{trial_num:03d}_VALacc{va_acc:.3f}_GAP{gap:.3f}_VALll{va_ll:.3f}{test_part}_{ts}.json"
    path = cfg.save_dir / fname
    model.save_model(str(path))
    print(f"  ↳ Autosaved: {path.name}")

    if evals_result is not None and best_idx is not None:
        png_path = cfg.trial_plots_dir / (path.stem + ".png")
        annotated_trial_plot(evals_result, title=f"Autosaved Trial {trial_num} (best@{best_n}, gap={gap:.3f})",
                           best_idx=best_idx, gap=gap, save_path_png=png_path,
                           show_plots=cfg.show_plots, save_plots_as_png=cfg.save_plots_as_png)


# ==================== UTILITIES ====================

def _choose_best_index(evals_result: dict) -> tuple[int, float, float, float]:
    """Choose best iteration based on configured objective."""
    va_ll = np.asarray(evals_result["validation_1"]["logloss"], dtype=float)
    tr_ll = np.asarray(evals_result["validation_0"]["logloss"], dtype=float)
    va_er = np.asarray(evals_result["validation_1"]["error"], dtype=float)
    best_idx = int(np.argmin(va_er if CONFIG.optuna_objective.lower() == "accuracy" else va_ll))
    return best_idx, float(va_ll[best_idx]), float(tr_ll[best_idx]), float(va_er[best_idx])


def train_single_split(optuna_trials=10, include_odds=True, run_tag="ufc_xgb_single", use_gpu=True) -> dict:
    """Train XGBoost with Optuna hyperparameter optimization and autosave."""
    cfg = CONFIG
    print("=" * 70)
    print(f"  SINGLE-SPLIT TRAINER (Objective: {cfg.optuna_objective.upper()})")
    print("=" * 70)

    training_controller.start_listener()
    X_tr_raw, y_tr, X_va_raw, y_va, X_te_raw, y_te, _ = load_datasets(include_odds=include_odds)
    X_tr, X_va, X_te = fit_transform_preprocess(X_tr_raw, X_va_raw, X_te_raw)
    sel_cols = select_top_features_by_xgb(X_tr, y_tr, cfg.top_k_features, cfg.use_top_k_features)
    X_tr_sel, X_va_sel, X_te_sel = X_tr[sel_cols], X_va[sel_cols], X_te[sel_cols]

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    trial_counter = [0]
    prune_metric = "validation_1-error" if cfg.optuna_objective.lower() == "accuracy" else "validation_1-logloss"
    study_direction = "minimize"

    def objective(trial: optuna.Trial) -> float:
        trial_counter[0] += 1
        if trial_counter[0] % 5 == 0:
            training_controller.check_pause()

        params = {
            "objective": "binary:logistic", "tree_method": "hist", "device": "cuda" if use_gpu else "cpu",
            "enable_categorical": True, "eval_metric": ["logloss", "error"], "early_stopping_rounds": 100,
            "n_estimators": trial.suggest_int("n_estimators", 200, 2500),
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.03, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 4),
            "min_child_weight": trial.suggest_int("min_child_weight", 20, 120, step=10),
            "subsample": trial.suggest_float("subsample", 0.55, 0.9),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.55, 0.9),
            "gamma": trial.suggest_float("gamma", 0.001, 10.0, log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.001, 25.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.001, 25.0, log=True),
            "max_delta_step": trial.suggest_int("max_delta_step", 3, 10),
            "sampling_method": "gradient_based",
            "grow_policy": trial.suggest_categorical("grow_policy", ["depthwise", "lossguide"]),
        }

        model = xgb.XGBClassifier(**params)
        # Note: XGBoost 2.0+ removed callbacks parameter, pruning handled by Optuna
        model.fit(X_tr_sel, y_tr, eval_set=[(X_tr_sel, y_tr), (X_va_sel, y_va)], verbose=False)

        ev = model.evals_result()
        if cfg.show_trial_plots:
            annotated_trial_plot(ev, title=f"Trial {trial.number}", best_idx=_choose_best_index(ev)[0],
                               gap=0.0, save_path_png=None, show_plots=cfg.show_plots,
                               save_plots_as_png=cfg.save_plots_as_png)

        best_idx, va_ll, tr_ll, va_err = _choose_best_index(ev)
        gap = abs(tr_ll - va_ll)
        trial.set_user_attr("loss_gap_at_best", float(gap))

        if cfg.autosave_intermediate and (va_ll <= cfg.val_logloss_save_max) and (gap <= cfg.gap_max):
            best_n = best_idx + 1
            base_params = {"objective": "binary:logistic", "tree_method": "hist",
                          "device": "cuda" if use_gpu else "cpu", "enable_categorical": True,
                          "eval_metric": ["logloss", "error"], **{k: v for k, v in trial.params.items()}}
            _save_candidate_model(trial.number, best_n, base_params, X_tr_sel, y_tr, X_va_sel, y_va,
                                X_te_sel if cfg.autosave_include_test else None,
                                y_te if cfg.autosave_include_test else None, run_tag, ev, best_idx, gap)

        return float(va_err if cfg.optuna_objective.lower() == "accuracy" else va_ll)

    study = optuna.create_study(direction=study_direction, sampler=optuna.samplers.TPESampler(),
                               pruner=MedianPruner(n_warmup_steps=10))
    study.optimize(objective, n_trials=optuna_trials)

    best_params = study.best_params
    print(f"\nBest VAL objective ({cfg.optuna_objective}): {study.best_value:.4f}")
    print("Best params:", json.dumps(best_params, indent=2))
    print(f"✓ Optimization complete. Intermediate models autosaved to: {cfg.save_dir}")

    return {
        "best_params": best_params,
        "best_value": float(study.best_value),
        "n_trials": optuna_trials,
        "selected_features": sel_cols,
        "optuna_objective": cfg.optuna_objective,
    }


# ==================== CONFIG ==============================

@dataclass
class Config:
    """XGBoost training configuration."""

    # === DISPLAY / PLOTTING ===
    show_plots: bool = True
    show_trial_plots: bool = True
    feature_preview_n: int = 100
    save_plots_as_png: bool = True

    # === FEATURE SELECTION ===
    use_top_k_features: bool = True
    top_k_features: int = 10000

    # === DATA ===
    include_odds_columns: bool = True
    data_dir: Path = None

    # === REFIT / AUTOSAVE ===
    refit_on_train_plus_val: bool = True
    autosave_intermediate: bool = True
    autosave_include_test: bool = False

    # === OPTUNA OBJECTIVE ===
    optuna_objective: str = "logloss"  # 'logloss' | 'accuracy'

    # === SAVE GATES ===
    val_logloss_save_max: float = 0.69
    gap_max: float = 0.06

    # === PATHS ===
    save_dir: Path = None
    trial_plots_dir: Path = None

    def __post_init__(self):
        """Initialize paths relative to project root."""
        if self.data_dir is None:
            project_root = Path(__file__).resolve().parents[3]
            self.data_dir = project_root / "data" / "train_test"
        if self.save_dir is None:
            project_root = Path(__file__).resolve().parents[3]
            self.save_dir = project_root / "saved_models" / "xgboost" / "single_split"
            self.save_dir.mkdir(parents=True, exist_ok=True)
        if self.trial_plots_dir is None:
            self.trial_plots_dir = self.save_dir / "trial_plots"
            self.trial_plots_dir.mkdir(parents=True, exist_ok=True)


# ==================== INITIALIZATION ==============================

CONFIG = Config(
    show_plots=True,
    show_trial_plots=True,
    feature_preview_n=100,
    save_plots_as_png=True,
    use_top_k_features=True,
    top_k_features=10000,
    include_odds_columns=False,
    refit_on_train_plus_val=True,
    autosave_intermediate=True,
    autosave_include_test=False,
    optuna_objective="logloss",
    val_logloss_save_max=1.0,
    gap_max=0.5,
)

training_controller = TrainingController()
set_matplotlib_backend(CONFIG.show_plots)


# ==================== MAIN ====================

if __name__ == "__main__":
    try:
        res = train_single_split(
            optuna_trials=500,
            include_odds=CONFIG.include_odds_columns,
            run_tag="ufc_xgb_single",
            use_gpu=True,
        )
        print("\nRESULTS:", json.dumps(res, indent=2))
    except KeyboardInterrupt:
        print("\nTraining interrupted. Exiting...")
    finally:
        training_controller.stop()
