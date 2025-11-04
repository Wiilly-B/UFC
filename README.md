# UFC Fight Outcome Prediction and Betting Strategy Analysis

A comprehensive machine learning system for predicting UFC fight outcomes and evaluating betting strategies. The system employs ensemble XGBoost models with probability calibration and Kelly Criterion optimization to achieve 60-68% prediction accuracy with ROI potential of 10-50% depending on betting strategy and market conditions.

## Overview

This project implements a complete end-to-end pipeline that processes historical UFC fight data, trains predictive models, and evaluates betting strategies. The system demonstrates advanced machine learning techniques, including feature engineering with ELO rating systems, ensemble modeling, probability calibration, and optimal bet sizing strategies.

**Key Capabilities:**
- Advanced feature engineering with dynamic ELO ratings and historical fighter statistics
- Ensemble XGBoost models with hyperparameter optimization via Optuna
- Isotonic and sigmoid probability calibration for accurate confidence estimates
- Kelly Criterion and fixed-fraction betting strategy evaluation with bankroll simulation
- Individual fight predictions with detailed statistical analysis

## Project Structure

```
UFC/
├── data/
│   ├── raw/                          # Raw UFC data files (provided)
│   │   ├── ufc_fight_stats.csv       # Round-by-round fight statistics
│   │   ├── ufc_fight_results.csv     # Fight outcomes and methods
│   │   ├── ufc_event_details.csv     # Event information and dates
│   │   ├── ufc_fighter_details.csv   # Fighter demographics
│   │   ├── ufc_fighter_tott.csv      # Fighter tale of the tape
│   │   └── fight_odds.csv            # Historical betting odds
│   ├── processed/                    # Intermediate processed data
│   │   ├── ufc_fight_merged.csv      # Merged raw data
│   │   ├── ufc_fight_processed.csv   # Cleaned and processed data
│   │   ├── combined_rounds.csv       # Round-level statistics
│   │   └── combined_sorted_fighter_stats.csv
│   ├── matchup data/                 # Matchup-formatted datasets
│   └── train_test/                   # ML-ready datasets
│       ├── train_data.csv            # Training set (chronological)
│       ├── val_data.csv              # Validation set (2024 fights)
│       └── test_data.csv             # Test set (2025 fights)
├── saved_models/
│   ├── xgboost/single_split/         # Trained XGBoost models
│   │   ├── ufc_xgb_single_TRIAL*.json
│   │   └── trial_plots/              # Training visualizations
│   └── encoders/                     # Feature encoders
│       └── category_encoder.pkl
├── src/
│   ├── data_processing/
│   │   ├── scraping/                 # Web scraping utilities (optional)
│   │   ├── cleaning/                 # Data processing pipeline
│   │   │   ├── consolidate_and_clean.py  # Merge raw data
│   │   │   └── data_cleaner.py       # Feature engineering and splits
│   │   └── features/
│   │       ├── Elo.py                # ELO rating system
│   │       └── helper.py             # Feature engineering utilities
│   └── models/
│       ├── xgboost_optimizer/        # Model training and optimization
│       │   ├── main.py               # Optuna hyperparameter tuning
│       │   └── helper.py             # Training utilities
│       ├── predict_testset/          # Model evaluation
│       │   ├── main.py               # Test set evaluation and betting
│       │   ├── betting_module.py     # Betting strategy simulation
│       │   └── visualization.py      # Calibration plots (optional)
│       └── create_predictions/       # Individual predictions
│           └── create_individual_matchup.py
└── notebooks/                        # Exploratory analysis (optional)
```

## Installation

### Prerequisites

- Python 3.8 or higher
- CPU-only training supported (no GPU required)

### Setup Instructions

1. **Clone the repository:**
```bash
git clone https://github.com/Wiilly-B/UFC.git
cd UFC
```

2. **Create and activate virtual environment (recommended):**
```bash

sudo apt install python3-venv
python3 -m venv venv
source venv/bin/activate

```

3. **Git LFS:**

Install git LSF
```bash
sudo apt update
sudo apt install git-lfs
```
Run these commands in the terminal to fetch CSV files.
```bash
git lfs install
git lfs pull
git lfs checkout
```

4. **Install dependencies:**
```bash
sudo apt install python3-pip
pip install -r requirements.txt
```

## Complete Workflow

### Step 1: Data Processing

**Objective:** Transform raw UFC data into machine learning-ready training, validation, and test sets.

#### Option A: Use Existing Processed Data (Recommended for Quick Start)

The repository includes pre-processed data in `data/train_test/`. If these files exist, proceed directly to **Step 2: Model Training**.

#### Option B: Process Data from Raw Files (Complete Pipeline)

**Step 1a: Merge Raw Data Files**

```bash
cd src/data_processing/cleaning
python consolidate_and_clean.py
```

**Process:**
- Merges `ufc_fight_stats.csv`, `ufc_fight_results.csv`, and `ufc_event_details.csv`
- Creates `data/raw/ufc_fight_merged.csv`
- Processes into `data/processed/ufc_fight_processed.csv`

**Operations:**
- Combines fight statistics (strikes, takedowns, control time, knockdowns)
- Integrates fight results (winners, methods, finish times)
- Adds event metadata (dates, locations)
- Standardizes column names and data types
- Converts time formats and percentages

**Step 1b: Feature Engineering and Data Splitting**

```bash
python data_cleaner.py
```

**Process:**
1. Loads `data/processed/ufc_fight_processed.csv`
2. **Calculates ELO ratings** for all fighters chronologically
3. **Engineers comprehensive features:**
   - Historical averages over last N fights (default: 3)
   - Win/loss streaks and momentum indicators
   - Fighter age (rounded) and years of experience
   - Days since last fight and activity penalties
   - Betting odds (opening, closing) and line movement
   - ELO ratings, differences, and ratios
   - Fighting style metrics (striker, grappler, pressure scores)
   - Opponent quality adjustments
   - Finish rates and damage ratios
4. **Creates matchup format** (Fighter A vs Fighter B rows)
5. **Applies feature selection** based on importance and correlation
6. **Removes highly correlated features** (correlation > 0.95)
7. **Splits data chronologically:**
   - Training: Historical fights (15 years back from validation start)
   - Validation: 2024 fights (used for calibration)
   - Test: 2025 fights (unseen data for evaluation)

**Configuration** (`data_cleaner.py` - ProcessingPipeline class):
```python
test_start_date: str = '2025-01-01'      # Test set start
test_end_date: str = '2025-12-31'        # Test set end
years_back: int = 15                     # Training data history
n_past_fights: int = 3                   # Historical statistics window
correlation_threshold: float = 0.95      # Feature correlation cutoff
select_features: bool = True             # Apply feature selection
```

**Output:**
- `data/train_test/train_data.csv` (approximately 2000 fights)
- `data/train_test/val_data.csv` (approximately 400 fights)
- `data/train_test/test_data.csv` (approximately 200 fights)
- `data/train_test/removed_features.txt` (list of removed correlated features)

**Expected Runtime:** 2-5 minutes

---

### Step 2: Model Training

**Objective:** Train ensemble XGBoost models with automated hyperparameter optimization.

```bash
cd ../..
cd models/xgboost_optimizer
python main.py
```

**Process:**
1. Loads train, validation, and test datasets
2. Applies preprocessing (median imputation, categorical encoding)
3. Performs feature selection (top 10,000 features by importance)
4. **Runs Optuna hyperparameter optimization** (500 trials default)
5. Tests hyperparameter combinations:
   - Learning rate (eta)
   - Maximum tree depth
   - Minimum child weight
   - Subsample ratio
   - Column subsample ratio
   - L1 regularization (alpha)
   - L2 regularization (lambda)
6. **Auto-saves models** after each trial meeting quality thresholds
7. Uses early stopping to prevent overfitting
8. Generates training curves and performance plots

**Configuration** (`main.py` - bottom of file):
```python
train_single_split(
    optuna_trials=500,          # Number of optimization trials
    include_odds=False,         # Include/exclude odds features
    run_tag="ufc_xgb_single",   # Model filename prefix
    use_gpu=True,               # Use GPU if available
)

# Additional config in Config class:
show_plots=True                 # Display training curves
save_plots_as_png=True          # Save trial visualizations
refit_on_train_plus_val=True    # Refit on combined train+val
val_logloss_save_max=1.0        # Maximum validation logloss for saving
gap_max=0.5                     # Maximum train-val gap for saving
```

**Interactive Controls:**
- Press `s` during a trial to skip to the next trial
- Press `q` to quit optimization early (saves all progress)

**Output:**
- `saved_models/xgboost/single_split/ufc_xgb_single_TRIAL000.json`
- `saved_models/xgboost/single_split/ufc_xgb_single_TRIAL001.json`
- ... (one model file per trial that meets save thresholds)
- `saved_models/xgboost/single_split/trial_plots/*.png` (training curves)

**Model Filename Format:**
```
ufc_xgb_single_TRIAL{num}_VALacc{acc}_GAP{gap}_VALll{logloss}_{timestamp}.json
```

**Expected Results:**
- Validation accuracy: 66-69%
- Training time: 30-90 minutes depending on CPU and number of trials
- Typical output: 15-30 saved models meeting quality thresholds

---

### Step 3: Model Evaluation and Betting Analysis

**Objective:** Evaluate model ensemble on test set and simulate betting strategies.

```bash
cd ../..
cd models/predict_testset
python main.py
```

**IMPORTANT - First Run Configuration:**

On your **first run only**, you must create the category encoder. Edit the configuration at the bottom of `main.py`:

```python
CONFIG = Config(
    ...
    require_trained_encoder=False,  # Change from True to False for first run only
)
```

After the first successful run, change this back to `True`. The encoder will be saved to `saved_models/encoders/category_encoder.pkl` and reused for all subsequent runs and individual predictions.

**Process:**
1. Loads all trained XGBoost models from `saved_models/xgboost/single_split/`
2. Creates or loads categorical feature encoder
3. Loads validation and test datasets
4. **Applies probability calibration** using validation set:
   - Isotonic regression (non-parametric, monotonic calibration)
   - Adjusts raw model probabilities to match true outcome frequencies
   - Critical for accurate Kelly Criterion bet sizing
5. **Generates predictions** on test set
6. **Simulates betting strategies:**
   - Kelly Criterion (optimal bet sizing based on edge)
   - Fixed Fraction (constant percentage betting)
7. Tracks daily and monthly bankroll progression
8. Filters bets based on odds thresholds
9. Calculates comprehensive performance metrics

**Configuration** (`main.py` - CONFIG at bottom):
```python
CONFIG = Config(
    # Calibration Settings
    use_calibration=True,
    calibration_type='isotonic',       # 'isotonic' or 'sigmoid'
    calibration_backend='cv',          # 'cv' or 'simple'

    # Betting Strategy Parameters
    initial_bankroll=10000,            # Starting capital
    kelly_fraction=0.25,               # Quarter Kelly (conservative)
    fixed_bet_fraction=0.1,            # 10% fixed bet size
    max_bet_percentage=0.1,            # Maximum 10% per bet

    # Betting Filters
    min_odds=-300,                     # Skip heavy favorites
    max_underdog_odds=200,             # Skip extreme underdogs
    odds_type='close',                 # 'open', 'close', or 'average'

    # Model Configuration
    use_ensemble=True,                 # Average all models
    require_trained_encoder=True,      # False for first run only

    # File Paths
    model_dir='../../../saved_models/xgboost/single_split/',
    model_filename_pattern=r'ufc_xgb_single_(?:TRIAL\d{3}|FINAL).*\.json$',
    val_data_path='../../../data/train_test/val_data.csv',
    test_data_path='../../../data/train_test/test_data.csv',
    encoder_path='../../../saved_models/encoders/category_encoder.pkl',
)
```

**Output Display:**

```
================================================================================
PREDICTION METRICS
================================================================================
Accuracy:   65.3%
Precision:  67.2%
Recall:     64.1%
F1 Score:   65.6%
ROC-AUC:    0.701

================================================================================
BETTING PERFORMANCE SUMMARY
================================================================================

Fixed Fraction Strategy (10% of bankroll):
  Total Bets:        147
  Correct:           98 (66.7%)
  Total Wagered:     $14,700
  Final Bankroll:    $12,450
  Net Profit:        $2,450
  ROI:               16.7%
  Max Drawdown:      -$890

Kelly Criterion Strategy (0.25 fraction):
  Total Bets:        147
  Correct:           98 (66.7%)
  Total Wagered:     $8,320
  Final Bankroll:    $13,780
  Net Profit:        $3,780
  ROI:               45.4%
  Max Drawdown:      -$1,240
  Sharpe Ratio:      1.87

Monthly Performance:
  2025-01:    +8.3%
  2025-02:    +12.1%
  2025-03:    +5.7%
  ...
```

**Expected Results:**
- Test set accuracy: 60-67%
- Kelly Criterion ROI: 10-50% (highly variable due to small sample size)
- Fixed Fraction ROI: 5-25%
- Results vary significantly based on market conditions and sample size

**Output Files:**
- `saved_models/encoders/category_encoder.pkl` (created on first run)

**Runtime:** 1-2 minutes

---

### Step 4: Individual Fight Predictions

**Objective:** Generate predictions for specific upcoming matchups with calibrated probabilities.

```bash
cd ../..
cd models/create_predictions
python create_individual_matchup.py
```

**Configuration** (edit bottom of `create_individual_matchup.py`):

```python
# Fighter Information
FIGHTER_A_NAME = "Sean O'Malley"              # Fighter A name (must match historical data)
FIGHTER_B_NAME = "Merab Dvalishvili"          # Fighter B name
REFERENCE_DATE = "2025-09-15"                 # Fight date (YYYY-MM-DD)

# Betting Odds (American format)
FIGHTER_A_ODDS = -180                         # Negative = favorite
FIGHTER_B_ODDS = 155                          # Positive = underdog
FIGHTER_A_CLOSING_ODDS = -200                 # Closing odds (if available)
FIGHTER_B_CLOSING_ODDS = 175

# Historical Data Settings
N_PAST_FIGHTS = 3                             # Number of recent fights for statistics
N_DETAILED_RESULTS = 3                        # Recent fight results to include
APPLY_FEATURE_SELECTION = True                # Use same features as training

# Model Settings (top of file)
MODEL_DIR = '../../../saved_models/xgboost/single_split/'
ENCODER_PATH = '../../../saved_models/encoders/category_encoder.pkl'
VAL_DATA_PATH = '../../../data/train_test/val_data.csv'
USE_CALIBRATION = True
CALIBRATION_TYPE = 'isotonic'
CALIBRATION_BACKEND = 'cv'
USE_ENSEMBLE = True
```

**Process:**
1. Retrieves historical fight data for both fighters
2. Calculates current ELO ratings based on fight history
3. Computes all features (streaks, averages, time-based features)
4. Loads trained model ensemble and encoder
5. Runs prediction through all models
6. Applies probability calibration using validation set
7. Averages predictions across ensemble
8. Displays results with model agreement metrics

**Example Output:**

```
================================================================================
CREATING INDIVIDUAL MATCHUP WITH FEATURE SELECTION
================================================================================

Fighters: Sean O'Malley vs Merab Dvalishvili
Date: 2025-09-15
Total Features: 247

--- KEY STATISTICS ---
current_fight_age: 30
current_fight_age_b: 33
current_fight_pre_fight_elo_a: 1685
current_fight_pre_fight_elo_b: 1702
current_fight_win_streak_a: 3
current_fight_win_streak_b: 10
current_fight_days_since_last_a: 180
current_fight_days_since_last_b: 165

================================================================================
LOADING MODELS AND RUNNING PREDICTION
================================================================================

1. Loading models...
Loaded 16 models from saved_models/xgboost/single_split

2. Loading encoder...
Encoder loaded successfully

3. Loading validation data for calibration...
Validation data loaded: 421 fights

4. Preparing matchup features...
Features prepared and encoded

5. Applying calibration (isotonic/cv)...
Calibration complete

================================================================================
PREDICTION RESULTS
================================================================================

Fight: Sean O'Malley vs Merab Dvalishvili
Date: 2025-09-15

--- CALIBRATED PROBABILITIES ---
Sean O'Malley wins:      43.2%
Merab Dvalishvili wins:  56.8%

--- PREDICTION ---
Predicted Winner:  Merab Dvalishvili
Confidence:        56.8%
Models Agreeing:   14/16

================================================================================
Analysis Complete
================================================================================
```

**Usage Instructions:**
1. Identify upcoming UFC fight
2. Obtain current betting odds from sportsbook
3. Edit configuration parameters in `create_individual_matchup.py`
4. Ensure fighter names exactly match historical data
5. Run script to generate prediction
6. Compare model probabilities to implied odds
7. Identify value betting opportunities (positive edge)

**Runtime:** Less than 10 seconds per prediction

---

## Understanding the System

### Probability Calibration

**Purpose:** Raw XGBoost probabilities may be systematically over-confident or under-confident. Calibration adjusts these probabilities to match real-world outcome frequencies.

**Methods:**
- **Isotonic Regression** (recommended): Non-parametric, flexible, monotonic calibration
- **Sigmoid (Platt Scaling)**: Parametric calibration assuming sigmoid-shaped miscalibration

**Example:**
- Uncalibrated: Model outputs 75% probability, fighter actually wins 65% of the time
- Calibrated: Model outputs 75% probability, fighter wins approximately 75% of the time

**Importance for Betting:** Calibrated probabilities are essential for accurate Kelly Criterion bet sizing. Miscalibrated probabilities lead to suboptimal bet sizes.

### Kelly Criterion

**Formula:** `f* = (p × (b + 1) - 1) / b`

Where:
- `f*` = fraction of bankroll to bet
- `p` = probability of winning
- `b` = decimal odds minus 1

**Benefits:**
- Mathematically optimal bet sizing for long-term growth
- Accounts for edge and odds
- Maximizes expected logarithmic growth rate

**Fractional Kelly:** Uses a fraction of the Kelly recommendation (e.g., 0.25 = Quarter Kelly) to reduce variance and drawdown risk.

### Ensemble Modeling

**Approach:** Multiple XGBoost models with different hyperparameters are trained and averaged.

**Advantages:**
- Reduces overfitting and model variance
- More robust to outliers and unusual matchups
- Model agreement serves as confidence indicator
- 16/16 models agree = high confidence
- 8/16 models agree = low confidence, uncertain prediction

### ELO Rating System

**Implementation:** Chess-style rating system adapted for MMA.

**Adjustments:**
- Weight class multipliers
- Title fight importance (1.5x)
- Win method (KO/SUB > Decision)
- Age factors (prime years: 27-32)
- Opponent quality
- Expected outcome based on rating difference

**Updates:** Ratings update chronologically after each fight, preventing data leakage.

### Feature Engineering

**Categories:**

1. **Historical Statistics** (last 3 fights average):
   - Significant strikes landed/absorbed per minute
   - Takedown success and defense rates
   - Control time and ground position
   - Knockdowns and submission attempts

2. **Streak and Momentum:**
   - Current win/loss streaks
   - Recent form (last 3 fight results)
   - Momentum indicators
   - Finish rates

3. **Time-Based Features:**
   - Fighter age (rounded to nearest year)
   - Days since last fight
   - Years of UFC experience
   - Layoff penalties and ring rust

4. **Betting Market Features:**
   - Opening odds
   - Closing odds (sharper, more accurate)
   - Line movement (sharp vs public money)
   - Implied probability

5. **ELO Features:**
   - Current ELO ratings
   - ELO difference
   - ELO ratio
   - Pre-fight and post-fight ELO

6. **Fighting Style:**
   - Striker score (striking preference)
   - Grappler score (grappling preference)
   - Pressure score (forward pressure)
   - Style matchup indicators

7. **Opponent Quality:**
   - Opponent average ELO
   - Quality-adjusted wins
   - Opponent recent form
   - Opponent finish threat

### Chronological Data Splitting

**Critical for validity:** All data splitting is strictly chronological to prevent information leakage.

**Splits:**
- **Training:** All fights more than 15 years before test set
- **Validation:** 2024 fights (used for calibration only)
- **Test:** 2025 fights (true out-of-sample evaluation)

**Guarantee:** Model never trains on future information.

---

## Troubleshooting

### Error: "Encoder not found"

**Cause:** First run of `predict_testset/main.py` without encoder.

**Solution:**
```python
# Edit main.py CONFIG:
require_trained_encoder=False  # Change to False for first run
```

After first successful run, change back to `True`.

### Error: "No model files found"

**Cause:** No trained models in `saved_models/xgboost/single_split/`.

**Solution:** Complete Step 2 (Model Training) first:
```bash
cd src/models/xgboost_optimizer
python main.py
```

### Error: "ModuleNotFoundError"

**Cause:** Running script from incorrect directory.

**Solution:** Always run scripts from their containing directory:
```bash
# Correct:
cd src/models/predict_testset
python main.py

# Incorrect:
python src/models/predict_testset/main.py  # May cause import errors
```

### Error: "Fighter not found"

**Cause:** Fighter name in individual matchup doesn't match historical data.

**Solution:**
- Check exact spelling and capitalization
- Use same name format as in historical data
- Check `data/processed/ufc_fight_processed.csv` for correct names

### Low Accuracy or Negative ROI

**Possible Causes:**
1. **Market Efficiency:** Betting odds already incorporate most predictive information
2. **Small Sample Size:** Test set of ~200 fights has high variance
3. **Variance:** Short-term results are inherently volatile
4. **Calibration Issues:** Try different calibration methods
5. **Odds Timing:** Closing odds are sharper than opening odds

**Recommendations:**
- Use fractional Kelly (0.25) to reduce variance
- Filter heavy favorites (min_odds = -300)
- Focus on medium-confidence predictions (55-65% probability range)
- Evaluate performance over larger samples (500+ bets)
- Consider market conditions and line movement

### Training Taking Too Long

**Solution:** Reduce number of trials:
```python
# In xgboost_optimizer/main.py:
train_single_split(
    optuna_trials=50,  # Reduce from 500 to 50
    ...
)
```

Quality may decrease slightly with fewer trials.

---

## Performance Expectations

### Accuracy Ranges

**Training Set:** 70-75%
- Expected with potential overfitting
- Not indicative of real-world performance

**Validation Set:** 66-69%
- More realistic performance estimate
- Used for calibration

**Test Set:** 60-67%
- True out-of-sample performance
- Most reliable indicator

### ROI Expectations

**Highly Variable Due To:**
- Small sample size (approximately 200 test fights)
- Market efficiency (odds are well-calibrated)
- Fight outcome variance (knockouts, injuries)
- Vig/juice (approximately 4-5% house edge)

**Realistic Ranges:**
- **Poor Period:** -10% to 0%
- **Average Period:** 5-15%
- **Good Period:** 15-30%
- **Exceptional Period:** 30-50%+

**Long-term expectations:** Positive ROI is difficult but achievable with proper calibration and bet sizing.

### Performance Indicators

**Positive Signs:**
- Test accuracy consistently above 60%
- Positive cumulative ROI over 100+ bets
- Kelly Criterion outperforms Fixed Fraction
- High model agreement correlates with accuracy
- Calibration error below 0.05

**Warning Signs:**
- Test accuracy below 55% over large sample
- Consistently losing on favorites
- Low model agreement (8/16 or worse)
- Negative ROI across all strategies
- Poor calibration (calibration error > 0.10)

---

## Configuration Reference

### Data Processing

**File:** `src/data_processing/cleaning/data_cleaner.py`

```python
class ProcessingPipeline:
    test_start_date: str = '2025-01-01'      # Test set start date
    test_end_date: str = '2025-12-31'        # Test set end date
    years_back: int = 15                     # Training data history
    n_past_fights: int = 3                   # Historical statistics window
    n_detailed_results: int = 3              # Recent fight results
    correlation_threshold: float = 0.95      # Feature correlation cutoff
    select_features: bool = True             # Apply feature selection
    include_names: bool = True               # Include fighter names in output
```

### Model Training

**File:** `src/models/xgboost_optimizer/main.py`

```python
# Main execution:
train_single_split(
    optuna_trials=500,                       # Number of Optuna trials
    include_odds=False,                      # Include odds features
    use_gpu=True,                            # Use GPU if available
)

# Config class:
class Config:
    show_plots: bool = True                  # Display training curves
    save_plots_as_png: bool = True           # Save trial plots
    refit_on_train_plus_val: bool = True     # Refit on train+val
    use_top_k_features: bool = True          # Feature selection
    top_k_features: int = 10000              # Maximum features
    val_logloss_save_max: float = 1.0        # Save threshold (logloss)
    gap_max: float = 0.5                     # Save threshold (train-val gap)
```

### Prediction and Betting

**File:** `src/models/predict_testset/main.py`

```python
CONFIG = Config(
    # Calibration
    use_calibration=True,
    calibration_type='isotonic',             # 'isotonic' or 'sigmoid'
    calibration_backend='cv',                # 'cv' or 'simple'

    # Betting
    initial_bankroll=10000,                  # Starting capital
    kelly_fraction=0.25,                     # Quarter Kelly
    fixed_bet_fraction=0.1,                  # Fixed 10%
    max_bet_percentage=0.1,                  # Max 10% per bet
    min_odds=-300,                           # Favorite threshold
    max_underdog_odds=200,                   # Underdog threshold
    odds_type='close',                       # Odds type

    # Model
    use_ensemble=True,                       # Use all models
    require_trained_encoder=True,            # False for first run
)
```

---

## Technical Stack

**Core Technologies:**
- **Python 3.8+:** Primary programming language
- **XGBoost 3.0+:** Gradient boosting framework
- **Optuna 3.3+:** Hyperparameter optimization
- **scikit-learn 1.3+:** Calibration, preprocessing, metrics
- **pandas 2.0+:** Data manipulation
- **numpy 1.24+:** Numerical computations
- **matplotlib 3.7+:** Visualization
- **rich 13.0+:** Terminal user interface

**Optional:**
- **Selenium 4.12+:** Web scraping (data collection only)
- **rapidfuzz 3.0+:** Fuzzy string matching
- **tqdm 4.65+:** Progress bars

---

## Future Development

**Planned Enhancements:**
- Web dashboard for live predictions and bet tracking
- Real-time odds integration for automated value detection
- Fighter style matchup analysis (striker vs grappler tendencies)
- Automated model retraining pipeline with new fight data
- Monte Carlo simulation for bankroll variance analysis
- Expanded feature engineering (fight location, altitude, fighter camp changes)
- Integration with additional data sources

---


## Contact

williambillqinshen@gmail.com
