"""
Visualization Module for MMA Betting Analysis
Handles all calibration plots and visual analytics
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict
from sklearn.calibration import calibration_curve


def create_calibration_plots(y_test: np.ndarray, y_pred_proba_list: List[np.ndarray], config) -> Dict[str, str]:
    """
    Main function to create all calibration plots based on configuration
    """
    # Create output directory
    os.makedirs(config.output_dir, exist_ok=True)
    calibration_dir = os.path.join(config.output_dir,
                                   config.calibration_type if config.use_calibration else 'uncalibrated')
    os.makedirs(calibration_dir, exist_ok=True)

    # Generate model names
    model_names = [os.path.splitext(f)[0] for f in config.model_files[:len(y_pred_proba_list)]]

    plot_files = {}

    # Create calibration curves
    if config.calibration_type == 'range_based' and config.use_calibration:
        plot_files['reliability'] = _create_range_based_reliability_diagram(
            y_test, y_pred_proba_list, config, calibration_dir, model_names
        )
    else:
        # Standard calibration curves
        plot_files.update(_create_standard_calibration_curves(
            y_test, y_pred_proba_list, config, calibration_dir, model_names
        ))

        # Reliability diagram
        plot_files['reliability'] = _create_reliability_diagram(
            y_test, y_pred_proba_list, config, calibration_dir, model_names
        )

    # Print calibration interpretation
    _print_calibration_interpretation(y_test, y_pred_proba_list, config, model_names)

    # Print plot locations
    print(
        f"\n[{config.calibration_type.capitalize() if config.use_calibration else 'Uncalibrated'} Calibration Plots Generated]")
    for plot_type, filepath in plot_files.items():
        if filepath:
            print(f"{plot_type.replace('_', ' ').title()}: {filepath}")

    return plot_files


def _create_standard_calibration_curves(y_test: np.ndarray, y_pred_proba_list: List[np.ndarray],
                                        config, output_dir: str, model_names: List[str]) -> Dict[str, str]:
    """Create standard calibration curve plots"""
    plot_files = {}

    # Individual model plots if ensemble
    if config.use_ensemble and len(y_pred_proba_list) > 1:
        plt.figure(figsize=(10, 6))
        plt.plot([0, 1], [0, 1], 'k--', label='Perfectly calibrated')

        for i, y_pred_proba in enumerate(y_pred_proba_list):
            prob_true, prob_pred = calibration_curve(y_test, y_pred_proba[:, 1], n_bins=10)
            cal_error = np.mean(np.abs(prob_true - prob_pred))
            plt.plot(prob_pred, prob_true, 's-', label=f'{model_names[i]} (Error: {cal_error:.4f})')

        plt.xlabel('Mean predicted probability')
        plt.ylabel('Fraction of positives')
        plt.title(
            f'Individual Models Calibration Curves ({config.calibration_type.capitalize() if config.use_calibration else "Uncalibrated"})')
        plt.legend(loc='best')
        plt.grid(True)

        filename = os.path.join(output_dir, 'individual_models_calibration.png')
        plt.savefig(filename, dpi=100, bbox_inches='tight')
        plt.close()
        plot_files['individual_models'] = filename

    # Main model plot (ensemble or single)
    plt.figure(figsize=(12, 8))
    plt.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Perfectly calibrated')

    if config.use_ensemble:
        y_pred_proba_avg = np.mean(y_pred_proba_list, axis=0)
        label = 'Ensemble model'
        filename = os.path.join(output_dir, 'ensemble_calibration_curve.png')
    else:
        y_pred_proba_avg = y_pred_proba_list[0]
        label = model_names[0]
        filename = os.path.join(output_dir, f'{model_names[0]}_calibration_curve.png')

    prob_true, prob_pred = calibration_curve(y_test, y_pred_proba_avg[:, 1], n_bins=10)
    cal_error = np.mean(np.abs(prob_true - prob_pred))

    plt.plot(prob_pred, prob_true, 'o-', linewidth=2, markersize=8,
             label=f'{label} (Calibration Error: {cal_error:.4f})')

    # Add annotations
    for i, (x, y) in enumerate(zip(prob_pred, prob_true)):
        plt.annotate(f'({x:.2f}, {y:.2f})', (x, y), textcoords="offset points",
                     xytext=(0, 10), ha='center', fontsize=9)

    plt.xlabel('Mean predicted probability', fontsize=12)
    plt.ylabel('Fraction of positives (true probability)', fontsize=12)
    plt.title(
        f'{label} Calibration Curve ({config.calibration_type.capitalize() if config.use_calibration else "Uncalibrated"})',
        fontsize=14)
    plt.legend(loc='best', fontsize=12)
    plt.grid(True)

    # Add interpretation text
    _add_interpretation_text(plt.gcf(), cal_error)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2)
    plt.savefig(filename, dpi=100)
    plt.close()
    plot_files['main_model'] = filename

    return plot_files


def _create_reliability_diagram(y_test: np.ndarray, y_pred_proba_list: List[np.ndarray],
                                config, output_dir: str, model_names: List[str]) -> str:
    """Create reliability diagram with histogram"""
    if config.use_ensemble:
        y_pred_proba = np.mean(y_pred_proba_list, axis=0)
        title = "Ensemble Model Reliability Diagram"
        filename = os.path.join(output_dir, 'ensemble_reliability_diagram.png')
    else:
        y_pred_proba = y_pred_proba_list[0]
        title = f"{model_names[0]} Reliability Diagram"
        filename = os.path.join(output_dir, f'{model_names[0]}_reliability_diagram.png')

    y_pred_prob_pos = y_pred_proba[:, 1]

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), gridspec_kw={'height_ratios': [1, 3]})

    # Top: histogram
    ax1.hist(y_pred_prob_pos, bins=20, range=(0, 1), histtype='step',
             lw=2, color='blue', density=True)
    ax1.set_ylabel('Density')
    ax1.set_xlim([0, 1])
    ax1.set_title('Distribution of Predicted Probabilities')

    # Bottom: reliability diagram
    prob_true, prob_pred = calibration_curve(y_test, y_pred_prob_pos, n_bins=10)
    cal_error = np.mean(np.abs(prob_true - prob_pred))
    bias = np.mean(prob_true - prob_pred)
    bias_type = "Under-confident" if bias > 0 else "Over-confident" if bias < 0 else "Well-calibrated"

    ax2.plot(prob_pred, prob_true, 's-', label=f'Calibration curve (Error: {cal_error:.4f})',
             color='blue', linewidth=2)
    ax2.plot([0, 1], [0, 1], 'k--', label='Perfectly calibrated')

    # Add calibration gap areas
    for i in range(len(prob_pred)):
        if prob_true[i] > prob_pred[i]:  # Under-confident
            ax2.fill_between([prob_pred[i], prob_pred[i]], [prob_pred[i], prob_true[i]],
                             alpha=0.2, color='green')
        elif prob_true[i] < prob_pred[i]:  # Over-confident
            ax2.fill_between([prob_pred[i], prob_pred[i]], [prob_pred[i], prob_true[i]],
                             alpha=0.2, color='red')

    # Annotate points
    for i, (x, y) in enumerate(zip(prob_pred, prob_true)):
        ax2.annotate(f'({x:.2f}, {y:.2f})', (x, y), textcoords="offset points",
                     xytext=(0, 10), ha='center')

    ax2.set_xlabel('Mean predicted probability')
    ax2.set_ylabel('Fraction of positives (true probability)')
    ax2.set_title(f'{title} - {bias_type}')
    ax2.legend(loc='best')
    ax2.grid(True)

    # Add interpretation text
    textstr = (
        'Interpretation:\n'
        'Green areas: Model is under-confident (true probability > predicted)\n'
        'Red areas: Model is over-confident (predicted > true probability)\n'
        'For Kelly betting: Under-confidence leads to smaller bets than optimal\n'
        'Over-confidence leads to larger bets than optimal\n\n'
        f'Calibration Error: {cal_error:.4f} (Lower is better)\n'
        f'Bias: {bias:.4f} ({bias_type})'
    )
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    fig.text(0.5, 0.02, textstr, fontsize=10, bbox=props, ha='center', va='center')

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.20)
    plt.savefig(filename, dpi=100)
    plt.close()

    return filename


def _create_range_based_reliability_diagram(y_test: np.ndarray, y_pred_proba_list: List[np.ndarray],
                                            config, output_dir: str, model_names: List[str]) -> str:
    """Create reliability diagram comparing original vs range-based calibration"""
    # Import here to avoid circular dependency
    from mma_betting_analysis import RangeBasedCalibrator

    if config.use_ensemble:
        y_pred_proba = np.mean(y_pred_proba_list, axis=0)
        title = "Ensemble Model with Range-Based Calibration"
        filename = os.path.join(output_dir, 'ensemble_range_calibration.png')
    else:
        y_pred_proba = y_pred_proba_list[0]
        title = f"{model_names[0]} with Range-Based Calibration"
        filename = os.path.join(output_dir, f'{model_names[0]}_range_calibration.png')

    y_pred_prob_pos = y_pred_proba[:, 1]

    # Create calibrator
    calibrator = RangeBasedCalibrator(ranges=config.range_calibration_ranges, method='isotonic')
    n_cal = int(len(y_test) * 0.3)
    calibrator.fit(y_pred_prob_pos[:n_cal], y_test[:n_cal])
    calibrated_probs = calibrator.transform(y_pred_prob_pos)

    # Create figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12), gridspec_kw={'height_ratios': [1, 3]})

    # Top: histogram comparison
    ax1.hist(y_pred_prob_pos, bins=20, range=(0, 1), histtype='step',
             lw=2, color='blue', density=True, label='Original')
    ax1.hist(calibrated_probs, bins=20, range=(0, 1), histtype='step',
             lw=2, color='red', density=True, label='Calibrated')
    ax1.set_ylabel('Density')
    ax1.set_xlim([0, 1])
    ax1.set_title('Distribution of Predicted Probabilities')
    ax1.legend(loc='best')

    # Bottom: calibration curves comparison
    prob_true_orig, prob_pred_orig = calibration_curve(y_test, y_pred_prob_pos, n_bins=10)
    cal_error_orig = np.mean(np.abs(prob_true_orig - prob_pred_orig))

    prob_true_cal, prob_pred_cal = calibration_curve(y_test, calibrated_probs, n_bins=10)
    cal_error_cal = np.mean(np.abs(prob_true_cal - prob_pred_cal))

    ax2.plot([0, 1], [0, 1], 'k--', label='Perfectly calibrated')
    ax2.plot(prob_pred_orig, prob_true_orig, 's-', color='blue',
             label=f'Original (Error: {cal_error_orig:.4f})')
    ax2.plot(prob_pred_cal, prob_true_cal, 'o-', color='red',
             label=f'Range Calibrated (Error: {cal_error_cal:.4f})')

    # Mark range boundaries
    for threshold in config.range_calibration_ranges:
        ax2.axvline(x=threshold, color='gray', linestyle=':', alpha=0.7)

    ax2.set_xlabel('Mean predicted probability')
    ax2.set_ylabel('Fraction of positives (true probability)')
    ax2.set_title('Reliability Diagram (Original vs Range-Based Calibration)')
    ax2.legend(loc='best')
    ax2.grid(True)

    # Calculate improvement
    improvement = ((cal_error_orig - cal_error_cal) / cal_error_orig) * 100 if cal_error_orig > 0 else 0

    # Add interpretation text
    textstr = (
        'Interpretation:\n'
        'Points above diagonal: Model is under-confident\n'
        'Points below diagonal: Model is over-confident\n\n'
        f'Original Calibration Error: {cal_error_orig:.4f}\n'
        f'Range-Based Calibration Error: {cal_error_cal:.4f}\n'
        f'Improvement: {improvement:.1f}%'
    )
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    fig.text(0.5, 0.02, textstr, fontsize=10, bbox=props, ha='center', va='center')

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.20)
    plt.savefig(filename, dpi=100)
    plt.close()

    return filename


def _add_interpretation_text(fig, cal_error: float):
    """Add interpretation text box to calibration plot"""
    textstr = (
        'Interpretation:\n'
        'Points above the diagonal: Model is under-confident\n'
        'Points below the diagonal: Model is over-confident\n'
        'Points on the diagonal: Perfect calibration\n'
        f'Calibration Error: {cal_error:.4f} (Lower is better)'
    )
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    fig.text(0.5, 0.02, textstr, fontsize=12, bbox=props, ha='center', va='center')


def _print_calibration_interpretation(y_test: np.ndarray, y_pred_proba_list: List[np.ndarray],
                                      config, model_names: List[str]):
    """Print calibration interpretation for betting"""
    print("\nInterpreting Calibration for Kelly Betting:")
    print("- Perfect calibration means optimal Kelly bet sizing")
    print("- Under-confidence (points above diagonal) leads to smaller bets than optimal")
    print("- Over-confidence (points below diagonal) leads to larger bets than optimal")
    print("- For maximum profit with Kelly criterion, calibration is critical")

    # Calculate calibration metrics
    if config.use_ensemble:
        y_pred_proba_avg = np.mean(y_pred_proba_list, axis=0)
        model_label = "Ensemble model"
    else:
        y_pred_proba_avg = y_pred_proba_list[0]
        model_label = model_names[0]

    prob_true, prob_pred = calibration_curve(y_test, y_pred_proba_avg[:, 1], n_bins=10)
    cal_error = np.mean(np.abs(prob_true - prob_pred))
    bias = np.mean(prob_true - prob_pred)

    print(f"\nAverage calibration error for {model_label}: {cal_error:.4f}")

    if bias > 0:
        print("Model tendency: Under-confident (true probabilities > predicted)")
        print("Betting implication: Bets are smaller than optimal")
    elif bias < 0:
        print("Model tendency: Over-confident (predicted > true probabilities)")
        print("Betting implication: Bets are larger than optimal")
    else:
        print("Model tendency: Well calibrated overall")
        print("Betting implication: Optimal bet sizing")