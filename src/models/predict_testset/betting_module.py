"""
Betting Module for MMA Analysis
Handles betting calculations, evaluation, and results display.
"""
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Tuple
from dataclasses import dataclass
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.columns import Columns
from rich.console import Group


class BettingEvaluator:
    """Evaluates betting strategies and calculates returns."""

    def __init__(self, config):
        self.config = config
        self.console = Console(width=160)

    def calculate_profit(self, odds: float, stake: float) -> float:
        """Calculate profit from American odds."""
        return (100 / abs(odds) if odds < 0 else odds / 100) * stake

    def calculate_kelly_fraction(self, p: float, b: float) -> float:
        """Calculate Kelly criterion optimal bet size."""
        return max(0, p - (1 - p) / b)

    def calculate_average_odds(self, open_odds: float, close_odds: float) -> float:
        """Calculate average odds between open and close."""
        def american_to_decimal(odds):
            return (odds / 100) + 1 if odds > 0 else (100 / abs(odds)) + 1
        avg_decimal = (american_to_decimal(open_odds) + american_to_decimal(close_odds)) / 2
        return round((avg_decimal - 1) * 100) if avg_decimal > 2 else round(-100 / (avg_decimal - 1))

    def calculate_expected_value(self, probability: float, odds: float) -> float:
        """Calculate expected value given win probability and American odds.

        EV = (probability × decimal_odds) - 1

        Args:
            probability: Win probability (0 to 1)
            odds: American odds (e.g., -150, +200)

        Returns:
            Expected value as a decimal (e.g., 0.05 = 5% edge)
        """
        # Convert American odds to decimal
        decimal_odds = (odds / 100) + 1 if odds > 0 else (100 / abs(odds)) + 1
        return (probability * decimal_odds) - 1

    def evaluate_bets(self, y_test: pd.Series, y_pred_proba_list: List[np.ndarray],
                      test_data: pd.DataFrame) -> Tuple:
        """
        Evaluate betting performance while ensuring consistent predictions regardless of event order
        """
        # Store fight data by unique identifier
        fight_data = {}

        # Determine number of models available
        num_models = len(y_pred_proba_list) if self.config.use_ensemble else 1

        # Build mapping of fights to predictions
        for i in range(len(test_data)):
            row = test_data.iloc[i]
            fight_id = (row['current_fight_date'], frozenset([row['fighter_a'], row['fighter_b']]))

            # Calculate prediction
            if self.config.use_ensemble:
                y_pred_proba_avg = np.mean([y_pred_proba[i] for y_pred_proba in y_pred_proba_list], axis=0)
                models_agreeing = sum([1 for y_pred_proba in y_pred_proba_list if
                                       (y_pred_proba[i][1] > y_pred_proba[i][0]) ==
                                       (y_pred_proba_avg[1] > y_pred_proba_avg[0])])
            else:
                y_pred_proba_avg = y_pred_proba_list[0][i]
                models_agreeing = 1

            fight_data[fight_id] = {
                'row': row,
                'prediction': y_pred_proba_avg,
                'true_outcome': y_test.iloc[i],
                'models_agreeing': models_agreeing,
                'num_models': num_models
            }

        # Initialize tracking variables
        fixed_bankroll = self.config.initial_bankroll
        kelly_bankroll = self.config.initial_bankroll
        fixed_total_volume = 0
        kelly_total_volume = 0
        fixed_correct_bets = 0
        kelly_correct_bets = 0
        fixed_total_bets = 0
        kelly_total_bets = 0
        confident_predictions = 0
        correct_confident_predictions = 0
        confident_bets = []

        # Sort dates for chronological processing
        all_dates = sorted(set(row['current_fight_date'] for row in [data['row'] for data in fight_data.values()]))

        # Initialize bankroll tracking
        daily_fixed_bankrolls = {}
        daily_kelly_bankrolls = {}
        daily_fixed_profits = {}
        daily_kelly_profits = {}

        available_fixed_bankroll = fixed_bankroll
        available_kelly_bankroll = kelly_bankroll

        # Process fights chronologically
        for current_date in all_dates:
            daily_fixed_profits[current_date] = 0
            daily_kelly_profits[current_date] = 0

            # Get all fights for current date
            date_fights = [(fight_id, fight_info) for fight_id, fight_info in fight_data.items()
                           if fight_id[0] == current_date]

            # Process each fight
            for fight_id, fight_info in date_fights:
                row = fight_info['row']
                y_pred_proba_avg = fight_info['prediction']
                true_outcome = fight_info['true_outcome']
                models_agreeing = fight_info['models_agreeing']
                num_models = fight_info['num_models']

                # Determine true winner
                true_winner = row['fighter_a'] if true_outcome == 1 else row['fighter_b']

                # Get probabilities for both fighters
                prob_fighter_a = y_pred_proba_avg[1]  # probability fighter A wins
                prob_fighter_b = y_pred_proba_avg[0]  # probability fighter B wins

                # Determine predicted winner for tracking purposes
                if prob_fighter_a > prob_fighter_b:
                    predicted_winner = row['fighter_a']
                else:
                    predicted_winner = row['fighter_b']

                # Track confident predictions
                confident_predictions += 1
                if predicted_winner == true_winner:
                    correct_confident_predictions += 1

                # Get odds for both fighters
                open_odds_a = row['current_fight_open_odds']
                close_odds_a = row['current_fight_closing_odds']
                open_odds_b = row['current_fight_open_odds_b']
                close_odds_b = row['current_fight_closing_odds_b']

                # Determine which odds to use
                if self.config.odds_type == 'open':
                    odds_a, odds_b = open_odds_a, open_odds_b
                elif self.config.odds_type == 'close':
                    odds_a, odds_b = close_odds_a, close_odds_b
                else:  # 'average'
                    odds_a = self.calculate_average_odds(open_odds_a, close_odds_a)
                    odds_b = self.calculate_average_odds(open_odds_b, close_odds_b)

                # Calculate expected value for both fighters
                ev_fighter_a = self.calculate_expected_value(prob_fighter_a, odds_a)
                ev_fighter_b = self.calculate_expected_value(prob_fighter_b, odds_b)

                # Determine which fighter (if any) to bet on based on EV
                # min_models: requires this many models to agree on prediction direction
                # Uses 60% agreement threshold scaled to actual number of models available
                min_models = max(1, int(np.ceil(num_models * 0.6))) if self.config.use_ensemble else 1

                # Use min_probability if available, otherwise fall back to manual_threshold
                min_prob = getattr(self.config, 'min_probability', self.config.manual_threshold)

                # Collect bets for both fighters if they have positive EV
                bets_to_place = []

                # Check fighter A for positive EV
                if ev_fighter_a > self.config.min_ev_threshold and models_agreeing >= min_models:
                    if prob_fighter_a >= min_prob:
                        # Skip if odds outside range
                        if not (odds_a < self.config.min_odds or odds_a > self.config.max_underdog_odds):
                            bets_to_place.append({
                                'fighter': row['fighter_a'],
                                'probability': prob_fighter_a,
                                'odds': odds_a,
                                'ev': ev_fighter_a
                            })

                # Check fighter B for positive EV
                if ev_fighter_b > self.config.min_ev_threshold and models_agreeing >= min_models:
                    if prob_fighter_b >= min_prob:
                        # Skip if odds outside range
                        if not (odds_b < self.config.min_odds or odds_b > self.config.max_underdog_odds):
                            bets_to_place.append({
                                'fighter': row['fighter_b'],
                                'probability': prob_fighter_b,
                                'odds': odds_b,
                                'ev': ev_fighter_b
                            })

                # Place all valid bets (can be 0, 1, or 2 bets per fight)
                for bet_info in bets_to_place:
                    bet_fighter = bet_info['fighter']
                    winning_probability = bet_info['probability']
                    odds = bet_info['odds']
                    ev = bet_info['ev']

                    # Calculate stakes
                    fixed_available_before = available_fixed_bankroll
                    kelly_available_before = available_kelly_bankroll

                    fixed_max_bet = fixed_available_before * self.config.max_bet_percentage
                    kelly_max_bet = kelly_available_before * self.config.max_bet_percentage

                    fixed_stake = min(fixed_available_before * self.config.fixed_bet_fraction,
                                      fixed_available_before, fixed_max_bet)

                    # Kelly calculation
                    b = odds / 100 if odds > 0 else 100 / abs(odds)
                    full_kelly = self.calculate_kelly_fraction(winning_probability, b)
                    adjusted_kelly = full_kelly * self.config.kelly_fraction
                    kelly_stake = min(kelly_available_before * adjusted_kelly,
                                      kelly_available_before, kelly_max_bet)

                    # Store bet information
                    bet_result = {
                        'Fight': confident_predictions,
                        'Fighter A': row['fighter_a'],
                        'Fighter B': row['fighter_b'],
                        'Date': current_date,
                        'True Winner': true_winner,
                        'Predicted Winner': predicted_winner,
                        'Bet On': bet_fighter,
                        'Confidence': f"{winning_probability:.2%}",
                        'Expected Value': f"{ev:.2%}",
                        'Odds': odds,
                        'Models Agreeing': models_agreeing,
                        'Num Models': num_models
                    }

                    # Process fixed bet
                    if fixed_stake > 0:
                        fixed_total_bets += 1
                        available_fixed_bankroll -= fixed_stake
                        fixed_profit = self.calculate_profit(odds, fixed_stake)
                        fixed_total_volume += fixed_stake

                        bet_result.update({
                            'Fixed Fraction Starting Bankroll': f"${fixed_bankroll:.2f}",
                            'Fixed Fraction Available Bankroll': f"${fixed_available_before:.2f}",
                            'Fixed Fraction Stake': f"${fixed_stake:.2f}",
                            'Fixed Fraction Potential Profit': f"${fixed_profit:.2f}",
                        })

                        if bet_fighter == true_winner:
                            daily_fixed_profits[current_date] += fixed_profit
                            fixed_correct_bets += 1
                            bet_result['Fixed Fraction Profit'] = fixed_profit
                        else:
                            daily_fixed_profits[current_date] -= fixed_stake
                            bet_result['Fixed Fraction Profit'] = -fixed_stake

                        bet_result[
                            'Fixed Fraction Bankroll After'] = f"${(fixed_bankroll + daily_fixed_profits[current_date]):.2f}"
                        bet_result['Fixed Fraction ROI'] = (bet_result[
                                                                'Fixed Fraction Profit'] / fixed_available_before) * 100

                    # Process Kelly bet
                    if kelly_stake > 0:
                        kelly_total_bets += 1
                        available_kelly_bankroll -= kelly_stake
                        kelly_profit = self.calculate_profit(odds, kelly_stake)
                        kelly_total_volume += kelly_stake

                        bet_result.update({
                            'Kelly Starting Bankroll': f"${kelly_bankroll:.2f}",
                            'Kelly Available Bankroll': f"${kelly_available_before:.2f}",
                            'Kelly Stake': f"${kelly_stake:.2f}",
                            'Kelly Potential Profit': f"${kelly_profit:.2f}",
                        })

                        if bet_fighter == true_winner:
                            daily_kelly_profits[current_date] += kelly_profit
                            kelly_correct_bets += 1
                            bet_result['Kelly Profit'] = kelly_profit
                        else:
                            daily_kelly_profits[current_date] -= kelly_stake
                            bet_result['Kelly Profit'] = -kelly_stake

                        bet_result[
                            'Kelly Bankroll After'] = f"${(kelly_bankroll + daily_kelly_profits[current_date]):.2f}"
                        bet_result['Kelly ROI'] = (bet_result['Kelly Profit'] / kelly_available_before) * 100

                    confident_bets.append(bet_result)

            # Update bankrolls at end of day
            daily_fixed_bankrolls[current_date] = fixed_bankroll + daily_fixed_profits[current_date]
            daily_kelly_bankrolls[current_date] = kelly_bankroll + daily_kelly_profits[current_date]

            fixed_bankroll = daily_fixed_bankrolls[current_date]
            kelly_bankroll = daily_kelly_bankrolls[current_date]
            available_fixed_bankroll = fixed_bankroll
            available_kelly_bankroll = kelly_bankroll

        # Print fight results
        self._print_fight_results(confident_bets)

        return (fixed_bankroll, fixed_total_volume, fixed_correct_bets, fixed_total_bets,
                kelly_bankroll, kelly_total_volume, kelly_correct_bets, kelly_total_bets,
                confident_predictions, correct_confident_predictions,
                daily_fixed_bankrolls, daily_kelly_bankrolls)

    def _print_fight_results(self, confident_bets: List[Dict]):
        """Print detailed results for each bet."""
        for bet in confident_bets:
            fighter_a, fighter_b = bet['Fighter A'].title(), bet['Fighter B'].title()
            date_obj = datetime.strptime(bet['Date'], '%Y-%m-%d')
            formatted_date = date_obj.strftime('%B %d, %Y')

            # Calculate stake percentages
            fixed_avail = float(bet.get('Fixed Fraction Available Bankroll', '0').replace('$', ''))
            fixed_stake = float(bet.get('Fixed Fraction Stake', '0').replace('$', ''))
            fixed_pct = (fixed_stake / fixed_avail) * 100 if fixed_avail > 0 else 0
            kelly_avail = float(bet.get('Kelly Available Bankroll', '0').replace('$', ''))
            kelly_stake = float(bet.get('Kelly Stake', '0').replace('$', ''))
            kelly_pct = (kelly_stake / kelly_avail) * 100 if kelly_avail > 0 else 0

            # Create panels
            fixed_panel = Panel(
                f"Starting: {bet.get('Fixed Fraction Starting Bankroll', 'N/A')}\n"
                f"Available: {bet.get('Fixed Fraction Available Bankroll', 'N/A')}\n"
                f"Stake: {bet.get('Fixed Fraction Stake', 'N/A')} ({fixed_pct:.2f}%)\n"
                f"Potential: {bet.get('Fixed Fraction Potential Profit', 'N/A')}\n"
                f"After: {bet.get('Fixed Fraction Bankroll After', 'N/A')}\n"
                f"Profit: ${bet.get('Fixed Fraction Profit', 0):.2f}\n"
                f"ROI: {bet.get('Fixed Fraction ROI', 0):.2f}%",
                title="Fixed Fraction", expand=True, width=42
            )

            kelly_panel = Panel(
                f"Starting: {bet.get('Kelly Starting Bankroll', 'N/A')}\n"
                f"Available: {bet.get('Kelly Available Bankroll', 'N/A')}\n"
                f"Stake: {bet.get('Kelly Stake', 'N/A')} ({kelly_pct:.2f}%)\n"
                f"Potential: {bet.get('Kelly Potential Profit', 'N/A')}\n"
                f"After: {bet.get('Kelly Bankroll After', 'N/A')}\n"
                f"Profit: ${bet.get('Kelly Profit', 0):.2f}\n"
                f"ROI: {bet.get('Kelly ROI', 0):.2f}%",
                title="Kelly", expand=True, width=42
            )

            fight_info = Group(
                Text(f"True: {bet['True Winner'].title()}", style="green"),
                Text(f"Predicted: {bet['Predicted Winner'].title()}", style="blue"),
                Text(f"Bet On: {bet['Bet On'].title()}", style="magenta"),
                Text(f"Confidence: {bet['Confidence']}", style="yellow"),
                Text(f"Expected Value: {bet['Expected Value']}", style="cyan"),
                Text(f"Models: {bet['Models Agreeing']}/{bet['Num Models']}", style="white")
            )

            main_panel = Panel(
                Group(Panel(fight_info, title="Fight Info"),
                      Columns([fixed_panel, kelly_panel], equal=False, expand=False, align="left")),
                title=f"Fight {bet['Fight']}: {fighter_a} vs {fighter_b} on {formatted_date}",
                subtitle=f"Odds: {bet['Odds']}", width=89
            )
            self.console.print(main_panel, style="magenta")

    def calculate_monthly_roi(self, daily_bankrolls: Dict, is_kelly: bool = False) -> Tuple[Dict, Dict, float]:
        """Calculate monthly ROI and profits"""
        monthly_roi = {}
        monthly_profit = {}
        current_month = None
        current_bankroll = self.config.initial_bankroll
        month_start_bankroll = self.config.initial_bankroll
        total_profit = 0

        print(f"\nDetailed {'Kelly' if is_kelly else 'Fixed Fraction'} ROI Calculation:")
        print(f"{'Month':<10}{'Profit':<15}{'ROI':<10}{'Start Bankroll':<20}{'End Bankroll':<20}")
        print("-" * 80)

        sorted_dates = sorted(daily_bankrolls.keys())
        for date in sorted_dates:
            bankroll = daily_bankrolls[date]
            month = date[:7]  # Extract YYYY-MM

            if month != current_month:
                if current_month is not None:
                    profit = current_bankroll - month_start_bankroll
                    monthly_profit[current_month] = profit
                    total_profit += profit
                    roi = (profit / month_start_bankroll) * 100
                    monthly_roi[current_month] = roi
                    print(
                        f"{current_month:<10}${profit:<14.2f}{roi:<10.2f}${month_start_bankroll:<19.2f}${current_bankroll:<19.2f}")

                current_month = month
                month_start_bankroll = current_bankroll

            current_bankroll = bankroll

        # Handle last month
        if current_month is not None:
            profit = current_bankroll - month_start_bankroll
            monthly_profit[current_month] = profit
            total_profit += profit
            roi = (profit / month_start_bankroll) * 100
            monthly_roi[current_month] = roi
            print(
                f"{current_month:<10}${profit:<14.2f}{roi:<10.2f}${month_start_bankroll:<19.2f}${current_bankroll:<19.2f}")

        total_roi = (total_profit / self.config.initial_bankroll) * 100
        sum_monthly_roi = sum(monthly_roi.values())

        print("-" * 80)
        print(f"{'Total':<10}${total_profit:<14.2f}{total_roi:<10.2f}")
        print(f"\nSum of monthly ROIs: {sum_monthly_roi:.2f}%")
        print(f"Total ROI: {total_roi:.2f}%")
        print(f"Difference: {total_roi - sum_monthly_roi:.2f}%")

        print("\nDebug Information:")
        print(f"Number of events in dataset: {len(sorted_dates)}")
        print(f"First date: {sorted_dates[0]}, Last date: {sorted_dates[-1]}")
        print(f"Initial bankroll: ${self.config.initial_bankroll:.2f}, Final bankroll: ${current_bankroll:.2f}")

        return monthly_roi, monthly_profit, total_roi

    def print_results(self, bet_results: Tuple, test_data: pd.DataFrame):
        """Print comprehensive betting results"""
        (fixed_final_bankroll, fixed_total_volume, fixed_correct_bets, fixed_total_bets,
         kelly_final_bankroll, kelly_total_volume, kelly_correct_bets, kelly_total_bets,
         confident_predictions, correct_confident_predictions,
         daily_fixed_bankrolls, daily_kelly_bankrolls) = bet_results

        if not daily_fixed_bankrolls or not daily_kelly_bankrolls:
            return

        # Calculate ROI
        earliest_fight_date = test_data['current_fight_date'].min()
        daily_fixed_roi = self._calculate_daily_roi(daily_fixed_bankrolls)
        daily_kelly_roi = self._calculate_daily_roi(daily_kelly_bankrolls)

        if daily_fixed_roi and daily_kelly_roi:
            self._print_daily_roi(daily_fixed_roi, daily_kelly_roi)

        # Calculate monthly ROI
        fixed_monthly_roi, fixed_monthly_profit, fixed_total_roi = self.calculate_monthly_roi(daily_fixed_bankrolls,
                                                                                              False)
        kelly_monthly_roi, kelly_monthly_profit, kelly_total_roi = self.calculate_monthly_roi(daily_kelly_bankrolls,
                                                                                              True)

        # Print monthly results
        if fixed_monthly_roi and kelly_monthly_roi:
            self._print_monthly_roi_table(fixed_monthly_roi, kelly_monthly_roi, fixed_total_roi, kelly_total_roi)

        # Print final summary
        self._print_betting_summary(
            len(test_data), confident_predictions, correct_confident_predictions,
            fixed_total_bets, fixed_correct_bets, fixed_final_bankroll,
            fixed_total_volume, kelly_final_bankroll, kelly_total_volume,
            kelly_correct_bets, kelly_total_bets, earliest_fight_date,
            fixed_monthly_profit, kelly_monthly_profit
        )

    def _calculate_daily_roi(self, daily_bankrolls: Dict) -> Dict:
        """Calculate daily ROI based on bankroll changes"""
        daily_roi = {}
        previous_bankroll = self.config.initial_bankroll

        for date, bankroll in sorted(daily_bankrolls.items()):
            daily_profit = bankroll - previous_bankroll
            daily_roi[date] = (daily_profit / previous_bankroll) * 100
            previous_bankroll = bankroll

        return daily_roi

    def _print_daily_roi(self, daily_fixed_roi: Dict, daily_kelly_roi: Dict):
        """Print daily ROI table"""
        console = Console()
        console.print("\nDaily ROI:")
        table = Table(title="Daily Return on Investment")
        table.add_column("Date", style="cyan")
        table.add_column("Fixed Fraction ROI", justify="right", style="magenta")
        table.add_column("Kelly ROI", justify="right", style="green")

        for date in sorted(daily_fixed_roi.keys()):
            table.add_row(date, f"{daily_fixed_roi[date]:.2f}%", f"{daily_kelly_roi[date]:.2f}%")

        console.print(table)

    def _print_monthly_roi_table(self, fixed_monthly_roi: Dict, kelly_monthly_roi: Dict,
                                 fixed_total_roi: float, kelly_total_roi: float):
        """Print monthly ROI table"""
        console = Console()
        console.print("\nMonthly ROI (based on monthly performance):")
        table = Table()
        table.add_column("Month", style="cyan")
        table.add_column("Fixed Fraction ROI", justify="right", style="magenta")
        table.add_column("Kelly ROI", justify="right", style="green")

        for month in sorted(fixed_monthly_roi.keys()):
            table.add_row(month, f"{fixed_monthly_roi[month]:.2f}%", f"{kelly_monthly_roi[month]:.2f}%")

        table.add_row("Total", f"{fixed_total_roi:.2f}%", f"{kelly_total_roi:.2f}%")
        console.print(table)

    def _print_betting_summary(self, total_fights: int, confident_predictions: int,
                               correct_confident_predictions: int, fixed_total_bets: int,
                               fixed_correct_bets: int, fixed_final_bankroll: float,
                               fixed_total_volume: float, kelly_final_bankroll: float,
                               kelly_total_volume: float, kelly_correct_bets: int,
                               kelly_total_bets: int, earliest_fight_date: str,
                               fixed_monthly_profit: Dict, kelly_monthly_profit: Dict):
        """Print comprehensive betting results summary."""
        console = Console()
        cfg = self.config

        # Calculate metrics
        conf_acc = correct_confident_predictions / confident_predictions if confident_predictions > 0 else 0
        fixed_acc = fixed_correct_bets / fixed_total_bets if fixed_total_bets > 0 else 0
        kelly_acc = kelly_correct_bets / kelly_total_bets if kelly_total_bets > 0 else 0
        fixed_profit = sum(fixed_monthly_profit.values())
        kelly_profit = sum(kelly_monthly_profit.values())
        fixed_roi = (fixed_profit / cfg.initial_bankroll) * 100
        kelly_roi = (kelly_profit / cfg.initial_bankroll) * 100
        avg_fixed = fixed_total_volume / fixed_total_bets if fixed_total_bets > 0 else 0
        avg_kelly = kelly_total_volume / kelly_total_bets if kelly_total_bets > 0 else 0

        console.print(Panel(f"Threshold: {cfg.manual_threshold:.4f}\nKelly ROI: {kelly_roi:.2f}%", title="Optimal Parameters"))

        # Results table
        table = Table(title=f"Betting Results ({cfg.manual_threshold:.0%} confidence)")
        table.add_column("Metric", style="cyan")
        table.add_column("Fixed", justify="right", style="magenta")
        table.add_column("Kelly", justify="right", style="green")
        table.add_row("Total fights", str(total_fights), str(total_fights))
        table.add_row("Predictions", str(confident_predictions), str(confident_predictions))
        table.add_row("Correct", str(correct_confident_predictions), str(correct_confident_predictions))
        table.add_row("Bets", str(fixed_total_bets), str(kelly_total_bets))
        table.add_row("Wins", str(fixed_correct_bets), str(kelly_correct_bets))
        table.add_row("Bet Accuracy", f"{fixed_acc:.2%}", f"{kelly_acc:.2%}")
        table.add_row("Pred Accuracy", f"{conf_acc:.2%}", f"{conf_acc:.2%}")
        console.print(table)

        # Panels
        fixed_panel = Panel(
            f"Initial: ${cfg.initial_bankroll:.2f}\nFinal: ${fixed_final_bankroll:.2f}\n"
            f"Volume: ${fixed_total_volume:.2f}\nProfit: ${fixed_profit:.2f}\nROI: {fixed_roi:.2f}%\n"
            f"Fraction: {cfg.fixed_bet_fraction:.3f}\nAvg Bet: ${avg_fixed:.2f}",
            title="Fixed Fraction"
        )

        kelly_panel = Panel(
            f"Initial: ${cfg.initial_bankroll:.2f}\nFinal: ${kelly_final_bankroll:.2f}\n"
            f"Volume: ${kelly_total_volume:.2f}\nProfit: ${kelly_profit:.2f}\nROI: {kelly_roi:.2f}%\n"
            f"Fraction: {cfg.kelly_fraction:.3f}\nAvg Bet: ${avg_kelly:.2f}",
            title="Kelly Criterion"
        )
        console.print(Columns([fixed_panel, kelly_panel]))