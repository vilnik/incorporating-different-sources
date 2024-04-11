import os
import pandas as pd
import data_handling
import portfolio_evaluation as evaluation
from dotenv import load_dotenv
import logging
import portfolio_calculations
import portfolio_specs

load_dotenv()
logging_level = os.environ.get("LOGGING_LEVEL", logging.INFO)
logging.basicConfig(level=logging_level)
logger = logging.getLogger(__name__)

# Get the absolute path of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Get the parent directory of the script's directory
parent_dir = os.path.dirname(script_dir)

# Construct the path to the results directory
results_dir = os.path.join(parent_dir, 'results')

os.makedirs(results_dir, exist_ok=True)

def main():
    # Dates as in the paper "Incorporating different sources of information for Bayesian optimal portfolio selection"
    str_start_date = "2007-01-01"
    str_end_date = "2023-06-30"
    #str_start_date = "2007-01-01"
    #str_end_date = "2015-04-01"
    #str_start_date = "2015-04-01"
    #str_end_date = "2023-06-30"

    ts_start_date = pd.Timestamp(str_start_date)
    ts_end_date = pd.Timestamp(str_end_date)

    # Portfolio specs
    all_portfolio_specs = portfolio_specs.create_portfolio_specs()

    # Get market data
    market_data = data_handling.get_market_data()
    portfolio_specs_simple_returns = {}
    portfolio_specs_turnover = {}
    portfolio_specs_portfolio_weights_metrics = {}

    # Evaluate portfolio performance
    for portfolio_spec_name, portfolio_spec in all_portfolio_specs.items():
        simple_returns_results_file = os.path.join(results_dir, f'{portfolio_spec_name}_simple_returns_{str_start_date}_{str_end_date}.csv')
        turnover_results_file = os.path.join(results_dir, f'{portfolio_spec_name}_turnover_{str_start_date}_{str_end_date}.csv')
        portfolio_weights_metrics_results_file = os.path.join(results_dir, f'{portfolio_spec_name}_portfolio_weights_metrics_{str_start_date}_{str_end_date}.csv')

        if os.path.exists(simple_returns_results_file) and \
            os.path.exists(turnover_results_file) and \
            os.path.exists(portfolio_weights_metrics_results_file):
            portfolio_specs_simple_returns[portfolio_spec_name] = pd.read_csv(simple_returns_results_file,
                                                                               index_col = 0,
                                                                               parse_dates = True,
                                                                               squeeze = True)
            portfolio_specs_turnover[portfolio_spec_name] = pd.read_csv(turnover_results_file,
                                                                         index_col = 0,
                                                                         parse_dates = True,
                                                                         squeeze = True)
            portfolio_specs_portfolio_weights_metrics[portfolio_spec_name] = pd.read_csv(portfolio_weights_metrics_results_file,
                                                                        index_col = 0,
                                                                        parse_dates = True)
        else:
            portfolio_performance = portfolio_calculations.backtest_portfolio(portfolio_spec,
                                                                               ts_start_date,
                                                                               ts_end_date,
                                                                               market_data)

            # Store the results
            portfolio_specs_simple_returns[portfolio_spec_name] = portfolio_performance["portfolio_simple_returns_series"]
            portfolio_specs_turnover[portfolio_spec_name] = portfolio_performance["portfolio_turnover_series"]
            portfolio_specs_portfolio_weights_metrics[portfolio_spec_name] = portfolio_performance["portfolio_weights_metrics_df"]

            # Save to CSV
            portfolio_performance["portfolio_simple_returns_series"].to_csv(simple_returns_results_file, header=True)
            portfolio_performance["portfolio_turnover_series"].to_csv(turnover_results_file, header=True)
            portfolio_performance["portfolio_weights_metrics_df"].to_csv(portfolio_weights_metrics_results_file, header = True)

    # Portfolio metrics
    evaluation.full_evaluation(portfolio_specs_simple_returns,
                            portfolio_specs_turnover,
                            portfolio_specs_portfolio_weights_metrics,
                            market_data["sp500_simple_returns_df"],
                            market_data["risk_free_rate_df"],
                            market_data["vix_prices_df"],
                            market_data["epu_prices_df"],
                            f"{str_start_date}_{str_end_date}")

if __name__ == "__main__":
    main()
