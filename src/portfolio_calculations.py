import os
import math
import pandas as pd
from pypfopt import EfficientFrontier
from pypfopt import risk_models
from pypfopt import expected_returns
from pypfopt import black_litterman
from pypfopt.black_litterman import BlackLittermanModel
from datetime import timedelta
from scipy.stats import gamma
import numpy as np
import data_handling
from dotenv import load_dotenv
import logging
import csv

# Logging setup
load_dotenv()
logging_level = os.environ.get("LOGGING_LEVEL", logging.INFO)
logging.basicConfig(level=logging_level)
logger = logging.getLogger(__name__)

# Get the absolute path of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Get the parent directory of the script's directory
parent_dir = os.path.dirname(script_dir)

# Construct the path to the conjugate parameters directory
conjugate_parameters_dir = os.path.join(parent_dir, 'parameters')

# Create the directory if it doesn't exist
os.makedirs(conjugate_parameters_dir, exist_ok=True)

def time_periods_per_year(portfolio_spec):
    if portfolio_spec["rebalancing_frequency"] == "daily":
        frequency = 252
    elif portfolio_spec["rebalancing_frequency"] == "weekly":
        frequency = 52
    elif portfolio_spec["rebalancing_frequency"] == "monthly":
        frequency = 12
    else:
        logger.error(f"Unknown rebalancing frequency")
        raise ValueError(f"Unknown rebalancing frequency")

    return frequency

def save_dict_as_csv(data_dict, csv_file):
    with open(csv_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for key, value in data_dict.items():
            writer.writerow([key, value])

def read_dict_from_csv(csv_file):
    data_dict = {}
    with open(csv_file, 'r') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            key, value = row
            key = key.strip().replace('"', '')
            if '(' in value and ')' in value:
                values = [v.strip().replace("'", "") for v in value.strip('()').split(',')]
                values = [int(v) if v.isdigit() or (v.startswith('-') and v[1:].isdigit()) else v for v in values]
                data_dict[key] = tuple(values)
            elif value.replace('.', '', 1).isdigit():
                data_dict[key] = float(value)
            else:
                data_dict[key] = value.replace('"', '').strip()
    return data_dict

def calculate_simple_returns_from_prices(stock_prices_df):
    logger.info(f"Calculating simple returns.")

    # Calculate the percentage change for each stock
    stock_simple_returns_df = stock_prices_df.pct_change()

    # Drop NaN values, which occur for the first data point
    stock_simple_returns_df.dropna(inplace=True)

    return stock_simple_returns_df


def calculate_log_returns_from_prices(stock_prices_df):
    logger.info(f"Calculating log returns.")

    # Calculate the log returns for each stock
    stock_log_returns_df = np.log(stock_prices_df / stock_prices_df.shift(1))

    # Drop NaN values, which occur for the first data point
    stock_log_returns_df.dropna(inplace=True)

    return stock_log_returns_df

def calculate_excess_log_returns_from_prices(portfolio_spec,
                                             stock_prices_df,
                                             risk_free_rate_df):
    logger.info(f"Calculating excess log returns.")

    if portfolio_spec["rebalancing_frequency"] == "daily":
        days_between_rebalancing = 1
    elif portfolio_spec["rebalancing_frequency"] == "weekly":
        days_between_rebalancing = 5
    elif portfolio_spec["rebalancing_frequency"] == "monthly":
        days_between_rebalancing = 21
    else:
        # Log an error and raise an exception if the rebalancing frequency is unknown
        logger.error("Unknown rebalancing frequency.")
        raise RuntimeError("Unknown rebalancing frequency.")

    # Calculate the log returns for each stock
    stock_log_returns_df = np.log(stock_prices_df / stock_prices_df.shift(1))

    # Adjust risk-free rate for rebalancing frequency
    risk_free_rate_adjusted = (1 + risk_free_rate_df) ** (days_between_rebalancing / 252) - 1

    # Resample and interpolate risk-free rates to match stock returns' dates
    risk_free_rate_resampled = risk_free_rate_adjusted.reindex(stock_log_returns_df.index, method='ffill')

    # Calculate the excess log returns
    stock_excess_log_returns_df = stock_log_returns_df - risk_free_rate_resampled.values

    # Drop NaN values, which occur for the first data point
    stock_excess_log_returns_df.dropna(inplace=True)

    return stock_excess_log_returns_df

def calculate_portfolio_variance(portfolio_comp_df, 
                                 covariance_matrix_df):
    logger.info(f"Calculating portfolio variance.")

    # Sort the portfolio DataFrame by index (stock symbols)
    sorted_portfolio_comp_df = portfolio_comp_df.sort_index()
    sorted_weights_np = sorted_portfolio_comp_df['Weight'].to_numpy()

    # Sort the covariance DataFrame by stock symbols and convert to a numpy array
    sorted_keys = sorted_portfolio_comp_df.index
    sorted_covariance_matrix_df = covariance_matrix_df.loc[sorted_keys, sorted_keys]
    sorted_covariance_matrix_np = sorted_covariance_matrix_df.to_numpy()

    # Compute the portfolio variance as w^T * S * w
    portfolio_variance = np.dot(sorted_weights_np.T, np.dot(sorted_covariance_matrix_np, sorted_weights_np))
    # Same as portfolio_comp_df["Weight"].T.dot(covariance_matrix_df.dot(portfolio_comp_df["Weight"]))

    return portfolio_variance

def calculate_rolling_window_frequency_adjusted(portfolio_spec):
    logger.info(f"Calculating rolling window frequency adjusted.")

    # Check the rebalancing frequency specified in portfolio_spec
    if portfolio_spec["rebalancing_frequency"] == "daily":
        rolling_window_frequency_adjusted = portfolio_spec["rolling_window_days"]
    elif portfolio_spec["rebalancing_frequency"] == "weekly":
        rolling_window_frequency_adjusted = math.floor(portfolio_spec["rolling_window_days"] / 5)
    elif portfolio_spec["rebalancing_frequency"] == "monthly":
        rolling_window_frequency_adjusted = math.floor(portfolio_spec["rolling_window_days"] / 21)
    else:
        # Log an error and raise an exception if the rebalancing frequency is unknown
        logger.error("Unknown rebalancing frequency.")
        raise RuntimeError("Unknown rebalancing frequency.")

    return rolling_window_frequency_adjusted


def daily_prices_to_rebalancing_frequency_and_window(portfolio_spec,
                                                     trading_date_ts,
                                                     k_stock_prices_df):
    logger.info(f"Adjusting daily prices to rebalancing frequency and rolling window.")

    # Calculate the rolling window size based on the portfolio's rebalancing frequency
    rolling_window_frequency_adjusted = calculate_rolling_window_frequency_adjusted(portfolio_spec)

    # Adjust the stock prices DataFrame based on the rebalancing frequency
    if portfolio_spec["rebalancing_frequency"] == "weekly":
        k_stock_prices_df_frequency_adjusted = k_stock_prices_df.iloc[::-1][::5][::-1]
    elif portfolio_spec["rebalancing_frequency"] == "monthly":
        k_stock_prices_df_frequency_adjusted = k_stock_prices_df.iloc[::-1][::21][::-1]
    else:
        logger.error("Unknown rebalancing frequency.")
        raise RuntimeError("Unknown rebalancing frequency.")

    # Find the position of the current trading date in the adjusted DataFrame
    position_current_date = k_stock_prices_df_frequency_adjusted.index.get_loc(trading_date_ts)

    # Calculate the start position for the rolling window
    start_position = position_current_date - (rolling_window_frequency_adjusted - 1)

    # Check for invalid start position
    if start_position < 0:
        logger.error(f"Start position is smaller than 0.")
        raise ValueError(f"Start position is smaller than 0.")

    # Slice the DataFrame to only include data within the rolling window
    k_stock_prices_frequency_and_window_adjusted_df = k_stock_prices_df_frequency_adjusted.iloc[
                                                      start_position:position_current_date + 1]

    return k_stock_prices_frequency_and_window_adjusted_df


def calculate_canonical_statistics_T(portfolio_spec,
                                     trading_date_ts,
                                     k_stock_prices_df,
                                     risk_free_rate_df):
    logger.info(f"Calculating canonical statistics T.")

    # Adjust the stock prices DataFrame based on the portfolio's rebalancing frequency and rolling window
    k_stock_prices_frequency_and_window_adjusted_df = daily_prices_to_rebalancing_frequency_and_window(
        portfolio_spec, trading_date_ts, k_stock_prices_df
    )

    # Calculate the excess log returns for the adjusted stock prices
    k_stock_excess_log_returns_frequency_and_window_adjusted = calculate_excess_log_returns_from_prices(
        portfolio_spec, k_stock_prices_frequency_and_window_adjusted_df, risk_free_rate_df
    )

    # Calculate Canonical Statistics T using DataFrame's dot product method
    canonical_statistics_T_df = k_stock_excess_log_returns_frequency_and_window_adjusted.T.dot(
        k_stock_excess_log_returns_frequency_and_window_adjusted
    )

    return canonical_statistics_T_df

def calculate_canonical_statistics_t(portfolio_spec,
                                     trading_date_ts,
                                     k_stock_prices_df,
                                     risk_free_rate_df):
    logger.info(f"Calculating canonical statistics t.")

    # Adjust the stock prices DataFrame based on the portfolio's rebalancing frequency and rolling window
    k_stock_prices_frequency_and_window_adjusted_df = daily_prices_to_rebalancing_frequency_and_window(
        portfolio_spec, trading_date_ts, k_stock_prices_df
    )

    # Calculate the excess log returns for the adjusted stock prices
    k_stock_excess_log_returns_frequency_and_window_adjusted = calculate_excess_log_returns_from_prices(
        portfolio_spec, k_stock_prices_frequency_and_window_adjusted_df, risk_free_rate_df)

    # Calculate Canonical Statistics t using DataFrame's sum method
    canonical_statistics_t_df = k_stock_excess_log_returns_frequency_and_window_adjusted.sum(axis=0).to_frame()

    return canonical_statistics_t_df

def calculate_conjugate_prior_n(portfolio_spec,
                                trading_date_ts,
                                vix_prices_df,
                                risk_free_rate_df):
    logger.info(f"Calculating conjugate prior n.")

    if portfolio_spec["prior_n"] == "vix_scaling":
        # Calculate the average VIX price over the specified rolling window
        average_vix_price = vix_prices_df[-portfolio_spec["rolling_window_days"]:].mean().item()

        # Get the current VIX price for the trading date
        current_vix_price = vix_prices_df.loc[trading_date_ts].item()

        # Calculate the VIX price fraction based on current and average VIX prices
        if current_vix_price > average_vix_price:
            vix_price_fraction = (current_vix_price / average_vix_price)**portfolio_spec["h_l"][0]
        else:
            vix_price_fraction = (average_vix_price / current_vix_price)**portfolio_spec["h_l"][1]

        # Calculate the rolling window adjusted to the portfolio's rebalancing frequency
        rolling_window_frequency_adjusted = calculate_rolling_window_frequency_adjusted(portfolio_spec)

        conjugate_prior_n = rolling_window_frequency_adjusted * vix_price_fraction
    elif portfolio_spec["prior_n"] == "certain":
        conjugate_prior_n = 1e30
    elif isinstance(portfolio_spec["prior_n"], int):
        conjugate_prior_n = portfolio_spec["prior_n"]
    else:
        logger.error(f"Unknown conjugate prior n specification.")
        raise ValueError(f"Unknown conjugate prior n specification.")

    return conjugate_prior_n

def calculate_conjugate_posterior_n(portfolio_spec,
                                    trading_date_ts,
                                    vix_prices_df,
                                    risk_free_rate_df,
                                    conjugate_prior_n=None):
    logger.info(f"Calculating conjugate posterior n.")

    # If conjugate_prior_n is not provided, calculate it
    if conjugate_prior_n is None:
        conjugate_prior_n = calculate_conjugate_prior_n(portfolio_spec,
                                                        trading_date_ts,
                                                        vix_prices_df,
                                                        risk_free_rate_df)

    # Calculate the rolling window adjusted for the portfolio's rebalancing frequency
    rolling_window_frequency_adjusted = calculate_rolling_window_frequency_adjusted(portfolio_spec)

    # Return the sum of the conjugate prior 'n' and the adjusted rolling window
    return conjugate_prior_n + rolling_window_frequency_adjusted


def calculate_conjugate_prior_S(portfolio_spec,
                                trading_date_ts,
                                k_stock_intraday_prices_df,
                                vix_prices_df,
                                risk_free_rate_df,
                                conjugate_prior_n=None):
    logger.info(f"Calculating conjugate prior S.")

    # If conjugate_prior_n is not provided, calculate it
    if conjugate_prior_n is None:
        conjugate_prior_n = calculate_conjugate_prior_n(portfolio_spec,
                                                        trading_date_ts,
                                                        vix_prices_df,
                                                        risk_free_rate_df)

    # Calculate log returns for the last period of intraday prices
    if portfolio_spec["rebalancing_frequency"] == "daily":
        days_between_rebalancing = 1
    elif portfolio_spec["rebalancing_frequency"] == "weekly":
        days_between_rebalancing = 7
    elif portfolio_spec["rebalancing_frequency"] == "monthly":
        days_between_rebalancing = 31
    else:
        # Log an error and raise an exception if the rebalancing frequency is unknown
        logger.error("Unknown rebalancing frequency.")
        raise RuntimeError("Unknown rebalancing frequency.")

    hf_start_date = trading_date_ts - pd.Timedelta(days=days_between_rebalancing)
    filtered_intraday_prices_df = k_stock_intraday_prices_df[(k_stock_intraday_prices_df.index > (hf_start_date + pd.Timedelta(days=1))) &
                                                            (k_stock_intraday_prices_df.index <= (trading_date_ts + pd.Timedelta(days=1)))]
    k_stock_intraday_log_returns_last_week = np.log(filtered_intraday_prices_df / filtered_intraday_prices_df.shift(1)).dropna()

    # Compute the covariance matrix of the log returns, scaled by the number of observations
    k_stock_intraday_cov_last_week = k_stock_intraday_log_returns_last_week.cov() * len(
        k_stock_intraday_log_returns_last_week)

    # Return the scaled covariance matrix
    return conjugate_prior_n * k_stock_intraday_cov_last_week


def calculate_conjugate_posterior_S(portfolio_spec,
                                    trading_date_ts,
                                    k_stock_prices_df,
                                    k_stock_intraday_prices_df,
                                    vix_prices_df,
                                    risk_free_rate_df,
                                    conjugate_prior_S_df=None):
    logger.info(f"Calculating conjugate posterior S.")

    # If conjugate_prior_S_df is not provided, calculate it
    if conjugate_prior_S_df is None:
        conjugate_prior_S_df = calculate_conjugate_prior_S(portfolio_spec,
                                                           trading_date_ts,
                                                           k_stock_intraday_prices_df,
                                                           vix_prices_df,
                                                           risk_free_rate_df)

    # Calculate the Canonical Statistics T matrix
    canonical_statistics_T_df = calculate_canonical_statistics_T(portfolio_spec,
                                                              trading_date_ts,
                                                              k_stock_prices_df,
                                                              risk_free_rate_df)

    # Return the sum of conjugate_prior_S_df and canonical_statistics_T
    return conjugate_prior_S_df + canonical_statistics_T_df


def calculate_conjugate_prior_w(portfolio_spec,
                                trading_date_ts,
                                k_stock_prices_df,
                                k_stock_market_caps_df,
                                vix_prices_df,
                                risk_free_rate_df):
    logger.info(f"Calculating conjugate prior w.")

    # Count the number of stocks in the portfolio
    num_stocks = len(k_stock_prices_df.columns)

    # Calculate the average VIX price over the specified rolling window
    average_vix_price = vix_prices_df[-portfolio_spec["rolling_window_days"]:].mean().item()

    # Get the current VIX price for the trading date
    current_vix_price = vix_prices_df.loc[trading_date_ts].item()

    # Calculate the VIX price fraction based on current and average VIX prices
    if current_vix_price > average_vix_price:
        prior_weights = portfolio_spec["prior_weights_h_l"][0]
    else:
        prior_weights = portfolio_spec["prior_weights_h_l"][1]

    # Initialize equal weights for each stock
    if prior_weights == "empty":
        portfolio_comp_df = pd.DataFrame({
            'Weight': [0] * num_stocks
        }, index=k_stock_prices_df.columns)

        portfolio_comp_df.index.name = 'Stock'
    elif prior_weights == "value_weighted":
        portfolio_comp_df = calculate_value_weighted_portfolio(portfolio_spec,
                                                               trading_date_ts,
                                                               k_stock_market_caps_df)
    elif prior_weights == "equally_weighted":
        portfolio_comp_df = calculate_equally_weighted_portfolio(portfolio_spec,
                                                                 k_stock_prices_df)
    else:
        logger.error(f"Unknown conjugate portfolio prior weights.")
        raise ValueError("Unknown conjugate portfolio prior weights.")

    return portfolio_comp_df

def calculate_conjugate_c(portfolio_spec,
                          trading_date_ts,
                          k_stock_prices_df,
                          k_stock_market_caps_df,
                          k_stock_intraday_prices_df,
                          vix_prices_df,
                          risk_free_rate_df,
                          conjugate_prior_n=None,
                          conjugate_prior_S_df=None,
                          conjugate_prior_w_df=None):
    logger.info(f"Calculating conjugate c.")

    # Calculate 'conjugate_prior_n' if not provided
    if conjugate_prior_n is None:
        conjugate_prior_n = calculate_conjugate_prior_n(portfolio_spec,
                                                        trading_date_ts,
                                                        vix_prices_df,
                                                         risk_free_rate_df)

    # Calculate 'conjugate_prior_S_df' if not provided
    if conjugate_prior_S_df is None:
        conjugate_prior_S_df = calculate_conjugate_prior_S(portfolio_spec,
                                                           trading_date_ts,
                                                           k_stock_intraday_prices_df,
                                                           vix_prices_df,
                                                           risk_free_rate_df)

    # Calculate 'conjugate_prior_w_df' if not provided
    if conjugate_prior_w_df is None:
        conjugate_prior_w_df = calculate_conjugate_prior_w(portfolio_spec,
                                                            trading_date_ts,
                                                            k_stock_prices_df,
                                                            k_stock_market_caps_df,
                                                            vix_prices_df,
                                                            risk_free_rate_df)

    # Compute 'conjugate_c' using the provided formulas
    conjugate_c = (2 * conjugate_prior_n) / ((conjugate_prior_n + portfolio_spec["portfolio_size"] + 2) +
                                      ((conjugate_prior_n + portfolio_spec["portfolio_size"] + 2) ** 2 +
                                      4 * conjugate_prior_n * calculate_portfolio_variance(conjugate_prior_w_df,
                                                                                           conjugate_prior_S_df)) ** (1 / 2))

    return conjugate_c

def calculate_conjugate_posterior_w(portfolio_spec,
                                    trading_date_ts,
                                    k_stock_prices_df,
                                    k_stock_market_caps_df,
                                    k_stock_intraday_prices_df,
                                    vix_prices_df,
                                    risk_free_rate_df,
                                    conjugate_c=None,
                                    conjugate_prior_w_df=None,
                                    conjugate_prior_S_df=None,
                                    conjugate_posterior_S_df=None):
    logger.info(f"Calculating conjugate posterior w.")

    # Calculate 'conjugate_c' if not provided
    if conjugate_c is None:
        conjugate_c = calculate_conjugate_c(portfolio_spec,
                                            trading_date_ts,
                                            k_stock_prices_df,
                                            k_stock_market_caps_df,
                                            k_stock_intraday_prices_df,
                                            vix_prices_df,
                                            risk_free_rate_df)

    # Calculate 'conjugate_prior_w_df' if not provided
    if conjugate_prior_w_df is None:
        conjugate_prior_w_df = calculate_conjugate_prior_w(portfolio_spec,
                                                            trading_date_ts,
                                                            k_stock_prices_df,
                                                            k_stock_market_caps_df,
                                                            vix_prices_df,
                                                            risk_free_rate_df)

    # Calculate 'conjugate_posterior_S_df' if not provided
    if conjugate_posterior_S_df is None:
        if conjugate_prior_S_df is None:
            conjugate_prior_S_df = calculate_conjugate_prior_S(portfolio_spec,
                                                               trading_date_ts,
                                                               k_stock_intraday_prices_df,
                                                               vix_prices_df,
                                                               risk_free_rate_df)

        conjugate_posterior_S_df = calculate_conjugate_posterior_S(portfolio_spec,
                                                                   trading_date_ts,
                                                                   k_stock_prices_df,
                                                                   k_stock_intraday_prices_df,
                                                                   vix_prices_df,
                                                                   risk_free_rate_df,
                                                                   conjugate_prior_S_df=conjugate_prior_S_df)

    # Calculate 'canonical_statistics_t_df'
    canonical_statistics_t_df = calculate_canonical_statistics_t(portfolio_spec,
                                                              trading_date_ts,
                                                              k_stock_prices_df,
                                                              risk_free_rate_df)

    # Compute 'conjugate_posterior_w_df' based on the various matrices and vectors
    conjugate_posterior_S_inv_df = pd.DataFrame(np.linalg.inv(conjugate_posterior_S_df.values),
                                                index = conjugate_posterior_S_df.columns,
                                                columns = conjugate_posterior_S_df.index)

    conjugate_posterior_w_series = conjugate_posterior_S_inv_df.dot(conjugate_c * conjugate_prior_S_df.dot(conjugate_prior_w_df['Weight']) + canonical_statistics_t_df.squeeze())
    conjugate_posterior_w_df = pd.DataFrame(conjugate_posterior_w_series, columns=['Weight'])

    if conjugate_posterior_w_df.isna().any().any():
        logger.error(f"conjugate_posterior_w_df contains NaN values.")
        raise ValueError(f"conjugate_posterior_w_df contains NaN values.")

    return conjugate_posterior_w_df


def calculate_mean_conjugate_posterior_nu(portfolio_spec,
                                          trading_date_ts,
                                          k_stock_prices_df,
                                          k_stock_market_caps_df,
                                          k_stock_intraday_prices_df,
                                          vix_prices_df,
                                          risk_free_rate_df,
                                          conjugate_c=None,
                                          conjugate_prior_n=None,
                                          conjugate_posterior_n=None,
                                          conjugate_prior_S_df=None,
                                          conjugate_posterior_S_df=None,
                                          conjugate_prior_w_df=None,
                                          conjugate_posterior_w_df=None):
    # Log information about the calculation
    logger.info(f"Calculating mean conjugate posterior nu")

    # Calculate 'conjugate_c' if not provided
    if conjugate_c is None:
        conjugate_c = calculate_conjugate_c(portfolio_spec,
                                            trading_date_ts,
                                            k_stock_prices_df,
                                            k_stock_market_caps_df,
                                            k_stock_intraday_prices_df,
                                            vix_prices_df,
                                            risk_free_rate_df)

    # Calculate 'conjugate_posterior_n' if not provided
    if conjugate_posterior_n is None:
        if conjugate_prior_n is None:
            conjugate_prior_n = calculate_conjugate_prior_n(portfolio_spec,
                                                            trading_date_ts,
                                                            vix_prices_df,
                                                            risk_free_rate_df)
        conjugate_posterior_n = calculate_conjugate_posterior_n(portfolio_spec,
                                                                trading_date_ts,
                                                                vix_prices_df,
                                                                risk_free_rate_df,
                                                                conjugate_prior_n=conjugate_prior_n)

    # Calculate 'conjugate_posterior_S_df' if not provided
    if conjugate_posterior_S_df is None:
        if conjugate_prior_S_df is None:
            conjugate_prior_S_df = calculate_conjugate_prior_S(portfolio_spec,
                                                               trading_date_ts,
                                                               k_stock_intraday_prices_df,
                                                               vix_prices_df,
                                                               risk_free_rate_df)
        conjugate_posterior_S_df = calculate_conjugate_posterior_S(portfolio_spec,
                                                                   trading_date_ts,
                                                                   k_stock_prices_df,
                                                                   k_stock_intraday_prices_df,
                                                                   vix_prices_df,
                                                                   risk_free_rate_df,
                                                                   conjugate_prior_S_df=conjugate_prior_S_df)

    # Calculate 'conjugate_posterior_w_df' if not provided
    if conjugate_posterior_w_df is None:
        if conjugate_prior_w_df is None:
            conjugate_prior_w_df = calculate_conjugate_prior_w(portfolio_spec,
                                                                trading_date_ts,
                                                                k_stock_prices_df,
                                                                k_stock_market_caps_df,
                                                                vix_prices_df,
                                                               risk_free_rate_df)
        conjugate_posterior_w_df = calculate_conjugate_posterior_w(portfolio_spec,
                                                                trading_date_ts,
                                                                k_stock_prices_df,
                                                                k_stock_market_caps_df,
                                                                k_stock_intraday_prices_df,
                                                                vix_prices_df,
                                                                risk_free_rate_df,
                                                                conjugate_c=conjugate_c,
                                                                conjugate_prior_w_df=conjugate_prior_w_df,
                                                                conjugate_prior_S_df=conjugate_prior_S_df,
                                                                conjugate_posterior_S_df=conjugate_posterior_S_df)

    # Compute and return the mean conjugate posterior nu
    mean_conjugate_posterior_nu_df = (conjugate_posterior_n + portfolio_spec["portfolio_size"] + 2) * \
                                  conjugate_posterior_w_df / (conjugate_posterior_n -
                                                           calculate_portfolio_variance(conjugate_posterior_w_df,
                                                                                        conjugate_posterior_S_df))

    return mean_conjugate_posterior_nu_df


def calculate_mean_jeffreys_posterior_nu(portfolio_spec,
                                          trading_date_ts,
                                          k_stock_prices_df,
                                          risk_free_rate_df):
    # Log information about the calculation
    logger.info(f"Calculating mean Jeffreys posterior nu")

    rolling_window_frequency_adjusted = calculate_rolling_window_frequency_adjusted(portfolio_spec)

    # Calculate 'canonical_statistics_t'
    canonical_statistics_t_df = calculate_canonical_statistics_t(portfolio_spec,
                                                              trading_date_ts,
                                                              k_stock_prices_df,
                                                              risk_free_rate_df)
    # Calculate 'canonical_statistics_T'
    canonical_statistics_T_df = calculate_canonical_statistics_T(portfolio_spec,
                                                              trading_date_ts,
                                                              k_stock_prices_df,
                                                              risk_free_rate_df)


    # Compute and return the mean Jeffreys posterior nu
    jeffreys_scaled_covariance_matrix_df = (canonical_statistics_T_df - 1 / rolling_window_frequency_adjusted * \
                                    canonical_statistics_t_df.dot(canonical_statistics_t_df.T))
    jeffreys_scaled_covariance_matrix_inv_df = pd.DataFrame(np.linalg.inv(jeffreys_scaled_covariance_matrix_df.values),
                                     index=jeffreys_scaled_covariance_matrix_df.columns,
                                     columns=jeffreys_scaled_covariance_matrix_df.index)

    mean_jeffreys_posterior_nu_df = jeffreys_scaled_covariance_matrix_inv_df.dot(canonical_statistics_t_df)
    mean_jeffreys_posterior_nu_df.columns = ['Weight']
    return mean_jeffreys_posterior_nu_df


def get_k_largest_stocks_market_caps(stock_market_caps_df,
                                     stock_prices_df,
                                     stock_intraday_prices_df,
                                     trading_date_ts,
                                     portfolio_size,
                                     rolling_window_days,
                                     rebalancing_frequency):
    # Get S&P 500 components for the current date
    tickers_list = data_handling.extract_unique_tickers(trading_date_ts,
                                          trading_date_ts)

    # Identify tickers that are present in stock_market_caps_df.columns
    present_tickers = [ticker for ticker in tickers_list if ticker in stock_market_caps_df.columns]
    missing_fraction = (len(tickers_list) - len(present_tickers)) / len(tickers_list)
    logger.info(f"Fraction of tickers missing from stock_market_caps_df: {missing_fraction:.2%}")

    # Days since last rebalancing
    if rebalancing_frequency == "daily":
        days_between_rebalancing = 1
    elif rebalancing_frequency == "weekly":
        days_between_rebalancing = 7
    elif rebalancing_frequency == "monthly":
        days_between_rebalancing = 31
    else:
        # Log an error and raise an exception if the rebalancing frequency is unknown
        logger.error("Unknown rebalancing frequency.")
        raise RuntimeError("Unknown rebalancing frequency.")

    eligible_stocks = [
        stock for stock in stock_prices_df.columns
        if (
                stock in stock_market_caps_df.columns and
                stock in stock_intraday_prices_df.columns and
                stock_prices_df.loc[trading_date_ts, stock] is not None and
                stock_prices_df[stock].loc[:trading_date_ts].tail(rolling_window_days).notna().all() and
                stock_intraday_prices_df[stock].loc[(trading_date_ts - timedelta(days=days_between_rebalancing)):(trading_date_ts + timedelta(days = 1))].notna().any()
        )
    ]

    # From these available stocks, get the portfolio_size largest based on market caps for the current date
    if trading_date_ts in stock_market_caps_df.index:
        daily_market_caps = stock_market_caps_df.loc[trading_date_ts, eligible_stocks].dropna()
        k_stock_market_caps_df = daily_market_caps.nlargest(portfolio_size)
        return k_stock_market_caps_df
    else:
        logger.error(f"The trading date {trading_date_ts} does not exist in the market capitalizations data.")
        raise ValueError(f"The trading date {trading_date_ts} does not exist in the market capitalizations data.")


def calculate_equally_weighted_portfolio(portfolio_spec,
                                         k_stock_prices_df):
    # Logging the calculation step
    logger.info(f"Calculating equally weighted portfolio")

    # Determine the number of stocks in the portfolio
    num_stocks = portfolio_spec["portfolio_size"]

    # Assign equal weight to each stock and create the resulting dataframe
    portfolio_comp_df = pd.DataFrame({
        'Weight': [1 / num_stocks] * num_stocks
    }, index=k_stock_prices_df.columns)

    # Rename the index to 'Stock'
    portfolio_comp_df.index.name = 'Stock'

    return portfolio_comp_df

def calculate_value_weighted_portfolio(portfolio_spec, 
                                       trading_date_ts, 
                                       k_stock_market_caps_df):
    logger.info(f"Calculating market cap portfolio weights.")
    k_stock_market_caps_series = k_stock_market_caps_df.iloc[-1].sort_values(ascending=False)

    # Total market cap of the k largest stocks
    total_market_cap = k_stock_market_caps_series.sum()

    # Calculate value weights
    portfolio_comp_df = pd.DataFrame(k_stock_market_caps_series / total_market_cap)

    # Fix labels
    portfolio_comp_df.index.name = 'Stock'
    portfolio_comp_df.columns = ['Weight']

    return portfolio_comp_df

def calculate_shrinkage_portfolio(portfolio_spec, 
                                  trading_date_ts, 
                                  k_stock_prices_df,
                                  risk_free_rate_df):
    logger.info(f"Calculating shrinkage portfolio weights.")
    k_stock_prices_frequency_and_window_adjusted_df = daily_prices_to_rebalancing_frequency_and_window(
        portfolio_spec,
        trading_date_ts,
        k_stock_prices_df)

    # Mean return
    mean_simple_returns = expected_returns.ema_historical_return(k_stock_prices_frequency_and_window_adjusted_df,
                                                                  frequency = time_periods_per_year(portfolio_spec))

    # Covariance matrix
    covariance_simple_returns = risk_models.CovarianceShrinkage(k_stock_prices_frequency_and_window_adjusted_df,
                                                                frequency = time_periods_per_year(portfolio_spec)).ledoit_wolf()

    # Add risk free asset
    most_recent_risk_free_rate = risk_free_rate_df.asof(trading_date_ts).iloc[0]
    mean_simple_returns_with_risk_free_asset = mean_simple_returns.copy()
    mean_simple_returns_with_risk_free_asset["RISK_FREE"] = most_recent_risk_free_rate

    covariance_simple_returns_with_risk_free_asset = covariance_simple_returns.copy()
    covariance_simple_returns_with_risk_free_asset["RISK_FREE"] = 0
    covariance_simple_returns_with_risk_free_asset.loc["RISK_FREE"] = 0


    ef = EfficientFrontier(mean_simple_returns_with_risk_free_asset, covariance_simple_returns_with_risk_free_asset, weight_bounds=(-100, 100))
    raw_portfolio_comp = ef.max_quadratic_utility(risk_aversion=portfolio_spec["risk_aversion"])

    # Convert cleaned weights to DataFrame
    portfolio_comp_df = pd.DataFrame(list(ef.clean_weights().items()), columns=['Stock', 'Weight'])
    portfolio_comp_df.set_index('Stock', inplace=True)
    portfolio_comp_df = portfolio_comp_df.drop("RISK_FREE")

    return portfolio_comp_df

def calculate_black_litterman_portfolio(portfolio_spec,
                                        trading_date_ts,
                                        k_stock_market_caps_df,
                                        k_stock_prices_df,
                                        sp500_prices_df,
                                        risk_free_rate_df):
    logger.info(f"Calculating Black-Litterman portfolio weights.")
    k_stock_prices_frequency_and_window_adjusted_df = daily_prices_to_rebalancing_frequency_and_window(
        portfolio_spec,
        trading_date_ts,
        k_stock_prices_df)
    k_stock_market_caps_latest_df = k_stock_market_caps_df.iloc[-1].sort_values(ascending=False)

    sp500_prices_df_frequency_and_window_adjusted_df = daily_prices_to_rebalancing_frequency_and_window(
                                                                                                    portfolio_spec,
                                                                                                    trading_date_ts,
                                                                                                    sp500_prices_df)

    # Covariance matrix
    covariance_simple_returns = risk_models.CovarianceShrinkage(
        k_stock_prices_frequency_and_window_adjusted_df, frequency = time_periods_per_year(portfolio_spec)).ledoit_wolf()

    viewdict = {}
    delta = black_litterman.market_implied_risk_aversion(sp500_prices_df_frequency_and_window_adjusted_df.squeeze(),
                                                         frequency = time_periods_per_year(portfolio_spec))
    market_prior = black_litterman.market_implied_prior_returns(k_stock_market_caps_latest_df.squeeze(),
                                                                delta,
                                                                covariance_simple_returns)

    bl = BlackLittermanModel(covariance_simple_returns, pi=market_prior, absolute_views=viewdict)
    bl_mean_simple_returns = bl.bl_returns()
    bl_covariance_simple_returns = bl.bl_cov()

    # Add risk free asset
    most_recent_risk_free_rate = risk_free_rate_df.asof(trading_date_ts).iloc[0]
    bl_mean_simple_returns_with_risk_free_asset = bl_mean_simple_returns.copy()
    bl_mean_simple_returns_with_risk_free_asset["RISK_FREE"] = most_recent_risk_free_rate

    bl_covariance_simple_returns_with_risk_free_asset = bl_covariance_simple_returns.copy()
    bl_covariance_simple_returns_with_risk_free_asset["RISK_FREE"] = 0
    bl_covariance_simple_returns_with_risk_free_asset.loc["RISK_FREE"] = 0

    ef = EfficientFrontier(bl_mean_simple_returns_with_risk_free_asset, bl_covariance_simple_returns_with_risk_free_asset, weight_bounds=(-100, 100))
    raw_portfolio_comp = ef.max_quadratic_utility(risk_aversion=portfolio_spec["risk_aversion"])

    # Convert cleaned weights to DataFrame
    portfolio_comp_df = pd.DataFrame(list(ef.clean_weights().items()), columns=['Stock', 'Weight'])
    portfolio_comp_df.set_index('Stock', inplace=True)
    portfolio_comp_df = portfolio_comp_df.drop("RISK_FREE")

    return portfolio_comp_df

def calculate_conjugate_hf_vix_portfolio(portfolio_spec,
                                         trading_date_ts,
                                         k_stock_market_caps_df,
                                         k_stock_prices_df,
                                         k_stock_intraday_prices_df,
                                         vix_prices_df,
                                         risk_free_rate_df):
    logger.info(f"Calculating conjugate high frequency VIX portfolio weights.")

    mean_conjugate_posterior_nu_df = calculate_mean_conjugate_posterior_nu(portfolio_spec,
                                                                        trading_date_ts,
                                                                        k_stock_prices_df,
                                                                        k_stock_market_caps_df,
                                                                        k_stock_intraday_prices_df,
                                                                        vix_prices_df,
                                                                        risk_free_rate_df)

    return 1 / portfolio_spec["risk_aversion"] * mean_conjugate_posterior_nu_df

def calculate_jeffreys_portfolio(portfolio_spec,
                                 trading_date_ts,
                                 k_stock_prices_df,
                                 risk_free_rate_df):
    logger.info(f"Calculating Jeffreys portfolio weights.")

    mean_jeffreys_posterior_nu_df = calculate_mean_jeffreys_posterior_nu(portfolio_spec,
                                                                        trading_date_ts,
                                                                        k_stock_prices_df,
                                                                        risk_free_rate_df)

    return 1 / portfolio_spec["risk_aversion"] * mean_jeffreys_posterior_nu_df

def calculate_jorion_hyperparameter_portfolio(portfolio_spec,
                                              trading_date_ts,
                                              k_stock_prices_df,
                                              risk_free_rate_df):

    logger.info(f"Calculating Jorion hyperparameter portfolio.")

    # Adjust the stock prices DataFrame based on the portfolio's rebalancing frequency and rolling window
    k_stock_prices_frequency_and_window_adjusted_df = daily_prices_to_rebalancing_frequency_and_window(
        portfolio_spec, trading_date_ts, k_stock_prices_df
    )

    # Calculate the excess log returns for the adjusted stock prices
    k_stock_excess_log_returns_frequency_and_window_adjusted_df = calculate_excess_log_returns_from_prices(
        portfolio_spec, k_stock_prices_frequency_and_window_adjusted_df, risk_free_rate_df
    )

    # Using notations of Bayesian Portfolio Analysis (2010) by Avramov and Zhou
    N = len(k_stock_prices_df.columns)
    T = len(k_stock_excess_log_returns_frequency_and_window_adjusted_df)

    # Sample mean
    mu_hat_df = k_stock_excess_log_returns_frequency_and_window_adjusted_df.mean().to_frame()

    # Sample covariance
    V_hat_df = k_stock_excess_log_returns_frequency_and_window_adjusted_df.cov()

    # Shrinkage
    V_bar_df = T / (T - N - 2) * V_hat_df
    V_bar_inverse_df = pd.DataFrame(np.linalg.inv(V_bar_df.to_numpy()), index=V_bar_df.index, columns=V_bar_df.columns)
    one_N_df = pd.DataFrame(np.ones(N), index=V_bar_inverse_df.index)
    mu_hat_g = (one_N_df.T.dot(V_bar_inverse_df).dot(mu_hat_df) / one_N_df.T.dot(V_bar_inverse_df).dot(one_N_df)).values[0,0]

    mu_hat_difference = mu_hat_df.sub(mu_hat_g * one_N_df.values, axis=0)
    lambda_hat = (N + 2) / (mu_hat_difference.T.dot(V_bar_inverse_df).dot(mu_hat_difference)).values[0, 0]

    v_hat = (N + 2) / ((N + 2) + T * mu_hat_difference.T.dot(V_bar_inverse_df).dot(mu_hat_difference)).values[0, 0]
    V_hat_PJ = (1 + 1 / (T + lambda_hat)) * V_bar_df + lambda_hat / (T * (T + 1 + lambda_hat)) * one_N_df.dot(one_N_df.T) / (one_N_df.T.dot(V_bar_inverse_df).dot(one_N_df)).values[0, 0]
    mu_hat_PJ_df = (1 - v_hat) * mu_hat_df + v_hat * mu_hat_g * one_N_df

    V_hat_PJ_inverse_df = pd.DataFrame(np.linalg.inv(V_hat_PJ.to_numpy()), index=V_hat_PJ.index, columns=V_hat_PJ.columns)

    portfolio_comp_df = 1 / portfolio_spec["risk_aversion"] * V_hat_PJ_inverse_df.dot(mu_hat_PJ_df).reset_index().rename(columns={'index': 'Stock', 0: 'Weight'}).set_index('Stock')

    return portfolio_comp_df

def calculate_hierarchical_portfolio(portfolio_spec,
                                      trading_date_ts,
                                      k_stock_prices_df,
                                      risk_free_rate_df):

    logger.info(f"Calculating hierarchical portfolio.")

    # Adjust the stock prices DataFrame based on the portfolio's rebalancing frequency and rolling window
    k_stock_prices_frequency_and_window_adjusted_df = daily_prices_to_rebalancing_frequency_and_window(
        portfolio_spec, trading_date_ts, k_stock_prices_df
    )

    # Calculate the excess log returns for the adjusted stock prices
    k_stock_excess_log_returns_frequency_and_window_adjusted_df = calculate_excess_log_returns_from_prices(
        portfolio_spec, k_stock_prices_frequency_and_window_adjusted_df, risk_free_rate_df
    )

    n = len(k_stock_excess_log_returns_frequency_and_window_adjusted_df)
    k = len(k_stock_prices_df.columns)
    x_bar_df = k_stock_excess_log_returns_frequency_and_window_adjusted_df.mean().to_frame()
    S_df = k_stock_excess_log_returns_frequency_and_window_adjusted_df.cov()
    S_h_df = pd.DataFrame(np.where(np.eye(k) == 1, 1, 0.5), index=S_df.index[:k], columns=S_df.index[:k])
    one_N_df = pd.DataFrame(np.ones(k), index=x_bar_df.index)
    kappa_h = round(0.1 * n)
    nu_h = k
    weights_b_storage = []
    # Using notations of Incorporating different sources of information for Bayesian optimal portfolio selection (2023) by Bodnar et al.
    for i in range(1000):
        xi_b = np.random.uniform(-1000, 1000)
        eta_b = gamma.rvs(a=1, scale=10)
        a_h = 1 / (n + kappa_h) * (n * x_bar_df + kappa_h * xi_b * one_N_df)
        D_h = (n - 1) * S_df + eta_b * S_h_df + n * x_bar_df.dot(x_bar_df.T) + kappa_h * xi_b**2 * one_N_df.dot(one_N_df.T) - (n + kappa_h) * a_h.dot(a_h.T)
        D_h_inverse = pd.DataFrame(np.linalg.inv(D_h.to_numpy()), index=D_h.index, columns=D_h.columns)
        weights_b_df = 1 / portfolio_spec["risk_aversion"] * (nu_h + n + 1) * (1 - 1 / (nu_h + n - k)) * D_h_inverse.dot(a_h)
        weights_b_storage.append(weights_b_df)

    weights_b_all_df = pd.concat(weights_b_storage, axis=1)
    weights_b_df_mean = weights_b_all_df.mean(axis=1)

    weights_b_df_mean.index.name = 'Stock'
    weights_b_df_mean.name = 'Weight'
    return weights_b_df_mean.to_frame(name='Weight')


def calculate_portfolio_weights(trading_date_ts,
                                portfolio_spec,
                                market_data):
    # Unpack market data
    stock_market_caps_df = market_data["stock_market_caps_df"]
    stock_prices_df = market_data["stock_prices_df"]
    stock_intraday_prices_df = market_data["stock_intraday_prices_df"]
    vix_prices_df = market_data["vix_prices_df"]
    risk_free_rate_df = market_data["risk_free_rate_df"]
    sp500_prices_df = market_data["sp500_prices_df"]

    # Get k largest stocks and market caps at trading_date_ts
    k_stock_market_caps_trading_date_df = get_k_largest_stocks_market_caps(stock_market_caps_df,
                                                              stock_prices_df,
                                                              stock_intraday_prices_df,
                                                              trading_date_ts,
                                                              portfolio_spec["portfolio_size"],
                                                              portfolio_spec["rolling_window_days"],
                                                              portfolio_spec["rebalancing_frequency"])

    # Filter all the data to only include data until current date. Very important!
    # Filter market caps
    k_stock_market_caps_df = stock_market_caps_df[k_stock_market_caps_trading_date_df.index.intersection(stock_market_caps_df.columns)]
    k_stock_market_caps_df = k_stock_market_caps_df.loc[:trading_date_ts]

    # Filter stock prices
    k_stock_prices_df = stock_prices_df[k_stock_market_caps_trading_date_df.index.intersection(stock_prices_df.columns)]
    k_stock_prices_df = k_stock_prices_df.loc[:trading_date_ts]

    # Filter intraday prices
    trading_date_ts_inclusive = pd.Timestamp(trading_date_ts).replace(hour=23, minute=59, second=59)
    k_stock_intraday_prices_df = stock_intraday_prices_df[k_stock_market_caps_trading_date_df.index.intersection(stock_intraday_prices_df.columns)]
    k_stock_intraday_prices_df = k_stock_intraday_prices_df.loc[k_stock_intraday_prices_df.index <= trading_date_ts_inclusive]

    # Filter VIX prices
    vix_prices_df = vix_prices_df.loc[vix_prices_df.index <= trading_date_ts]

    # Filter S&P 500 prices
    sp500_prices_df = sp500_prices_df.loc[sp500_prices_df.index <= trading_date_ts]

    # Check for NA values in the filtered DataFrame
    if k_stock_prices_df.tail(portfolio_spec["rolling_window_days"]).isna().any().any():
        logger.error(f"Found NA values in the filtered stock prices.")
        raise ValueError(f"The filtered stock prices contain NA values.")

    if portfolio_spec["weights_spec"] == "value_weighted":
        portfolio_comp_df = calculate_value_weighted_portfolio(portfolio_spec,
                                                                trading_date_ts,
                                                                k_stock_market_caps_df)

    elif portfolio_spec["weights_spec"] == "equally_weighted":
        portfolio_comp_df = calculate_equally_weighted_portfolio(portfolio_spec,
                                                                k_stock_prices_df)

    elif portfolio_spec["weights_spec"] == "shrinkage":
        portfolio_comp_df = calculate_shrinkage_portfolio(portfolio_spec,
                                                           trading_date_ts,
                                                           k_stock_prices_df,
                                                          risk_free_rate_df)

    elif portfolio_spec["weights_spec"] == "black_litterman":
        portfolio_comp_df = calculate_black_litterman_portfolio(portfolio_spec,
                                                                 trading_date_ts,
                                                                 k_stock_market_caps_df,
                                                                 k_stock_prices_df,
                                                                 sp500_prices_df,
                                                                 risk_free_rate_df)

    elif portfolio_spec["weights_spec"] == "conjugate_hf_vix":
        portfolio_comp_df = calculate_conjugate_hf_vix_portfolio(portfolio_spec,
                                                                  trading_date_ts,
                                                                  k_stock_market_caps_df,
                                                                  k_stock_prices_df,
                                                                  k_stock_intraday_prices_df,
                                                                  vix_prices_df,
                                                                  risk_free_rate_df)

    elif portfolio_spec["weights_spec"] == "jeffreys":
        portfolio_comp_df = calculate_jeffreys_portfolio(portfolio_spec,
                                                          trading_date_ts,
                                                          k_stock_prices_df,
                                                          risk_free_rate_df)

    elif portfolio_spec["weights_spec"] == "jorion_hyper":
        portfolio_comp_df = calculate_jorion_hyperparameter_portfolio(portfolio_spec,
                                                                      trading_date_ts,
                                                                      k_stock_prices_df,
                                                                      risk_free_rate_df)

    elif portfolio_spec["weights_spec"] == "hierarchical":
        portfolio_comp_df = calculate_hierarchical_portfolio(portfolio_spec,
                                                              trading_date_ts,
                                                              k_stock_prices_df,
                                                              risk_free_rate_df)

    else:
        logger.error(f"Unknown weights spec.")
        raise ValueError(f"Unknown weights spec.")

    return portfolio_comp_df

def compute_portfolio_turnover(portfolio_comp_before_df, portfolio_comp_after_df):

    # Merging the old and new weights with a suffix to differentiate them
    portfolio_comp_before_after_df = portfolio_comp_before_df.merge(portfolio_comp_after_df,
                                                how='outer',
                                                left_index=True,
                                                right_index=True,
                                                suffixes=('_before', '_after'))

    # Fill missing values with 0s (for new stocks or those that have been removed)
    portfolio_comp_before_after_df.fillna(0, inplace=True)

    # Calculate absolute difference for each stock and then compute turnover
    portfolio_comp_before_after_df['weight_diff'] = abs(portfolio_comp_before_after_df['Weight_before'] - portfolio_comp_before_after_df['Weight_after'])

    # Calculate turnover corresponding to risk free asset
    risk_free_turnover = abs(portfolio_comp_before_df['Weight'].sum() - portfolio_comp_after_df['Weight'].sum())

    # Calculate total turnover
    turnover = (portfolio_comp_before_after_df['weight_diff'].sum() + risk_free_turnover) / 2

    return turnover

class Portfolio:

    def get_portfolio_simple_returns_series(self):
        return self.portfolio_simple_returns_series

    def get_portfolio_turnover(self):
        return self.portfolio_turnover_series

    def __init__(self,
                 ts_start_date,
                 portfolio_spec):
        self.ts_start_date = ts_start_date
        self.portfolio_spec = portfolio_spec
        self.portfolio_simple_returns_series = pd.Series(dtype = "float64", name = portfolio_spec["display_name"])
        self.portfolio_turnover_series = pd.Series(dtype = "float64", name = portfolio_spec["display_name"])
        self.last_rebalance_date_ts = None

    def update_portfolio(self,
                         trading_date_ts,
                         market_data):

        # Calculate daily portfolio return
        if self.ts_start_date != trading_date_ts:
            # Filter out stocks not in the portfolio
            filtered_stock_simple_returns_series = market_data["stock_simple_returns_df"].loc[trading_date_ts].reindex(self.portfolio_comp_df.index)

            # Multiply returns by weights element-wise and then sum to get the portfolio return
            portfolio_simple_return = (filtered_stock_simple_returns_series * self.portfolio_comp_df['Weight']).sum()

            # Add risk-free return
            risk_free_rate_df = market_data["risk_free_rate_df"]
            most_recent_risk_free_rate = risk_free_rate_df.asof(trading_date_ts).iloc[0]
            risk_free_daily_return = ((most_recent_risk_free_rate + 1) ** (1 / 252) - 1)
            portfolio_simple_return += (1 - self.portfolio_comp_df['Weight'].sum()) * risk_free_daily_return

            self.portfolio_simple_returns_series[trading_date_ts] = portfolio_simple_return

            # Update weight for the risk-free asset
            current_risk_free_weight = 1 - self.portfolio_comp_df['Weight'].sum()
            updated_risk_free_weight = current_risk_free_weight * (1 + risk_free_daily_return)

            # Update weights for the stocks
            self.portfolio_comp_df['Weight'] = (
                        self.portfolio_comp_df['Weight'] * (1 + filtered_stock_simple_returns_series))


            # Update the total invested value by adding the updated risk-free weight
            total_value = self.portfolio_comp_df['Weight'].sum() + updated_risk_free_weight

            # Normalize the weights so they sum up to 1
            self.portfolio_comp_df['Weight'] = self.portfolio_comp_df['Weight'] / total_value

            # Check that weights sum to 1
            if abs((self.portfolio_comp_df['Weight'].values.sum() + updated_risk_free_weight / total_value) - 1) > 1e-5:
                logger.error(f"Weights do not sum to 1.")
                raise ValueError(f"Weights do not sum to 1.")

        if self.last_rebalance_date_ts is None:
            rebalance = True
        elif self.portfolio_spec["rebalancing_frequency"] == "daily":
            rebalance = True
        elif self.portfolio_spec["rebalancing_frequency"] == "weekly":
            rebalance = trading_date_ts.weekday() == 2 or (trading_date_ts - self.last_rebalance_date_ts).days > 7
        elif self.portfolio_spec["rebalancing_frequency"] == "monthly":
            rebalance = trading_date_ts.month != self.last_rebalance_date_ts.month
        else:
            logger.error(f"Unknown rebalancing frequency.")
            raise ValueError(f"Unknown rebalancing frequency.")

        if rebalance:
            if not self.last_rebalance_date_ts is None:
                # Make a copy of the current weights to calculate turnover later
                portfolio_comp_before_df = self.portfolio_comp_df.copy()

            # Calculate the new portfolio weights
            self.portfolio_comp_df = calculate_portfolio_weights(trading_date_ts,
                                                                 self.portfolio_spec,
                                                                 market_data)

            if not self.last_rebalance_date_ts is None:
                turnover = compute_portfolio_turnover(portfolio_comp_before_df, self.portfolio_comp_df)
                self.portfolio_turnover_series[trading_date_ts] = turnover
                turnover_cost = self.portfolio_spec["turnover_cost_bps"] / 10000 * turnover
                self.portfolio_simple_returns_series[trading_date_ts] -= turnover_cost

            logger.info(f"Portfolio size {trading_date_ts}: {len(self.portfolio_comp_df.index)}")

            self.last_rebalance_date_ts = trading_date_ts

def evaluate_portfolio(portfolio_spec,
                       ts_start_date,
                       ts_end_date,
                       market_data):

    # Trading dates
    trading_date_ts = [pd.Timestamp(ts) for ts in market_data["stock_prices_df"].index]
    trading_date_ts = [ts for ts in trading_date_ts if ts_start_date <= ts <= ts_end_date]

    portfolio = Portfolio(trading_date_ts[0], portfolio_spec)

    for trading_date_ts in trading_date_ts:
        portfolio.update_portfolio(trading_date_ts,
                                   market_data)

    return {"portfolio_simple_returns_series": portfolio.get_portfolio_simple_returns_series(),
            "portfolio_turnover_series": portfolio.get_portfolio_turnover()}