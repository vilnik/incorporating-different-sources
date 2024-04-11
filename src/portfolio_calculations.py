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

# Check calculations
CHECK = True
def calculate_excess_log_returns_from_prices(portfolio_spec,
                                             stock_prices_df,
                                             risk_free_rate_df):
    logger.info("Calculating excess log returns.")

    # Calculate the log returns for each stock
    stock_log_returns_df = np.log(stock_prices_df / stock_prices_df.shift(1))

    # Calculate the actual frequency of stock log returns in days
    dates_diff = stock_prices_df.index.to_series().diff().dt.days.dropna()
    average_frequency_stock_log_returns = dates_diff.mean()

    # Ensure the max diff is not significantly greater than the average
    assert dates_diff.max() <= average_frequency_stock_log_returns + 4, "Unexpected large gap between return dates."

    # Adjust the risk-free rate for the observed frequency
    # Assuming risk_free_rate_df contains annualized risk-free rates
    risk_free_rate_adjusted = (1 + risk_free_rate_df) ** (average_frequency_stock_log_returns / 365) - 1

    # Ensure risk_free_rate_adjusted index aligns with risk_free_rate_df for correct operations
    risk_free_rate_adjusted.index = risk_free_rate_df.index

    # Resample and interpolate risk-free rates to match stock returns' dates
    risk_free_rate_resampled = risk_free_rate_adjusted.reindex(stock_log_returns_df.index, method = 'ffill')

    # Calculate the excess log returns
    stock_excess_log_returns_df = stock_log_returns_df - risk_free_rate_resampled.values

    # Drop NaN values, which occur for the first data point
    stock_excess_log_returns_df.dropna(inplace = True)

    return stock_excess_log_returns_df

def calculate_portfolio_variance(portfolio_weights_df,
                                 covariance_matrix_df):
    logger.info(f"Calculating portfolio variance.")

    # Sort the portfolio DataFrame by index (stock symbols)
    sorted_portfolio_weights_df = portfolio_weights_df.sort_index()
    sorted_weights_np = sorted_portfolio_weights_df['Weight'].to_numpy()

    # Sort the covariance DataFrame by stock symbols and convert to a numpy array
    sorted_keys = sorted_portfolio_weights_df.index
    sorted_covariance_matrix_df = covariance_matrix_df.loc[sorted_keys, sorted_keys]
    sorted_covariance_matrix_np = sorted_covariance_matrix_df.to_numpy()

    # Compute the portfolio variance as w^T * S * w
    portfolio_variance = np.dot(sorted_weights_np.T, np.dot(sorted_covariance_matrix_np, sorted_weights_np))

    # Check calculations
    if CHECK:
        portfolio_variance_check = portfolio_weights_df["Weight"].T.dot(covariance_matrix_df.dot(portfolio_weights_df["Weight"]))

        is_close = np.isclose(portfolio_variance, portfolio_variance_check, atol = 1e-4)
        if not is_close:
            raise ValueError(f"Portfolio variance is not consistent.")

    return portfolio_variance

def calculate_average_mcm_window(portfolio_spec,
                                trading_date_ts,
                                mcm_prices_df):

    # Ensure k_stock_prices_df.index is a DateTimeIndex and is sorted
    mcm_prices_df = mcm_prices_df.sort_index()

    # Check if trading_date_ts is the last date in mcm_prices_df
    if trading_date_ts != mcm_prices_df.index[-1]:
        logger.error(f"trading_date_ts {trading_date_ts} is not the last date in the DataFrame.")
        raise ValueError(f"trading_date_ts {trading_date_ts} must be the last date in the DataFrame.")

    if portfolio_spec["rolling_window_frequency"] == "daily":
        mcm_prices_window_df = mcm_prices_df
    elif portfolio_spec["rolling_window_frequency"] == "weekly":
        # Resample to get the last trading day of each week
        mcm_prices_window_df = mcm_prices_df.resample('W').last()
    elif portfolio_spec["rolling_window_frequency"] == "monthly":
        # Resample to get the last trading day of each month
        mcm_prices_window_df = mcm_prices_df.resample('M').last()

    # Calculate the average mcm price over the specified rolling window
    average_mcm_price = mcm_prices_window_df.iloc[-portfolio_spec["rolling_window"]:].mean().item()

    return average_mcm_price

def get_window_annualization_factor(portfolio_spec):
    if portfolio_spec["rolling_window_frequency"] == "daily":
        window_annualization_factor= 252
    elif portfolio_spec["rolling_window_frequency"] == "weekly":
        window_annualization_factor = 52
    elif portfolio_spec["rolling_window_frequency"] == "monthly":
        window_annualization_factor = 12

    return window_annualization_factor

def get_window_trading_days(portfolio_spec):
    if portfolio_spec["rolling_window_frequency"] == "daily":
        window_days = portfolio_spec["rolling_window"]
    elif portfolio_spec["rolling_window_frequency"] == "weekly":
        window_days = portfolio_spec["rolling_window"]*5
    elif portfolio_spec["rolling_window_frequency"] == "monthly":
        window_days = portfolio_spec["rolling_window"]*22

    return window_days

def adjust_stock_prices_window(portfolio_spec,
                               trading_date_ts,
                               k_stock_prices_df):
    logger.info("Adjusting daily prices to rolling window.")

    # Ensure k_stock_prices_df.index is a DateTimeIndex and is sorted
    k_stock_prices_df = k_stock_prices_df.sort_index()

    # Check if trading_date_ts is the last date in k_stock_prices_df
    if trading_date_ts != k_stock_prices_df.index[-1]:
        logger.error(f"trading_date_ts {trading_date_ts} is not the last date in the DataFrame.")
        raise ValueError(f"trading_date_ts {trading_date_ts} must be the last date in the DataFrame.")

    if portfolio_spec["rolling_window_frequency"] == "daily":
        k_stock_prices_window_df = k_stock_prices_df
    elif portfolio_spec["rolling_window_frequency"] == "weekly":
        # Resample to get the last trading day of each week
        k_stock_prices_window_df = k_stock_prices_df.resample('W').last()
    elif portfolio_spec["rolling_window_frequency"] == "monthly":
        # Resample to get the last trading day of each month
        k_stock_prices_window_df = k_stock_prices_df.resample('M').last()

    # Select the last 'rolling_window' observations from the DataFrame
    k_stock_prices_window_df = k_stock_prices_window_df.iloc[-portfolio_spec["rolling_window"]:]

    return k_stock_prices_window_df

def calculate_canonical_statistics_T(portfolio_spec,
                                     trading_date_ts,
                                     k_stock_prices_df,
                                     risk_free_rate_df):
    logger.info(f"Calculating canonical statistics T.")

    # Adjust the stock prices DataFrame based on rolling window
    k_stock_prices_window_df = adjust_stock_prices_window(
        portfolio_spec, trading_date_ts, k_stock_prices_df
    )

    # Calculate the excess log returns for the adjusted stock prices
    k_stock_excess_log_returns_window_df = calculate_excess_log_returns_from_prices(
        portfolio_spec, k_stock_prices_window_df, risk_free_rate_df
    )

    # Calculate Canonical Statistics T using DataFrame's dot product method
    canonical_statistics_T_df = k_stock_excess_log_returns_window_df.T.dot(
        k_stock_excess_log_returns_window_df
    )

    # Check calculations
    if CHECK:
        num_columns = k_stock_excess_log_returns_window_df.shape[1]
        canonical_statistics_T_matrix_check = np.zeros((num_columns, num_columns))
        for index, row in k_stock_excess_log_returns_window_df.iterrows():
            xi = row.values
            canonical_statistics_T_matrix_check += np.outer(xi, xi)

        # Convert the numpy matrix to a DataFrame
        canonical_statistics_T_df_check = pd.DataFrame(canonical_statistics_T_matrix_check)

        # Set the row and column labels to match those from k_stock_excess_log_returns_window_df
        canonical_statistics_T_df_check.columns = k_stock_excess_log_returns_window_df.columns
        canonical_statistics_T_df_check.index = k_stock_excess_log_returns_window_df.columns

        are_equal = np.isclose(canonical_statistics_T_df, canonical_statistics_T_df_check, rtol = 1e-4,
                                      atol = 1e-4).all().all()
        if not are_equal:
            raise ValueError(f"Canonical statistics T is not consistent.")

    return canonical_statistics_T_df

def calculate_canonical_statistics_t(portfolio_spec,
                                     trading_date_ts,
                                     k_stock_prices_df,
                                     risk_free_rate_df):
    logger.info(f"Calculating canonical statistics t.")

    # Adjust the stock prices DataFrame based on rolling window
    k_stock_prices_window_df = adjust_stock_prices_window(
        portfolio_spec, trading_date_ts, k_stock_prices_df
    )

    # Calculate the excess log returns for the adjusted stock prices
    k_stock_excess_log_returns_window_df = calculate_excess_log_returns_from_prices(
        portfolio_spec, k_stock_prices_window_df, risk_free_rate_df)

    # Calculate Canonical Statistics t using DataFrame's sum method
    canonical_statistics_t_df = k_stock_excess_log_returns_window_df.sum(axis=0).to_frame()

    # Check calculations
    if CHECK:
        num_columns = k_stock_excess_log_returns_window_df.shape[1]
        canonical_statistics_t_vector_check = np.zeros(num_columns)
        for index, row in k_stock_excess_log_returns_window_df.iterrows():
            xi = row.values
            canonical_statistics_t_vector_check += xi

        # Convert the numpy vector to a DataFrame
        canonical_statistics_t_df_check = pd.DataFrame(canonical_statistics_t_vector_check)

        # Set the row labels to match those from k_stock_excess_log_returns_window_df
        canonical_statistics_t_df_check.index = k_stock_excess_log_returns_window_df.columns

        are_equal = np.isclose(canonical_statistics_t_df, canonical_statistics_t_df_check, rtol = 1e-4,
                                      atol = 1e-4).all().all()

        if not are_equal:
            raise ValueError(f"Canonical statistics t is not consistent.")


    return canonical_statistics_t_df

def calculate_conjugate_prior_n(portfolio_spec,
                                trading_date_ts,
                                mcm_prices_df):
    logger.info(f"Calculating conjugate prior n.")

    average_mcm_price = calculate_average_mcm_window(portfolio_spec,
                                                     trading_date_ts,
                                                     mcm_prices_df)

    # Get the current mcm price for the trading date
    current_mcm_price = mcm_prices_df.loc[trading_date_ts].item()

    # Calculate the mcm price fraction based on current and average mcm prices
    if current_mcm_price > average_mcm_price:
        mcm_price_fraction = (current_mcm_price / average_mcm_price)
    else:
        mcm_price_fraction = (average_mcm_price / current_mcm_price)

    conjugate_prior_n = portfolio_spec["rolling_window"] * mcm_price_fraction * portfolio_spec["mcm_scaling"]

    return conjugate_prior_n

def calculate_conjugate_posterior_n(portfolio_spec,
                                    trading_date_ts,
                                    mcm_prices_df,
                                    conjugate_prior_n=None):
    logger.info(f"Calculating conjugate posterior n.")

    # If conjugate_prior_n is not provided, calculate it
    if conjugate_prior_n is None:
        conjugate_prior_n = calculate_conjugate_prior_n(portfolio_spec,
                                                        trading_date_ts,
                                                        mcm_prices_df)

    # Return the sum of the conjugate prior 'n' and the adjusted rolling window
    return conjugate_prior_n + portfolio_spec["rolling_window"]


def calculate_conjugate_prior_S(portfolio_spec,
                                trading_date_ts,
                                k_stock_intraday_prices_df,
                                mcm_prices_df,
                                conjugate_prior_n=None):
    logger.info(f"Calculating conjugate prior S.")

    # If conjugate_prior_n is not provided, calculate it
    if conjugate_prior_n is None:
        conjugate_prior_n = calculate_conjugate_prior_n(portfolio_spec,
                                                        trading_date_ts,
                                                        mcm_prices_df)

    # Calculate log returns for the last period of intraday prices
    if portfolio_spec["rolling_window_frequency"] == "daily":
        days_in_single_rolling_window = 1
    elif portfolio_spec["rolling_window_frequency"] == "weekly":
        days_in_single_rolling_window = 7
    elif portfolio_spec["rolling_window_frequency"] == "monthly":
        days_in_single_rolling_window = 31
    else:
        # Log an error and raise an exception if the rebalancing frequency is unknown
        logger.error("Unknown rolling window frequency.")
        raise RuntimeError("Unknown rolling window frequency.")

    hf_start_date = trading_date_ts - pd.Timedelta(days=days_in_single_rolling_window)
    filtered_intraday_prices_df = k_stock_intraday_prices_df[(k_stock_intraday_prices_df.index > (hf_start_date + pd.Timedelta(days=1))) &
                                                            (k_stock_intraday_prices_df.index <= (trading_date_ts + pd.Timedelta(days=1)))]

    k_stock_intraday_log_returns_last_period_df = np.log(filtered_intraday_prices_df / filtered_intraday_prices_df.shift(1)).dropna()

    # Compute the covariance matrix of the log returns, scaled by the number of observations
    k_stock_intraday_cov_last_period_df = k_stock_intraday_log_returns_last_period_df.cov() * len(
        k_stock_intraday_log_returns_last_period_df)

    # Check calculations
    if CHECK:
        k_stock_intraday_cov_last_period_df_check = k_stock_intraday_log_returns_last_period_df.T.dot(
            k_stock_intraday_log_returns_last_period_df
        )

        are_equal = np.isclose(k_stock_intraday_cov_last_period_df, k_stock_intraday_cov_last_period_df_check, rtol = 1e-3,
                                      atol = 1e-3).all().all()

        if not are_equal:
            raise ValueError(f"Realized covariance matrix is not consistent.")

    # Return the scaled covariance matrix
    return conjugate_prior_n * k_stock_intraday_cov_last_period_df

def calculate_conjugate_posterior_S(portfolio_spec,
                                    trading_date_ts,
                                    k_stock_prices_df,
                                    k_stock_intraday_prices_df,
                                    mcm_prices_df,
                                    risk_free_rate_df,
                                    conjugate_prior_S_df=None):
    logger.info(f"Calculating conjugate posterior S.")

    # If conjugate_prior_S_df is not provided, calculate it
    if conjugate_prior_S_df is None:
        conjugate_prior_S_df = calculate_conjugate_prior_S(portfolio_spec,
                                                           trading_date_ts,
                                                           k_stock_intraday_prices_df,
                                                           mcm_prices_df)

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
                                mcm_prices_df):
    logger.info(f"Calculating conjugate prior w.")

    # Initialize weights for each stock
    if "vw" in portfolio_spec["weighting_strategy"]:
        portfolio_weights_df = calculate_value_weighted_portfolio(portfolio_spec,
                                                               trading_date_ts,
                                                               k_stock_market_caps_df)
    elif "ew" in portfolio_spec["weighting_strategy"]:
        portfolio_weights_df = calculate_equally_weighted_portfolio(portfolio_spec,
                                                                 k_stock_prices_df)
    else:
        logger.error(f"Unknown conjugate portfolio prior weights.")
        raise ValueError("Unknown conjugate portfolio prior weights.")

    return portfolio_weights_df

def calculate_conjugate_c(portfolio_spec,
                          trading_date_ts,
                          k_stock_prices_df,
                          k_stock_market_caps_df,
                          k_stock_intraday_prices_df,
                          mcm_prices_df,
                          conjugate_prior_n=None,
                          conjugate_prior_S_df=None,
                          conjugate_prior_w_df=None):
    logger.info(f"Calculating conjugate c.")

    # Calculate 'conjugate_prior_n' if not provided
    if conjugate_prior_n is None:
        conjugate_prior_n = calculate_conjugate_prior_n(portfolio_spec,
                                                        trading_date_ts,
                                                        mcm_prices_df)

    # Calculate 'conjugate_prior_S_df' if not provided
    if conjugate_prior_S_df is None:
        conjugate_prior_S_df = calculate_conjugate_prior_S(portfolio_spec,
                                                           trading_date_ts,
                                                           k_stock_intraday_prices_df,
                                                           mcm_prices_df)

    # Calculate 'conjugate_prior_w_df' if not provided
    if conjugate_prior_w_df is None:
        conjugate_prior_w_df = calculate_conjugate_prior_w(portfolio_spec,
                                                            trading_date_ts,
                                                            k_stock_prices_df,
                                                            k_stock_market_caps_df,
                                                            mcm_prices_df)

    # Compute 'conjugate_c' using the provided formulas
    conjugate_c = (2 * conjugate_prior_n) / ((conjugate_prior_n + portfolio_spec["size"] + 2) +
                                      ((conjugate_prior_n + portfolio_spec["size"] + 2) ** 2 +
                                      4 * conjugate_prior_n * calculate_portfolio_variance(conjugate_prior_w_df,
                                                                                           conjugate_prior_S_df)) ** (1 / 2))

    if CHECK:
        conjugate_c_check = (-(conjugate_prior_n + portfolio_spec["size"] + 2) + \
                            ((conjugate_prior_n + portfolio_spec["size"] + 2)**2 + 4*conjugate_prior_n*calculate_portfolio_variance(conjugate_prior_w_df,
                                                                                           conjugate_prior_S_df))**(1/2)) / \
                            (2*calculate_portfolio_variance(conjugate_prior_w_df,conjugate_prior_S_df))

        is_close = np.isclose(conjugate_c, conjugate_c_check, atol = 1e-3)
        if not is_close:
            raise ValueError(f"Portfolio c is not consistent.")

    return conjugate_c

def calculate_conjugate_posterior_w(portfolio_spec,
                                    trading_date_ts,
                                    k_stock_prices_df,
                                    k_stock_market_caps_df,
                                    k_stock_intraday_prices_df,
                                    mcm_prices_df,
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
                                            mcm_prices_df)

    # Calculate 'conjugate_prior_w_df' if not provided
    if conjugate_prior_w_df is None:
        conjugate_prior_w_df = calculate_conjugate_prior_w(portfolio_spec,
                                                            trading_date_ts,
                                                            k_stock_prices_df,
                                                            k_stock_market_caps_df,
                                                            mcm_prices_df)

    # Calculate 'conjugate_posterior_S_df' if not provided
    if conjugate_posterior_S_df is None:
        if conjugate_prior_S_df is None:
            conjugate_prior_S_df = calculate_conjugate_prior_S(portfolio_spec,
                                                               trading_date_ts,
                                                               k_stock_intraday_prices_df,
                                                               mcm_prices_df)

        conjugate_posterior_S_df = calculate_conjugate_posterior_S(portfolio_spec,
                                                                   trading_date_ts,
                                                                   k_stock_prices_df,
                                                                   k_stock_intraday_prices_df,
                                                                   mcm_prices_df,
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
                                          mcm_prices_df,
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
                                            mcm_prices_df)

    # Calculate 'conjugate_posterior_n' if not provided
    if conjugate_posterior_n is None:
        if conjugate_prior_n is None:
            conjugate_prior_n = calculate_conjugate_prior_n(portfolio_spec,
                                                            trading_date_ts,
                                                            mcm_prices_df)
        conjugate_posterior_n = calculate_conjugate_posterior_n(portfolio_spec,
                                                                trading_date_ts,
                                                                mcm_prices_df,
                                                                conjugate_prior_n=conjugate_prior_n)

    # Calculate 'conjugate_posterior_S_df' if not provided
    if conjugate_posterior_S_df is None:
        if conjugate_prior_S_df is None:
            conjugate_prior_S_df = calculate_conjugate_prior_S(portfolio_spec,
                                                               trading_date_ts,
                                                               k_stock_intraday_prices_df,
                                                               mcm_prices_df)
        conjugate_posterior_S_df = calculate_conjugate_posterior_S(portfolio_spec,
                                                                   trading_date_ts,
                                                                   k_stock_prices_df,
                                                                   k_stock_intraday_prices_df,
                                                                   mcm_prices_df,
                                                                   risk_free_rate_df,
                                                                   conjugate_prior_S_df=conjugate_prior_S_df)

    # Calculate 'conjugate_posterior_w_df' if not provided
    if conjugate_posterior_w_df is None:
        if conjugate_prior_w_df is None:
            conjugate_prior_w_df = calculate_conjugate_prior_w(portfolio_spec,
                                                                trading_date_ts,
                                                                k_stock_prices_df,
                                                                k_stock_market_caps_df,
                                                                mcm_prices_df)
        conjugate_posterior_w_df = calculate_conjugate_posterior_w(portfolio_spec,
                                                                trading_date_ts,
                                                                k_stock_prices_df,
                                                                k_stock_market_caps_df,
                                                                k_stock_intraday_prices_df,
                                                                mcm_prices_df,
                                                                risk_free_rate_df,
                                                                conjugate_c=conjugate_c,
                                                                conjugate_prior_w_df=conjugate_prior_w_df,
                                                                conjugate_prior_S_df=conjugate_prior_S_df,
                                                                conjugate_posterior_S_df=conjugate_posterior_S_df)

    # Compute and return the mean conjugate posterior nu
    mean_conjugate_posterior_nu_df = (conjugate_posterior_n + portfolio_spec["size"] + 2) * \
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
    jeffreys_scaled_covariance_matrix_df = (canonical_statistics_T_df - 1 / portfolio_spec["rolling_window"] * \
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
                                     rolling_window_frequency):
    # Get S&P 500 components for the current date
    tickers_list = data_handling.extract_unique_tickers(trading_date_ts,
                                          trading_date_ts)

    # Identify tickers that are present in stock_market_caps_df.columns
    present_tickers = [ticker for ticker in tickers_list if ticker in stock_market_caps_df.columns]
    missing_fraction = (len(tickers_list) - len(present_tickers)) / len(tickers_list)
    logger.info(f"Fraction of tickers missing from stock_market_caps_df: {missing_fraction:.2%}")

    # Days in single rolling window
    if rolling_window_frequency == "daily":
        days_in_single_rolling_window = 1
    elif rolling_window_frequency == "weekly":
        days_in_single_rolling_window = 7
    elif rolling_window_frequency == "monthly":
        days_in_single_rolling_window = 31
    else:
        # Log an error and raise an exception if the rebalancing frequency is unknown
        logger.error("Unknown rolling window frequency.")
        raise RuntimeError("Unknown rolling window frequency.")

    eligible_stocks = [
        stock for stock in stock_prices_df.columns
        if (
                stock in tickers_list and
                stock in stock_market_caps_df.columns and
                stock in stock_intraday_prices_df.columns and
                stock_prices_df.loc[trading_date_ts, stock] is not None and
                stock_prices_df[stock].loc[:trading_date_ts].tail(rolling_window_days).notna().all() and
                stock_intraday_prices_df[stock].loc[(trading_date_ts - timedelta(days=days_in_single_rolling_window)):(trading_date_ts + timedelta(days = 1))].notna().any()
        )
    ]

    # From these available stocks, get the size largest based on market caps for the current date
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
    num_stocks = portfolio_spec["size"]

    # Assign equal weight to each stock and create the resulting dataframe
    portfolio_weights_df = pd.DataFrame({
        'Weight': [1 / num_stocks] * num_stocks
    }, index=k_stock_prices_df.columns)

    # Rename the index to 'Stock'
    portfolio_weights_df.index.name = 'Stock'

    return portfolio_weights_df

def calculate_value_weighted_portfolio(portfolio_spec, 
                                       trading_date_ts, 
                                       k_stock_market_caps_df):
    logger.info(f"Calculating market cap portfolio weights.")
    k_stock_market_caps_series = k_stock_market_caps_df.iloc[-1].sort_values(ascending=False)

    # Extract the last index date from k_stock_market_caps_df
    last_index_date = k_stock_market_caps_df.index[-1]

    # Assert that the last index date is the same as trading_date_ts
    assert last_index_date == trading_date_ts, "The last index date does not match the trading date."

    # Total market cap of the k largest stocks
    total_market_cap = k_stock_market_caps_series.sum()

    # Calculate value weights
    portfolio_weights_df = pd.DataFrame(k_stock_market_caps_series / total_market_cap)

    # Fix labels
    portfolio_weights_df.index.name = 'Stock'
    portfolio_weights_df.columns = ['Weight']

    return portfolio_weights_df

def calculate_shrinkage_portfolio(portfolio_spec, 
                                  trading_date_ts, 
                                  k_stock_prices_df,
                                  risk_free_rate_df):
    logger.info(f"Calculating shrinkage portfolio weights.")
    k_stock_prices_window_df = adjust_stock_prices_window(
        portfolio_spec,
        trading_date_ts,
        k_stock_prices_df)

    # Calculate the excess log returns for the adjusted stock prices
    k_stock_excess_log_returns_window_df = calculate_excess_log_returns_from_prices(
        portfolio_spec, k_stock_prices_window_df, risk_free_rate_df
    )

    window_annualization_factor = get_window_annualization_factor(portfolio_spec)

    # Mean return
    mean_log_returns = expected_returns.mean_historical_return(k_stock_excess_log_returns_window_df,
                                                                 returns_data=True,
                                                                 compounding=False,
                                                                  frequency = window_annualization_factor)

    # Covariance matrix
    covariance_log_returns = risk_models.CovarianceShrinkage(k_stock_excess_log_returns_window_df,
                                                                returns_data=True,
                                                                frequency = window_annualization_factor).ledoit_wolf()

    # Add risk free asset
    mean_log_returns_with_risk_free_asset = mean_log_returns.copy()
    mean_log_returns_with_risk_free_asset["RISK_FREE"] = 0

    covariance_log_returns_with_risk_free_asset = covariance_log_returns.copy()
    covariance_log_returns_with_risk_free_asset["RISK_FREE"] = 0
    covariance_log_returns_with_risk_free_asset.loc["RISK_FREE"] = 0

    ef = EfficientFrontier(mean_log_returns_with_risk_free_asset, covariance_log_returns_with_risk_free_asset, weight_bounds=(-100, 100))
    raw_portfolio_comp = ef.max_quadratic_utility(risk_aversion=portfolio_spec["risk_aversion"])

    # Convert cleaned weights to DataFrame
    portfolio_weights_df = pd.DataFrame(list(ef.clean_weights().items()), columns=['Stock', 'Weight'])
    portfolio_weights_df.set_index('Stock', inplace=True)
    portfolio_weights_df = portfolio_weights_df.drop("RISK_FREE")

    # Check that it is the same as doing 1/gamma * Sigma^{-1}mu
    if CHECK:
        # Calculate the inverse of the covariance matrix
        inv_covariance = np.linalg.inv(covariance_log_returns.values)

        # Calculate portfolio weights for the tangency portfolio
        weights = 1 / portfolio_spec["risk_aversion"] * np.dot(inv_covariance, mean_log_returns.values)

        # Create a DataFrame for the portfolio composition
        portfolio_weights_df_check = pd.DataFrame(weights, index = covariance_log_returns.index, columns = ['Weight'])

        are_equal = np.isclose(portfolio_weights_df_check, portfolio_weights_df, rtol = 1e-4,
                               atol = 1e-4).all().all()

        if not are_equal:
            raise ValueError(f"Shrinkage weights are not consistent.")

    return portfolio_weights_df

def calculate_black_litterman_portfolio(portfolio_spec,
                                        trading_date_ts,
                                        k_stock_market_caps_df,
                                        k_stock_prices_df,
                                        risk_free_rate_df):
    logger.info(f"Calculating Black-Litterman portfolio weights.")
    k_stock_prices_window_df = adjust_stock_prices_window(
        portfolio_spec,
        trading_date_ts,
        k_stock_prices_df)

    # Calculate the excess log returns for the adjusted stock prices
    k_stock_excess_log_returns_window_df = calculate_excess_log_returns_from_prices(
        portfolio_spec, k_stock_prices_window_df, risk_free_rate_df
    )

    k_stock_market_caps_latest_df = k_stock_market_caps_df.iloc[-1].sort_values(ascending=False)

    window_annualization_factor = get_window_annualization_factor(portfolio_spec)

    # Covariance matrix
    covariance_log_returns = risk_models.CovarianceShrinkage(k_stock_excess_log_returns_window_df,
                                                             returns_data=True,
                                                             frequency = window_annualization_factor).ledoit_wolf()

    viewdict = {}
    market_prior = black_litterman.market_implied_prior_returns(k_stock_market_caps_latest_df.squeeze(),
                                                                portfolio_spec["risk_aversion"],
                                                                covariance_log_returns,
                                                                risk_free_rate = 0)

    bl = BlackLittermanModel(covariance_log_returns, pi=market_prior, absolute_views=viewdict)
    bl_mean_log_returns = bl.bl_returns()
    bl_covariance_log_returns = bl.bl_cov()

    # Add risk free asset
    bl_mean_log_returns_with_risk_free_asset = bl_mean_log_returns.copy()
    bl_mean_log_returns_with_risk_free_asset["RISK_FREE"] = 0

    bl_covariance_log_returns_with_risk_free_asset = bl_covariance_log_returns.copy()
    bl_covariance_log_returns_with_risk_free_asset["RISK_FREE"] = 0
    bl_covariance_log_returns_with_risk_free_asset.loc["RISK_FREE"] = 0

    ef = EfficientFrontier(bl_mean_log_returns_with_risk_free_asset, bl_covariance_log_returns_with_risk_free_asset, weight_bounds=(-100, 100))
    raw_portfolio_comp = ef.max_quadratic_utility(risk_aversion=portfolio_spec["risk_aversion"])

    # Convert cleaned weights to DataFrame
    portfolio_weights_df = pd.DataFrame(list(ef.clean_weights().items()), columns=['Stock', 'Weight'])
    portfolio_weights_df.set_index('Stock', inplace=True)
    portfolio_weights_df = portfolio_weights_df.drop("RISK_FREE")

    return portfolio_weights_df

def calculate_conjugate_hf_mcm_portfolio(portfolio_spec,
                                         trading_date_ts,
                                         k_stock_market_caps_df,
                                         k_stock_prices_df,
                                         k_stock_intraday_prices_df,
                                         mcm_prices_df,
                                         risk_free_rate_df):
    logger.info(f"Calculating conjugate high frequency MCM portfolio weights.")

    mean_conjugate_posterior_nu_df = calculate_mean_conjugate_posterior_nu(portfolio_spec,
                                                                        trading_date_ts,
                                                                        k_stock_prices_df,
                                                                        k_stock_market_caps_df,
                                                                        k_stock_intraday_prices_df,
                                                                        mcm_prices_df,
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

def calculate_jorion_portfolio(portfolio_spec,
                              trading_date_ts,
                              k_stock_prices_df,
                              risk_free_rate_df):

    logger.info(f"Calculating Jorion portfolio.")

    # Adjust the stock prices DataFrame based on rolling window
    k_stock_prices_window_df = adjust_stock_prices_window(
        portfolio_spec, trading_date_ts, k_stock_prices_df
    )

    # Calculate the excess log returns for the adjusted stock prices
    k_stock_excess_log_returns_window_df = calculate_excess_log_returns_from_prices(
        portfolio_spec, k_stock_prices_window_df, risk_free_rate_df
    )

    # Using notations of Bayesian Portfolio Analysis (2010) by Avramov and Zhou
    N = len(k_stock_prices_df.columns)
    T = len(k_stock_excess_log_returns_window_df)

    # Sample mean
    mu_hat_df = k_stock_excess_log_returns_window_df.mean().to_frame()

    # Sample covariance
    V_hat_df = k_stock_excess_log_returns_window_df.cov()

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

    portfolio_weights_df = 1 / portfolio_spec["risk_aversion"] * V_hat_PJ_inverse_df.dot(mu_hat_PJ_df).reset_index().rename(columns={'index': 'Stock', 0: 'Weight'}).set_index('Stock')

    return portfolio_weights_df

def calculate_greyserman_portfolio(portfolio_spec,
                                      trading_date_ts,
                                      k_stock_prices_df,
                                      risk_free_rate_df):

    logger.info(f"Calculating Greyserman portfolio.")

    # Adjust the stock prices DataFrame based on the portfolio's rebalancing frequency and rolling window
    k_stock_prices_window_df = adjust_stock_prices_window(
        portfolio_spec, trading_date_ts, k_stock_prices_df
    )

    # Calculate the excess log returns for the adjusted stock prices
    k_stock_excess_log_returns_window_df = calculate_excess_log_returns_from_prices(
        portfolio_spec, k_stock_prices_window_df, risk_free_rate_df
    )

    n = len(k_stock_excess_log_returns_window_df)
    k = len(k_stock_prices_df.columns)
    x_bar_df = k_stock_excess_log_returns_window_df.mean().to_frame()
    S_df = k_stock_excess_log_returns_window_df.cov()
    S_h_df = pd.DataFrame(np.where(np.eye(k) == 1, 1, 0.5), index=S_df.index[:k], columns=S_df.index[:k])
    one_N_df = pd.DataFrame(np.ones(k), index=x_bar_df.index)
    kappa_h = round(0.1 * n)
    nu_h = k
    weights_b_storage = []
    # Using notations of Incorporating different sources of information for Bayesian optimal portfolio selection by Bodnar et al.
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
    epu_prices_df = market_data["epu_prices_df"]
    risk_free_rate_df = market_data["risk_free_rate_df"]
    sp500_prices_df = market_data["sp500_prices_df"]

    # Get k largest stocks and market caps at trading_date_ts
    k_stock_market_caps_trading_date_df = get_k_largest_stocks_market_caps(stock_market_caps_df,
                                                              stock_prices_df,
                                                              stock_intraday_prices_df,
                                                              trading_date_ts,
                                                              portfolio_spec["size"],
                                                              get_window_trading_days(portfolio_spec),
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

    # Filter EPU prices
    epu_prices_df = epu_prices_df.loc[epu_prices_df.index <= trading_date_ts]

    # Filter S&P 500 prices
    sp500_prices_df = sp500_prices_df.loc[sp500_prices_df.index <= trading_date_ts]

    # Check for NA values in the filtered DataFrame
    if k_stock_prices_df.tail(get_window_trading_days(portfolio_spec)).isna().any().any():
        logger.error(f"Found NA values in the filtered stock prices.")
        raise ValueError(f"The filtered stock prices contain NA values.")

    if portfolio_spec["weighting_strategy"] == "vw":
        portfolio_weights_df = calculate_value_weighted_portfolio(portfolio_spec,
                                                                trading_date_ts,
                                                                k_stock_market_caps_df)

    elif portfolio_spec["weighting_strategy"] == "ew":
        portfolio_weights_df = calculate_equally_weighted_portfolio(portfolio_spec,
                                                                k_stock_prices_df)

    elif portfolio_spec["weighting_strategy"] == "shrinkage":
        portfolio_weights_df = calculate_shrinkage_portfolio(portfolio_spec,
                                                           trading_date_ts,
                                                           k_stock_prices_df,
                                                          risk_free_rate_df)

    elif portfolio_spec["weighting_strategy"] == "black_litterman":
        portfolio_weights_df = calculate_black_litterman_portfolio(portfolio_spec,
                                                                 trading_date_ts,
                                                                 k_stock_market_caps_df,
                                                                 k_stock_prices_df,
                                                                 risk_free_rate_df)

    elif portfolio_spec["weighting_strategy"] == "conjugate_hf_vix_vw" or portfolio_spec["weighting_strategy"] == "conjugate_hf_vix_ew":
        portfolio_weights_df = calculate_conjugate_hf_mcm_portfolio(portfolio_spec,
                                                                  trading_date_ts,
                                                                  k_stock_market_caps_df,
                                                                  k_stock_prices_df,
                                                                  k_stock_intraday_prices_df,
                                                                  vix_prices_df,
                                                                  risk_free_rate_df)

    elif portfolio_spec["weighting_strategy"] == "conjugate_hf_epu_vw" or portfolio_spec["weighting_strategy"] == "conjugate_hf_epu_ew":
        portfolio_weights_df = calculate_conjugate_hf_mcm_portfolio(portfolio_spec,
                                                                  trading_date_ts,
                                                                  k_stock_market_caps_df,
                                                                  k_stock_prices_df,
                                                                  k_stock_intraday_prices_df,
                                                                  epu_prices_df,
                                                                  risk_free_rate_df)

    elif portfolio_spec["weighting_strategy"] == "jeffreys":
        portfolio_weights_df = calculate_jeffreys_portfolio(portfolio_spec,
                                                          trading_date_ts,
                                                          k_stock_prices_df,
                                                          risk_free_rate_df)

    elif portfolio_spec["weighting_strategy"] == "jorion":
        portfolio_weights_df = calculate_jorion_portfolio(portfolio_spec,
                                                                      trading_date_ts,
                                                                      k_stock_prices_df,
                                                                      risk_free_rate_df)

    elif portfolio_spec["weighting_strategy"] == "greyserman":
        portfolio_weights_df = calculate_greyserman_portfolio(portfolio_spec,
                                                              trading_date_ts,
                                                              k_stock_prices_df,
                                                              risk_free_rate_df)

    else:
        logger.error(f"Unknown weights spec.")
        raise ValueError(f"Unknown weights spec.")

    return portfolio_weights_df

def compute_portfolio_turnover(portfolio_weights_before_df, portfolio_weights_after_df):

    # Merging the old and new weights with a suffix to differentiate them
    portfolio_weights_before_after_df = portfolio_weights_before_df.merge(portfolio_weights_after_df,
                                                how='outer',
                                                left_index=True,
                                                right_index=True,
                                                suffixes=('_before', '_after'))

    # Fill missing values with 0s (for new stocks or those that have been removed)
    portfolio_weights_before_after_df.fillna(0, inplace=True)

    # Calculate absolute difference for each stock and then compute turnover
    portfolio_weights_before_after_df['weight_diff'] = abs(portfolio_weights_before_after_df['Weight_before'] - portfolio_weights_before_after_df['Weight_after'])

    # Calculate turnover corresponding to risk free asset
    risk_free_turnover = abs(portfolio_weights_before_df['Weight'].sum() - portfolio_weights_after_df['Weight'].sum())

    # Calculate total turnover
    turnover = (portfolio_weights_before_after_df['weight_diff'].sum() + risk_free_turnover) / 2

    return turnover

def calculate_average_distance_to_comparison_portfolio(portfolio_weights_df,
                                                  portfolio_spec,
                                                  trading_date_ts,
                                                  market_data,
                                                  comparison_portfolio_weighting_strategy):
    comparison_portfolio_spec = {"size": portfolio_spec["size"],
                                 "rebalancing_frequency": portfolio_spec["rebalancing_frequency"],
                                 "rolling_window": portfolio_spec["rolling_window"],
                                 "rolling_window_frequency": portfolio_spec["rolling_window_frequency"]}

    if comparison_portfolio_weighting_strategy == "vw":
        comparison_portfolio_spec["weighting_strategy"] = "vw"
    else:
        raise ValueError("Unknown comparison portfolio.")

    comparison_portfolio_weights_df = calculate_portfolio_weights(trading_date_ts,
                                                                    comparison_portfolio_spec,
                                                                    market_data)

    # Ensure portfolio_weights_df and comparison_portfolio_weights_df have exactly the same stocks
    if not portfolio_weights_df.index.equals(comparison_portfolio_weights_df.index):
        raise ValueError("The portfolios do not match exactly in terms of stocks.")

    # Compute average distance between weights using L1 norm
    scaling = portfolio_spec["risk_aversion"] if portfolio_spec.get("risk_aversion") is not None else 1
    average_distance = np.abs(portfolio_weights_df * scaling - comparison_portfolio_weights_df).mean().item()

    return average_distance

class Portfolio:

    def get_portfolio_simple_returns(self):
        return self.portfolio_simple_returns_series

    def get_portfolio_turnover(self):
        return self.portfolio_turnover_series

    def get_portfolio_weights_metrics(self):
        return self.portfolio_weights_metrics_df

    def __init__(self,
                 ts_start_date,
                 portfolio_spec):
        self.ts_start_date = ts_start_date
        self.portfolio_spec = portfolio_spec
        self.portfolio_simple_returns_series = pd.Series(dtype = "float64", name = portfolio_spec["display_name"])
        self.portfolio_turnover_series = pd.Series(dtype = "float64", name = portfolio_spec["display_name"])
        self.portfolio_weights_metrics_df = pd.DataFrame(dtype = "float64")
        self.last_rebalance_date_ts = None

    def update_portfolio(self,
                         trading_date_ts,
                         market_data):

        # Calculate daily portfolio return
        if self.ts_start_date != trading_date_ts:
            # Filter out stocks not in the portfolio
            filtered_stock_simple_returns_series = market_data["stock_simple_returns_df"].loc[trading_date_ts].reindex(self.portfolio_weights_df.index)

            # Multiply returns by weights element-wise and then sum to get the portfolio return
            portfolio_simple_return = (filtered_stock_simple_returns_series * self.portfolio_weights_df['Weight']).sum()

            # Add risk-free return
            risk_free_rate_df = market_data["risk_free_rate_df"]
            most_recent_risk_free_rate = risk_free_rate_df.asof(trading_date_ts).iloc[0]
            risk_free_daily_return = ((most_recent_risk_free_rate + 1) ** (1 / 252) - 1)
            portfolio_simple_return += (1 - self.portfolio_weights_df['Weight'].sum()) * risk_free_daily_return

            self.portfolio_simple_returns_series[trading_date_ts] = portfolio_simple_return

            # Update weight for the risk-free asset
            current_risk_free_weight = 1 - self.portfolio_weights_df['Weight'].sum()
            updated_risk_free_weight = current_risk_free_weight * (1 + risk_free_daily_return)

            # Update weights for the stocks
            self.portfolio_weights_df['Weight'] = (
                        self.portfolio_weights_df['Weight'] * (1 + filtered_stock_simple_returns_series))

            # Update the total invested value by adding the updated risk-free weight
            total_value = self.portfolio_weights_df['Weight'].sum() + updated_risk_free_weight

            # Normalize the weights so they sum up to 1
            self.portfolio_weights_df['Weight'] = self.portfolio_weights_df['Weight'] / total_value

            # Check that weights sum to 1
            if abs((self.portfolio_weights_df['Weight'].values.sum() + updated_risk_free_weight / total_value) - 1) > 1e-5:
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
                portfolio_weights_before_df = self.portfolio_weights_df.copy()

            # Calculate the new portfolio weights
            self.portfolio_weights_df = calculate_portfolio_weights(trading_date_ts,
                                                                 self.portfolio_spec,
                                                                 market_data)

            average_distance_to_comparison_portfolio = calculate_average_distance_to_comparison_portfolio(self.portfolio_weights_df,
                                                                                                          self.portfolio_spec,
                                                                                                          trading_date_ts,
                                                                                                          market_data,
                                                                                                          "vw")

            portfolio_weights_max_long = self.portfolio_weights_df ['Weight'][self.portfolio_weights_df['Weight'] > 0].max()
            portfolio_weights_max_short = self.portfolio_weights_df ['Weight'][self.portfolio_weights_df['Weight'] < 0].min()
            portfolio_weights_avg_long = self.portfolio_weights_df ['Weight'][self.portfolio_weights_df['Weight'] > 0].mean()
            portfolio_weights_avg_short = self.portfolio_weights_df ['Weight'][self.portfolio_weights_df['Weight'] < 0].mean()
            portfolio_weights_avg_distance = average_distance_to_comparison_portfolio

            # Prepare a DataFrame row to append
            portfolio_weights_metrics_date_df = pd.DataFrame({
                "max_long": [portfolio_weights_max_long],
                "max_short": [portfolio_weights_max_short],
                "avg_long": [portfolio_weights_avg_long],
                "avg_short": [portfolio_weights_avg_short],
                "average_distance_to_comparison_portfolio": [portfolio_weights_avg_distance]
            }, index = [trading_date_ts])

            self.portfolio_weights_metrics_df = pd.concat([self.portfolio_weights_metrics_df, portfolio_weights_metrics_date_df])

            if not self.last_rebalance_date_ts is None:
                turnover = compute_portfolio_turnover(portfolio_weights_before_df, self.portfolio_weights_df)
                self.portfolio_turnover_series[trading_date_ts] = turnover
                turnover_cost = self.portfolio_spec["turnover_cost"] / 10000 * turnover
                self.portfolio_simple_returns_series[trading_date_ts] -= turnover_cost

            logger.info(f"Portfolio size {trading_date_ts}: {len(self.portfolio_weights_df.index)}")

            self.last_rebalance_date_ts = trading_date_ts

def backtest_portfolio(portfolio_spec,
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

    return {"portfolio_simple_returns_series": portfolio.get_portfolio_simple_returns(),
            "portfolio_turnover_series": portfolio.get_portfolio_turnover(),
            "portfolio_weights_metrics_df": portfolio.get_portfolio_weights_metrics()}
