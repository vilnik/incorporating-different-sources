import os
import pandas as pd
import quantstats as qs
from dotenv import load_dotenv
import logging
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import FuncFormatter

load_dotenv()
logging_level = os.environ.get("LOGGING_LEVEL", logging.INFO)
logging.basicConfig(level=logging_level)
logger = logging.getLogger(__name__)

# Get the absolute path of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))


def format_pct_axis(x, pos):
    return f"{x * 100:.0f}%"

def plot_vix_and_sp500(vix_prices_df, sp500_simple_returns_df):
    # Calculate the cumulative returns for S&P 500
    sp500_cumulative_returns = (1 + sp500_simple_returns_df).cumprod() - 1

    # Create the figure and axis objects
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Formatting
    ax1.spines["top"].set_visible(False)
    ax1.spines["bottom"].set_visible(False)
    ax1.spines["left"].set_visible(False)

    # main title
    fig.suptitle("VIX Prices and S&P 500 Cumulative Returns", y=0.96, fontweight="bold", fontsize=14, color="black")

    # Date range
    date_range = "{} - {}".format(
        sp500_cumulative_returns.index[0].strftime('%-d %b \'%y'),
        sp500_cumulative_returns.index[-1].strftime('%-d %b \'%y')
    )

    fig.text(0.5, 0.91, date_range, ha='center', va='center', fontsize=12, color="gray")

    ax1.set_facecolor("white")
    fig.set_facecolor("white")

    # Plotting the VIX Prices
    ax1.plot(vix_prices_df.index, vix_prices_df, label="VIX Prices", color=(0.5, 0.5, 0.5))
    ax1.set_ylabel("VIX Prices", color=(0.5, 0.5, 0.5))
    ax1.tick_params(axis='y', labelcolor=(0.5, 0.5, 0.5))

    # Create another y-axis for the S&P 500 Cumulative Returns
    ax2 = ax1.twinx()
    ax2.spines["right"].set_visible(False)
    ax2.plot(sp500_cumulative_returns.index, sp500_cumulative_returns, label="S&P 500 Cumulative Returns",
             color=(79 / 255, 139 / 255, 188 / 255))
    ax2.set_ylabel("S&P 500 Cumulative Returns", color=(79 / 255, 139 / 255, 188 / 255))
    ax2.tick_params(axis='y', labelcolor=(79 / 255, 139 / 255, 188 / 255))
    ax2.yaxis.set_major_formatter(FuncFormatter(format_pct_axis))

    # Legend below the figure
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='upper center', bbox_to_anchor=(0.5, -0.15), fontsize=11, ncol=2)

    # Show grid
    ax1.grid(True, linestyle='-', linewidth=0.5, color='gray')

    # Rotate and align the tick labels
    fig.autofmt_xdate()
    ax1.fmt_xdata = mdates.DateFormatter('%Y-%m-%d')

    # Save the figure
    plt.savefig(f"../results/vix_vs_sp500.png", dpi=400, bbox_inches='tight')


def plot_performance(portfolio_setups_simple_returns_df,
                     portfolio_setups_excess_simple_returns_df,
                     tc):
    # Filter columns
    # Create empty DataFrames for storing the filtered columns
    filtered_simple_returns_df = pd.DataFrame()
    filtered_excess_simple_returns_df = pd.DataFrame()

    # Loop through each column name to check the conditions
    for col in portfolio_setups_simple_returns_df.columns:
        if f'tc={tc}' in col:
            new_col_name = col.replace(f'tc={tc}', '').replace(',', '')
            filtered_simple_returns_df[new_col_name] = portfolio_setups_simple_returns_df[col]

    for col in portfolio_setups_excess_simple_returns_df.columns:
        if f'tc={tc}' in col:
            new_col_name = col.replace(f'tc={tc}', '').replace(',', '')
            filtered_excess_simple_returns_df[new_col_name] = portfolio_setups_excess_simple_returns_df[col]

    # Plot daily returns
    qs.plots.returns(filtered_simple_returns_df,
                           savefig = f"../results/returns_tc{tc}")

    # Plot yearly returns
    qs.plots.yearly_returns(filtered_simple_returns_df,
                           savefig = f"../results/yearly_returns_tc{tc}")

    # Plot rolling Sharpe
    qs.plots.rolling_sharpe(filtered_excess_simple_returns_df,
                           savefig = f"../results/rolling_sharpe_tc{tc}")

    # Plot rolling Sharpe
    qs.plots.rolling_sortino(filtered_excess_simple_returns_df,
                           savefig = f"../results/rolling_sortino_tc{tc}")

    # Plot rolling volatility
    qs.plots.rolling_volatility(filtered_simple_returns_df,
                           savefig = f"../results/rolling_volatility_tc{tc}")

    # Plot drawdown
    qs.plots.drawdown(filtered_simple_returns_df,
                      savefig=f"../results/drawdown_tc{tc}")


def process_and_highlight_values(metrics_df):
    # Set of metrics that are better when higher
    higher_is_better = {
        'Cum. Return', 'CAGR', 'Sharpe Ratio', 'Prob. Sharpe Ratio',
        'Sortino Ratio', 'Calmar Ratio', 'Max. DD',
        'Avg. Loss', 'Avg. Return', 'Avg. Win', 'Best Day',
        'Worst Day', 'Daily VaR'
    }

    # Set of metrics that are better when lower
    lower_is_better = {
        'Ann. Vol.', 'Avg. Turnover'
    }

    # Set of metrics that should not be converted to percentages
    not_percentage = {'Sharpe Ratio', 'Sortino Ratio', 'Calmar Ratio'}

    for row_label, row_series in metrics_df.iterrows():
        if row_label not in higher_is_better and row_label not in lower_is_better:
            raise ValueError(f"Unexpected row label: {row_label}")

        # Convert and round the values if necessary before finding max/min
        processed_values = [round(100 * val, 3) if row_label not in not_percentage else round(val, 3) for val in
                            row_series]
        max_value = max(processed_values)
        min_value = min(processed_values)

        new_values = []
        for proc_val in processed_values:
            str_val = f"{proc_val:.3f}%" if row_label not in not_percentage else f"{proc_val:.3f}"
            if row_label in higher_is_better:
                if proc_val == max_value:
                    new_values.append(f"{str_val} (Best)")
                elif proc_val == min_value:
                    new_values.append(f"{str_val} (Worst)")
                else:
                    new_values.append(str_val)
            elif row_label in lower_is_better:
                if proc_val == min_value:
                    new_values.append(f"{str_val} (Best)")
                elif proc_val == max_value:
                    new_values.append(f"{str_val} (Worst)")
                else:
                    new_values.append(str_val)

        metrics_df.loc[row_label] = new_values

    return metrics_df


def performance_metrics(portfolio_setups_simple_returns_df,
                         portfolio_setups_excess_simple_returns_df,
                        portfolio_setups_turnover,
                         tc):

    # Filter columns
    # Create empty DataFrames for storing the filtered columns
    filtered_simple_returns_df = pd.DataFrame()
    filtered_excess_simple_returns_df = pd.DataFrame()
    filtered_turnovers_df = pd.DataFrame()

    # Loop through each column name to check the conditions
    for col in portfolio_setups_simple_returns_df.columns:
        if f'tc={tc}' in col:
                new_col_name = col.replace(f'tc={tc}', '').replace(',', '')
                filtered_simple_returns_df[new_col_name] = portfolio_setups_simple_returns_df[col]

    for col in portfolio_setups_excess_simple_returns_df.columns:
        if f'tc={tc}' in col:
                new_col_name = col.replace(f'tc={tc}', '').replace(',', '')
                filtered_excess_simple_returns_df[new_col_name] = portfolio_setups_excess_simple_returns_df[col]

    for col in portfolio_setups_turnover.columns:
        if f'tc={tc}' in col:
                new_col_name = col.replace(f'tc={tc}', '').replace(',', '')
                filtered_turnovers_df[new_col_name] = portfolio_setups_turnover[col]

    # Df to store portfolio metrics
    portfolio_setups_metrics_df = pd.DataFrame(columns=filtered_excess_simple_returns_df.columns)

    # Calculate the cumulative return for each portfolio and add it as a row
    for column_name in filtered_simple_returns_df.columns:
        portfolio_simple_returns_series = filtered_simple_returns_df[column_name]

        portfolio_comp_return = qs.stats.comp(portfolio_simple_returns_series)
        portfolio_setups_metrics_df.at['Cum. Return', column_name] = portfolio_comp_return

    # Calculate the CAGR for each portfolio and add it as a row
    for column_name in filtered_simple_returns_df.columns:
        portfolio_simple_returns_series = filtered_simple_returns_df[column_name]

        portfolio_cagr = qs.stats.cagr(portfolio_simple_returns_series, periods=365)
        portfolio_setups_metrics_df.at['CAGR', column_name] = portfolio_cagr

    # Calculate the Sharpe ratio for each portfolio and add it as a row
    for column_name in filtered_excess_simple_returns_df.columns:
        portfolio_excess_simple_returns_series = filtered_excess_simple_returns_df[column_name]

        portfolio_sharpe_ratio = qs.stats.sharpe(portfolio_excess_simple_returns_series)
        portfolio_setups_metrics_df.at['Sharpe Ratio', column_name] = portfolio_sharpe_ratio

    # Calculate the Probabilistic Sharpe ratio for each portfolio and add it as a row
    for column_name in filtered_excess_simple_returns_df.columns:
        portfolio_excess_simple_returns_series = filtered_excess_simple_returns_df[column_name]

        portfolio_prob_sharpe_ratio = qs.stats.probabilistic_ratio(portfolio_excess_simple_returns_series)
        portfolio_setups_metrics_df.at['Prob. Sharpe Ratio', column_name] = portfolio_prob_sharpe_ratio

    # Calculate the Sortino ratio for each portfolio and add it as a row
    for column_name in filtered_excess_simple_returns_df.columns:
        portfolio_excess_simple_returns_series = filtered_excess_simple_returns_df[column_name]

        portfolio_sortino_ratio = qs.stats.sortino(portfolio_excess_simple_returns_series)
        portfolio_setups_metrics_df.at['Sortino Ratio', column_name] = portfolio_sortino_ratio

    # Calculate the Calmar ratio for each portfolio and add it as a row
    for column_name in filtered_simple_returns_df.columns:
        portfolio_simple_returns_series = filtered_simple_returns_df[column_name]

        portfolio_calmar_ratio = qs.stats.cagr(portfolio_simple_returns_series, periods=365) / abs(qs.stats.max_drawdown(portfolio_simple_returns_series))
        portfolio_setups_metrics_df.at['Calmar Ratio', column_name] = portfolio_calmar_ratio

    # Calculate the max drawdown for each portfolio and add it as a row
    for column_name in filtered_simple_returns_df.columns:
        portfolio_simple_returns_series = filtered_simple_returns_df[column_name]

        portfolio_max_drawdown = qs.stats.max_drawdown(portfolio_simple_returns_series)
        portfolio_setups_metrics_df.at['Max. DD', column_name] = portfolio_max_drawdown

    # Calculate the avg loss for each portfolio and add it as a row
    for column_name in filtered_simple_returns_df.columns:
        portfolio_simple_returns_series = filtered_simple_returns_df[column_name]

        portfolio_avg_loss = qs.stats.avg_loss(portfolio_simple_returns_series)
        portfolio_setups_metrics_df.at['Avg. Loss', column_name] = portfolio_avg_loss

    # Calculate the avg return for each portfolio and add it as a row
    for column_name in filtered_simple_returns_df.columns:
        portfolio_simple_returns_series = filtered_simple_returns_df[column_name]

        portfolio_avg_return = qs.stats.avg_return(portfolio_simple_returns_series)
        portfolio_setups_metrics_df.at['Avg. Return', column_name] = portfolio_avg_return

    # Calculate the avg win for each portfolio and add it as a row
    for column_name in filtered_simple_returns_df.columns:
        portfolio_simple_returns_series = filtered_simple_returns_df[column_name]

        portfolio_avg_win = qs.stats.avg_win(portfolio_simple_returns_series)
        portfolio_setups_metrics_df.at['Avg. Win', column_name] = portfolio_avg_win

    # Calculate the best day for each portfolio and add it as a row
    for column_name in filtered_simple_returns_df.columns:
        portfolio_simple_returns_series = filtered_simple_returns_df[column_name]

        portfolio_best = qs.stats.best(portfolio_simple_returns_series)
        portfolio_setups_metrics_df.at['Best Day', column_name] = portfolio_best

    # Calculate the worst day for each portfolio and add it as a row
    for column_name in filtered_simple_returns_df.columns:
        portfolio_simple_returns_series = filtered_simple_returns_df[column_name]

        portfolio_worst = qs.stats.worst(portfolio_simple_returns_series)
        portfolio_setups_metrics_df.at['Worst Day', column_name] = portfolio_worst

    # Calculate the volatility for each portfolio and add it as a row
    for column_name in filtered_simple_returns_df.columns:
        portfolio_simple_returns_series = filtered_simple_returns_df[column_name]

        portfolio_volatility = qs.stats.volatility(portfolio_simple_returns_series)
        portfolio_setups_metrics_df.at['Ann. Vol.', column_name] = portfolio_volatility

    # Calculate the VaR for each portfolio and add it as a row
    for column_name in filtered_simple_returns_df.columns:
        portfolio_simple_returns_series = filtered_simple_returns_df[column_name]

        portfolio_VaR = qs.stats.value_at_risk(portfolio_simple_returns_series)
        portfolio_setups_metrics_df.at['Daily VaR', column_name] = portfolio_VaR

    # Calculate the turnover for each portfolio and add it as a row
    for column_name in filtered_turnovers_df.columns:
        portfolio_turnover_series = filtered_turnovers_df[column_name]
        portfolio_turnover = portfolio_turnover_series.mean()
        portfolio_setups_metrics_df.at['Avg. Turnover', column_name] = portfolio_turnover

    highlighted_metrics_df = process_and_highlight_values(portfolio_setups_metrics_df)
    highlighted_metrics_df.to_csv(f"../results/metrics_tc{tc}.csv", index=True)

def compute_excess_returns(portfolio_simple_returns_series,
                           risk_free_rate_df):
    # Reindex risk_free_rate_df to match the dates of portfolio_simple_returns_series, then forward-fill
    matching_risk_free_rates_series = risk_free_rate_df['DTB3'].reindex(portfolio_simple_returns_series.index).ffill()

    # For any NaNs at the beginning, use backfill to fill them with the next valid value
    matching_risk_free_rates_series = matching_risk_free_rates_series.bfill()

    # Compute the excess returns
    portfolio_excess_simple_returns_series = portfolio_simple_returns_series - (
            (matching_risk_free_rates_series + 1) ** (1 / 252) - 1)

    # Rename the series to match the original portfolio name
    portfolio_excess_simple_returns_series.name = portfolio_simple_returns_series.name

    return portfolio_excess_simple_returns_series


def full_evaluation(portfolio_setups_simple_returns_df,
                    portfolio_setups_turnover,
                    sp500_simple_returns_df,
                    risk_free_rate_df,
                    vix_prices_df):
    filtered_sp500_simple_returns_df = sp500_simple_returns_df.loc[portfolio_setups_simple_returns_df.index]
    filtered_vix_prices_df = vix_prices_df.loc[portfolio_setups_simple_returns_df.index]

    portfolio_setups_simple_returns_df['S&P 500'] = filtered_sp500_simple_returns_df['S&P 500']
    portfolio_setups_simple_returns_df = portfolio_setups_simple_returns_df[
        ['S&P 500'] + [col for col in portfolio_setups_simple_returns_df.columns if col != 'S&P 500']]

    # Get excess returns
    portfolio_setups_excess_simple_returns_df = pd.DataFrame(columns=portfolio_setups_simple_returns_df.columns)
    for column_name in portfolio_setups_simple_returns_df.columns:
        portfolio_simple_returns_series = portfolio_setups_simple_returns_df[column_name]

        portfolio_excess_simple_returns_series = compute_excess_returns(portfolio_simple_returns_series,
                                                                        risk_free_rate_df)

        # Add portfolio_excess_simple_returns_series to the new DataFrame
        portfolio_setups_excess_simple_returns_df[column_name] = portfolio_excess_simple_returns_series

    # Performance metrics for "tc=5"
    performance_metrics(portfolio_setups_simple_returns_df,
                     portfolio_setups_excess_simple_returns_df,
                    portfolio_setups_turnover,
                     5)

    # Performance metrics for "tc=15"
    performance_metrics(portfolio_setups_simple_returns_df,
                     portfolio_setups_excess_simple_returns_df,
                    portfolio_setups_turnover,
                     15)

    # Performance metrics for "tc=35"
    performance_metrics(portfolio_setups_simple_returns_df,
                     portfolio_setups_excess_simple_returns_df,
                    portfolio_setups_turnover,
                     35)

    # Plot performance for "tc=5"
    plot_performance(portfolio_setups_simple_returns_df,
                     portfolio_setups_excess_simple_returns_df,
                     5)

    # Plot performance for "tc=15"
    plot_performance(portfolio_setups_simple_returns_df,
                     portfolio_setups_excess_simple_returns_df,
                     15)

    # Plot performance for "tc=35"
    plot_performance(portfolio_setups_simple_returns_df,
                     portfolio_setups_excess_simple_returns_df,
                     35)


    plot_vix_and_sp500(filtered_vix_prices_df,
                       filtered_sp500_simple_returns_df)




