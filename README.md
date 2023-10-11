# Incorporating Different Sources of Information for Bayesian Optimal Portfolio Selection

This repository contains code related to the research project 'Incorporating Different Sources of Information for Bayesian Optimal Portfolio Selection'. 

## Research Overview
The associated paper introduces Bayesian inference procedures for tangency portfolios, with a primary focus on deriving a new conjugate prior for portfolio weights. This approach not only enables direct inference about the weights but also seamlessly integrates additional information into the prior specification. Specifically, it automatically incorporates high-frequency returns and the VIX, significantly enhancing the decision-making process for optimal portfolio construction. While the Jeffreys prior is also acknowledged, emphasis is placed on the advantages and practical applications of the conjugate prior. An extensive empirical study reveals that our method, leveraging this conjugate prior, consistently outperforms existing trading strategies in the majority of examined cases.

## Prerequisites

Before you begin, ensure you have met the following requirements:

- **API Keys:**
  - **Alpha Vantage:** Obtain your API key by signing up [here](https://www.alphavantage.co/).
  - **Financial Modeling Prep:** Sign up [here](https://financialmodelingprep.com/developer/docs/) to get your API key.

### Manual Data Collection
In addition to the data automatically gathered through the APIs, there is some data you'll need to collect and place manually:

- **S&P 500 Historical Components:**
  - Download `S&P 500 Historical Components & Changes(04-16-2023).csv` from [this repository](https://github.com/fja05680/sp500).
  - Place the downloaded file in `data/s500_components`.
- **Risk-Free Rate (DTB3):**
  - Download the DTB3 data from [FRED](https://fred.stlouisfed.org/series/DTB3).
  - Save the data as `DTB3.csv` and place it in `data/risk_free_rate`.

## Setup

### Environment Variables

1. Create a `.env` file in your project directory.
2. Insert your API keys into the `.env` file as shown below:

    ```
    ALPHA_VANTAGE_KEY=your_alpha_vantage_api_key
    FINANCIAL_MODELING_PREP_KEY=your_financial_modeling_prep_api_key
    ```

    Replace `your_alpha_vantage_api_key` and `your_financial_modeling_prep_api_key` with your actual keys.

### Dependencies Installation

To install dependencies, navigate to the project directory in your terminal and run:

    ```shell
    pip3 install -r requirements.txt
    ```

## Running the Project

After setting up your API keys, downloading the necessary data, and installing dependencies, you can execute the project. Running the main script will download relevant data, perform portfolio backtesting for the period between January 3, 2008, and June 30, 2023 (as discussed in the paper), save the results, generate performance metrics, and create visualizations to aid in understanding the portfolio's performance over time. Navigate to the project directory in your terminal and execute the main script:

    ```shell
    python3 main.py
    ```
