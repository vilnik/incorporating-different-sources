# Incorporating different sources of information for Bayesian optimal portfolio selection

This repository contains code related to the research project 'Incorporating different sources of information for Bayesian optimal portfolio selection' 

## Prerequisites

Before you begin, ensure you have met the following requirements:

- **API Keys:**
  - **Alpha Vantage:** Obtain your API key by signing up [here](https://www.alphavantage.co/).
  - **Financial Modeling Prep:** Sign up [here](https://financialmodelingprep.com/developer/docs/) to get your API key.

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

Install dependecies:

    ```shell
    pip3 install -r requirements.txt
    ```

## Running the Project

With your API keys in place and dependencies installed, navigate to the project directory in your terminal and run:

    ```shell
    python3 main.py
    ``

