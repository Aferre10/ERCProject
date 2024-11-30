import pandas as pd
import numpy as np
from typing import Tuple

# ----------------------------- Input Data -----------------------------
file_paths = {
    "Equities": "https://raw.githubusercontent.com/beasilvacosta/Quarm-Data-/main/Project_Equities.xlsx",
    "Fixed Income": "https://raw.githubusercontent.com/beasilvacosta/Quarm-Data-/main/Project_FixedIncome.xlsx",
    "Real Estate": "https://raw.githubusercontent.com/beasilvacosta/Quarm-Data-/main/Project_RealEstate.xlsx",
    "Commodities": "https://raw.githubusercontent.com/beasilvacosta/Quarm-Data-/main/Project_Commodities.xlsx",
    "Alternative Investments": "https://raw.githubusercontent.com/beasilvacosta/Quarm-Data-/main/Project_AlternativeInvestments.xlsx",
    "Money Market": "https://raw.githubusercontent.com/beasilvacosta/Quarm-Data-/main/Project_MoneyMarket.xlsx",
    "Currencies": "https://raw.githubusercontent.com/beasilvacosta/Quarm-Data-/main/Project_Currencies.xlsx"
}


rf_path = "https://raw.githubusercontent.com/beasilvacosta/Quarm-Data-/main/NEW_RF_ERC.xlsx"
benchmarks_path = "https://raw.githubusercontent.com/beasilvacosta/Quarm-Data-/main/NEW_BMK_ERC.xlsx"


# ----------------------------- Functions -----------------------------
def simple_returns(df):
    """Calculate daily returns."""
    return df.pct_change()

def calculate_dynamic_equal_weights(daily_returns):
    """Calculate dynamically adjusted equal weights each day."""
    weights = daily_returns.copy()
    weights[:] = 1 / daily_returns.shape[1]
    return weights

def calculate_dynamic_portfolio_returns(daily_returns, dynamic_weights):
    """Calculate portfolio returns using dynamically adjusted daily weights."""
    portfolio_returns = (daily_returns * dynamic_weights).sum(axis=1)
    return portfolio_returns

def compute_performance_metrics(portfolios, risk_free_rate_series, market_returns):
    """Calculate performance metrics for each portfolio."""
    performance_metrics = {
        'Category': [], 'Annualized Return (%)': [], 'Annualized Volatility (%)': [],

    }

    for category, portfolio_df in portfolios.items():
        daily_returns = portfolio_df["Daily Equal Weighted Return"].dropna()
        risk_free_rate = risk_free_rate_series.loc[daily_returns.index].dropna()
        daily_returns = daily_returns.loc[risk_free_rate.index]

        annualized_return = daily_returns.mean() * 252 * 100
        annualized_volatility = daily_returns.std() * np.sqrt(252) * 100
        excess_return = daily_returns - (risk_free_rate / 100)
        #sharpe_ratio = (excess_return.mean() / daily_returns.std()) * np.sqrt(252)

        aligned_market_returns = market_returns.loc[daily_returns.index].dropna()
        beta = (
            np.cov(daily_returns, aligned_market_returns)[0, 1] / np.var(aligned_market_returns)
            if len(aligned_market_returns) == len(daily_returns)
            else np.nan
        )
        downside_risk = np.sqrt(np.mean(np.minimum(0, excess_return) ** 2)) * np.sqrt(252)


        performance_metrics['Category'].append(category)
        performance_metrics['Annualized Return (%)'].append(annualized_return)
        performance_metrics['Annualized Volatility (%)'].append(annualized_volatility)


    return pd.DataFrame(performance_metrics)

def generate_asset_class_composition(file_paths):
    """Generate a consolidated table of assets by class."""
    assets_in_classes = {}
    for category, file_path in file_paths.items():
        df = pd.read_excel(file_path)
        if "Date" not in df.columns:
            print(f"Error: 'Date' column missing in {file_path}")
            continue

        df = df.set_index("Date")
        assets_in_classes[category] = list(df.columns)  # Store the assets for the category

    # Convert to a DataFrame for display
    assets_table = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in assets_in_classes.items()]))
    return assets_table.fillna('')

def process_equal_weighted_portfolios():
    """Process and calculate equally weighted portfolios for all asset classes."""
    equal_weighted_portfolios = {}
    for category, file_path in file_paths.items():
        df = pd.read_excel(file_path)
        if "Date" not in df.columns:
            print(f"Error: 'Date' column missing in {file_path}")
            continue
        df = df.set_index("Date")
        daily_returns = simple_returns(df).dropna()
        dynamic_weights = calculate_dynamic_equal_weights(daily_returns)
        portfolio_returns = calculate_dynamic_portfolio_returns(daily_returns, dynamic_weights)
        equal_weighted_portfolios[category] = pd.DataFrame(portfolio_returns, columns=["Daily Equal Weighted Return"])
    return equal_weighted_portfolios

def load_external_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load external data for risk-free rate and benchmarks."""
    RF = pd.read_excel(rf_path, index_col='Date')
    Benchs = pd.read_excel(benchmarks_path, index_col='Date')
    RF.index, Benchs.index = pd.to_datetime(RF.index), pd.to_datetime(Benchs.index)
    return RF, Benchs

# ----------------------------- Workflow -----------------------------
# Load external data
RF, Benchs = load_external_data()
risk_free_rate_series = RF['RF6M'] / 252
market_returns = (Benchs["AQR MULTI-ASSET FUND I"] / Benchs["AQR MULTI-ASSET FUND I"].shift(1) - 1).dropna()

# Process portfolios
equal_weighted_portfolios = process_equal_weighted_portfolios()

# Compute performance metrics
performance_df = compute_performance_metrics(equal_weighted_portfolios, risk_free_rate_series, market_returns)

# Generate asset class composition table
assets_table = generate_asset_class_composition(file_paths)

# Display results
print("\nPerformance Metrics:")
print(performance_df.to_string(index=False))

print("\nConsolidated Table of Assets by Class:")
print(assets_table)