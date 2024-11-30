import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations
from scipy.optimize import minimize, LinearConstraint
from typing import Tuple, Dict
import plotly.graph_objects as go
import datetime

# ----------------------------- Configuration -----------------------------
rebalancing_freq_months = 6  # Change this value to 1, 3, or 6 as needed

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



# ----------------------------- Utility Functions -----------------------------
def simple_returns(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate daily returns."""
    return df.pct_change()


def calculate_dynamic_equal_weights(daily_returns: pd.DataFrame) -> pd.DataFrame:
    """Calculate dynamically adjusted equal weights each day."""
    weights = daily_returns.copy()
    weights[:] = 1 / daily_returns.shape[1]  # Equal weight each day
    return weights


def calculate_dynamic_portfolio_returns(daily_returns: pd.DataFrame, dynamic_weights: pd.DataFrame) -> pd.Series:
    """Calculate portfolio returns using dynamically adjusted daily weights."""
    portfolio_returns = (daily_returns * dynamic_weights).sum(axis=1)
    return portfolio_returns


def load_external_data2() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load external data for risk-free rates and benchmarks."""
    RF = pd.read_excel(rf_path, index_col="Date")
    Benchs = pd.read_excel(benchmarks_path, index_col="Date")
    RF.index = pd.to_datetime(RF.index)
    Benchs.index = pd.to_datetime(Benchs.index)
    return RF, Benchs


def process_equal_weighted_portfolios(file_paths: Dict[str, str]) -> Dict[str, pd.DataFrame]:
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


def compute_performance_metrics(equal_weighted_portfolios: Dict[str, pd.DataFrame],
                                risk_free_rate_series: pd.Series,
                                market_returns: pd.Series) -> pd.DataFrame:
    """Compute performance metrics for portfolios."""
    performance_metrics = {
        'Category': [],
        'Annualized Return (%)': [],
        'Annualized Volatility (%)': []
    }

    for category, portfolio_df in equal_weighted_portfolios.items():
        daily_returns = portfolio_df["Daily Equal Weighted Return"].dropna()

        annualized_return = daily_returns.mean() * 252 * 100
        annualized_volatility = daily_returns.std() * np.sqrt(252) * 100

        performance_metrics['Category'].append(category)
        performance_metrics['Annualized Return (%)'].append(annualized_return)
        performance_metrics['Annualized Volatility (%)'].append(annualized_volatility)

    return pd.DataFrame(performance_metrics)


def generate_asset_class_composition(file_paths: Dict[str, str]) -> pd.DataFrame:
    """Generate a consolidated table of assets by class."""
    assets_in_classes = {}
    for category, file_path in file_paths.items():
        df = pd.read_excel(file_path)
        if "Date" not in df.columns:
            print(f"Error: 'Date' column missing in {file_path}")
            continue
        df = df.set_index("Date")
        assets_in_classes[category] = list(df.columns)

    # Convert to DataFrame for display
    assets_table = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in assets_in_classes.items()]))
    return assets_table.fillna('')


def plot_cumulative_returns(equal_weighted_portfolios: Dict[str, pd.DataFrame]):
    """Generate and display cumulative returns plot."""
    plt.figure(figsize=(14, 8))
    start_date = min([portfolio_df.index.min() for portfolio_df in equal_weighted_portfolios.values()])
    for category, portfolio_df in equal_weighted_portfolios.items():
        cumulative_returns = (1 + portfolio_df["Daily Equal Weighted Return"]).cumprod()
        cumulative_returns = cumulative_returns[cumulative_returns.index >= start_date]
        plt.plot(cumulative_returns, label=f'{category}')
    plt.title('Cumulative Returns of All Portfolios')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Return')
    plt.legend(loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    plt.xlim(left=start_date)
    plt.show()


# ----------------------------- Main Workflow -----------------------------
if __name__ == "__main__":
    # Load external data
    RF, Benchs = load_external_data()

    # Risk-free rate and market returns
    risk_free_rate_series = RF[f'RF{rebalancing_freq_months}M'] / 100 / 252
    market_prices = Benchs["AQR MULTI-ASSET FUND I"]
    market_returns = market_prices.pct_change().dropna()

    # Process equally weighted portfolios
    equal_weighted_portfolios = process_equal_weighted_portfolios(file_paths)

    # Compute performance metrics
    performance_df = compute_performance_metrics(equal_weighted_portfolios, risk_free_rate_series, market_returns)
    print("\nPerformance Metrics:")
    print(performance_df.to_string(index=False))

    # Plot cumulative returns
    plot_cumulative_returns(equal_weighted_portfolios)

    # Generate and display asset class composition
    assets_table = generate_asset_class_composition(file_paths)
    print("\nConsolidated Table of Assets by Class:")
    print(assets_table)

def compute_class_correlation_matrix(file_paths: Dict[str, str]) -> pd.DataFrame:
    """
    Compute and return the correlation matrix between asset classes.

    Args:
        file_paths: Dictionary containing file paths for each asset class.

    Returns:
        A DataFrame representing the correlation matrix between asset classes.
    """
    average_returns_per_class = {}

    for category, file_path in file_paths.items():
        df = pd.read_excel(file_path)
        if "Date" not in df.columns:
            print(f"Error: 'Date' column missing in {file_path}")
            continue

        df = df.set_index("Date")
        daily_returns = simple_returns(df).dropna()  # Calculate daily returns
        daily_returns.index = pd.to_datetime(daily_returns.index)

        # Compute the average daily return for the asset class
        average_returns_per_class[category] = daily_returns.mean(axis=1)

    # Combine the average returns into a single DataFrame
    combined_class_returns = pd.DataFrame(average_returns_per_class).dropna()

    # Compute the correlation matrix between asset classes
    class_correlation_matrix = combined_class_returns.corr()

    return class_correlation_matrix

def plot_correlation_matrix(correlation_matrix: pd.DataFrame):
    """
    Plot a heatmap for the correlation matrix.

    Args:
        correlation_matrix: DataFrame containing the correlation matrix.

    Returns:
        A Matplotlib figure object of the heatmap.
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(correlation_matrix, cmap="coolwarm", annot=True, ax=ax)
    ax.set_title("Correlation Matrix Between Asset Classes")
    plt.tight_layout()
    plt.show()
    return fig

# --- Workflow Additions ---
if __name__ == "__main__":
    # Compute and display the correlation matrix
    correlation_matrix = compute_class_correlation_matrix(file_paths)
    print("\nCorrelation Matrix Between Asset Classes:")
    print(correlation_matrix)

    # Plot the correlation matrix
    plot_correlation_matrix(correlation_matrix)

##------ERC-General---------------------------------
# Define fees for each sector (percentage fees as decimals)
fees_per_sector = {
    "Equities": 0.001,  # 0.10%
    "Fixed Income": 0.0005,  # 0.05%
    "Real Estate": 0.005,  # 0.50%
    "Commodities": 0.002,  # 0.20%
    "Alternative Investments": 0.005,  # 0.50%
    "Money Market": 0.0002,  # 0.02%
    "Currencies": 0.001  # 0.10%
}

def generate_fees_table(fees_per_sector: Dict[str, float]) -> pd.DataFrame:
    """
    Generate a DataFrame showing fees for each sector.

    Args:
        fees_per_sector: A dictionary with sectors as keys and fees (as decimals) as values.

    Returns:
        A DataFrame with sectors and their corresponding fees in percentage format.
    """
    fees_table = pd.DataFrame(list(fees_per_sector.items()), columns=["Sector", "Fees (%)"])
    fees_table["Fees (%)"] = fees_table["Fees (%)"] * 100  # Convert fees to percentage format
    return fees_table

# --- Workflow Addition ---
if __name__ == "__main__":
    # Generate and display fees table
    fees_table = generate_fees_table(fees_per_sector)
    print("\nFees by Sector:")
    print(fees_table.to_string(index=False))

# ----------------------------- Risk Contribution and ERC Functions -----------------------------

def risk_contribution(weights: np.ndarray, cov_matrix: np.ndarray) -> np.ndarray:
    """
    Calculate the absolute risk contribution for each asset in a portfolio.

    Args:
        weights (np.ndarray): Weights of the assets in the portfolio.
        cov_matrix (np.ndarray): Covariance matrix of asset returns.

    Returns:
        np.ndarray: Absolute risk contribution of each asset.
    """
    portfolio_volatility = np.sqrt(weights @ cov_matrix @ weights)
    marginal_contributions = cov_matrix @ weights
    abs_risk_contributions = weights * marginal_contributions / portfolio_volatility
    return abs_risk_contributions

def objective_erc(weights: np.ndarray, cov_matrix: np.ndarray) -> float:
    """
    Objective function to minimize for Equally Risk-Contributed (ERC) portfolios.

    Args:
        weights (np.ndarray): Weights of the assets in the portfolio.
        cov_matrix (np.ndarray): Covariance matrix of asset returns.

    Returns:
        float: Sum of squared differences between risk contributions and target risk contributions.
    """
    abs_contrib = risk_contribution(weights, cov_matrix)
    target_contrib = np.mean(abs_contrib)
    return np.sum((abs_contrib - target_contrib) ** 2)

def calculate_cov_matrix(returns: pd.DataFrame) -> np.ndarray:
    """
    Calculate the covariance matrix for the given returns.

    Args:
        returns (pd.DataFrame): DataFrame containing the asset returns.

    Returns:
        np.ndarray: Covariance matrix of the asset returns.
    """
    deviations = returns - returns.mean()
    return deviations.T @ deviations / len(returns)

def generate_erc_portfolios(equal_weighted_portfolios: dict, min_classes: int = 3, start_date: str = '2017-01-31', rebalancing_freq_months: int = 6) -> pd.DataFrame:
    """
    Generate ERC portfolios and compute their metrics.

    Args:
        equal_weighted_portfolios (dict): Dictionary containing equal-weighted portfolios with daily returns.
        min_classes (int): Minimum number of asset classes for portfolio combinations.
        start_date (str): Starting date for rebalancing.
        rebalancing_freq_months (int): Rebalancing frequency in months.

    Returns:
        pd.DataFrame: DataFrame containing the ERC portfolio details.
    """
    asset_classes = list(equal_weighted_portfolios.keys())
    all_portfolios = []

    # Iterate over combinations of asset classes
    for num_assets in range(min_classes, len(asset_classes) + 1):
        for selected_classes in combinations(asset_classes, num_assets):
            print(f"\nProcessing portfolio: {selected_classes}")

            # Combine returns for selected asset classes
            combined_daily_returns = pd.concat(
                [equal_weighted_portfolios[cls]["Daily Equal Weighted Return"] for cls in selected_classes],
                axis=1
            )
            combined_daily_returns.columns = selected_classes

            # Define rebalancing dates
            rebalancing_dates = pd.date_range(start=start_date, end=combined_daily_returns.index[-1], freq=f'{rebalancing_freq_months}M')

            for rebalance_date in rebalancing_dates:
                # Rolling 3-year window
                window_start_date = rebalance_date - pd.DateOffset(years=3)

                if window_start_date >= combined_daily_returns.index.min():
                    window_returns = combined_daily_returns.loc[window_start_date:rebalance_date].dropna()

                    if not window_returns.empty and window_returns.shape[1] == num_assets:
                        cov_matrix = calculate_cov_matrix(window_returns) * 252

                        # Optimize ERC weights
                        x0_ew = np.ones(num_assets) / num_assets
                        lin_constraint = LinearConstraint(np.ones(num_assets), 1, 1)
                        bounds = [(0, 1) for _ in range(num_assets)]

                        res = minimize(
                            objective_erc,
                            x0_ew,
                            args=(cov_matrix,),
                            method='SLSQP',
                            constraints=[lin_constraint],
                            bounds=bounds
                        )

                        if res.success:
                            weights_erc = res.x / np.sum(res.x)

                            # Calculate fees
                            fees = sum(
                                weights_erc[i] * fees_per_sector.get(selected_classes[i], 0)
                                for i in range(num_assets)
                            )

                            # Out-of-sample returns
                            oos_period_start = rebalance_date + pd.Timedelta(days=1)
                            oos_period_end = rebalance_date + pd.DateOffset(months=rebalancing_freq_months)
                            oos_returns = combined_daily_returns.loc[oos_period_start:oos_period_end].dropna()

                            if not oos_returns.empty:
                                oos_portfolio_returns = oos_returns @ weights_erc
                                annualized_return = np.mean(oos_portfolio_returns) * 252 - fees
                                annualized_volatility = np.std(oos_portfolio_returns) * np.sqrt(252)

                                abs_risk_contrib = risk_contribution(weights_erc, cov_matrix)

                                # Store portfolio data
                                portfolio_data = {
                                    "Rebalance Date": rebalance_date,
                                    "Portfolio Volatility (Annualized)": annualized_volatility,
                                    "Portfolio Return (Annualized)": annualized_return,
                                    "Fees at Rebalancing": fees,
                                    "Absolute Risk Contribution": abs_risk_contrib.tolist(),
                                    "Weights": weights_erc.tolist(),
                                    "Selected Classes": selected_classes
                                }
                                all_portfolios.append(portfolio_data)

    # Convert to DataFrame
    return pd.DataFrame(all_portfolios)




# ----------------------------- Risk Contribution Table -----------------------------

def relative_risk_contribution(abs_risk_contrib: np.ndarray) -> np.ndarray:
    """
    Compute the relative risk contribution for each asset.

    Args:
        abs_risk_contrib (np.ndarray): Absolute risk contributions for the portfolio.

    Returns:
        np.ndarray: Relative risk contributions.
    """
    return abs_risk_contrib / np.sum(abs_risk_contrib)

def generate_risk_contribution_table(erc_portfolios_df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate a detailed risk contribution table for each portfolio.

    Args:
        erc_portfolios_df (pd.DataFrame): DataFrame containing ERC portfolio details.

    Returns:
        pd.DataFrame: DataFrame containing detailed risk contributions.
    """
    risk_contributions = []

    for _, row in erc_portfolios_df.iterrows():
        portfolio_name = ", ".join(row["Selected Classes"])
        weights = np.array(row["Weights"])
        abs_contrib = np.array(row["Absolute Risk Contribution"])
        rel_contrib = relative_risk_contribution(abs_contrib)
        fees = row["Fees at Rebalancing"]

        for i, asset in enumerate(row["Selected Classes"]):
            risk_contributions.append({
                "Portfolio": portfolio_name,
                "Asset": asset,
                "Weight": weights[i],
                "Marginal Risk Contribution": abs_contrib[i] / weights[i] if weights[i] != 0 else 0,
                "Absolute Risk Contribution": abs_contrib[i],
                "Relative Risk Contribution": rel_contrib[i],
                "Fees at Rebalancing": fees
            })

    # Create a DataFrame
    risk_contribution_df = pd.DataFrame(risk_contributions)
    return risk_contribution_df


# ----------------------------- Yearly and Final Metrics with Fees -----------------------------

def calculate_yearly_and_final_metrics_with_fees(
    erc_portfolios_df: pd.DataFrame,
    equal_weighted_portfolios: dict,
    months: int,
    risk_free_rate_series: pd.Series = None,
    benchs_daily_returns: pd.DataFrame = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Calculate yearly and final portfolio metrics, accounting for fees.

    Args:
        erc_portfolios_df (pd.DataFrame): DataFrame containing ERC portfolio details.
        equal_weighted_portfolios (dict): Equal-weighted portfolios by category.
        months (int): Rebalancing period in months.
        risk_free_rate_series (pd.Series): Risk-free rate series for calculating excess returns.
        benchs_daily_returns (pd.DataFrame): Daily returns of the benchmark.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Final metrics DataFrame and yearly metrics DataFrame.
    """
    yearly_metrics = {}
    final_metrics = []

    # Group by unique portfolio compositions
    portfolios_grouped = erc_portfolios_df.groupby("Selected Classes")

    for selected_classes, group in portfolios_grouped:
        aggregated_returns = []

        for _, row in group.iterrows():
            rebalance_date = row["Rebalance Date"]
            weights_erc = np.array(row["Weights"])
            fees = row["Fees at Rebalancing"]

            # Combine the daily returns for the selected asset classes
            combined_daily_returns = pd.concat(
                [equal_weighted_portfolios[cls]["Daily Equal Weighted Return"] for cls in selected_classes],
                axis=1
            )
            combined_daily_returns.columns = selected_classes

            # Define the OOS period for this rebalance date
            oos_period_start = rebalance_date + pd.Timedelta(days=1)
            oos_period_end = rebalance_date + pd.DateOffset(months=months)

            # Get the OOS returns for this rebalance period
            oos_returns = combined_daily_returns.loc[oos_period_start:oos_period_end].dropna()

            if not oos_returns.empty:
                # Compute portfolio returns for this period
                portfolio_returns = oos_returns @ weights_erc

                # Apply fees: Adjust the returns by subtracting fees equally over the period
                daily_fee_adjustment = fees / len(portfolio_returns)
                adjusted_returns = portfolio_returns - daily_fee_adjustment

                aggregated_returns.append(adjusted_returns)

                # Add to yearly data
                for date, ret in portfolio_returns.items():
                    year = date.year
                    if year not in yearly_metrics:
                        yearly_metrics[year] = {"returns": []}
                    yearly_metrics[year]["returns"].append(ret)

        if aggregated_returns:
            # Concatenate all aggregated returns across rebalance periods
            full_period_returns = pd.concat(aggregated_returns)

            # Calculate final metrics (fees already accounted for in returns)
            total_return = (1 + full_period_returns).prod() - 1
            annualized_return = np.mean(full_period_returns) * 252
            annualized_volatility = np.std(full_period_returns) * np.sqrt(252)

            # Risk-free rate (average over period)
            if risk_free_rate_series is not None:
                rf_full = risk_free_rate_series.loc[oos_period_start:oos_period_end].mean() / 100
            else:
                rf_full = 0

            # Sharpe Ratio
            excess_return = full_period_returns - rf_full / 252
            sharpe_ratio = excess_return.mean() / full_period_returns.std() if full_period_returns.std() != 0 else np.nan

            # Sortino Ratio
            downside_risk = np.sqrt(((full_period_returns[full_period_returns < 0]) ** 2).mean()) * np.sqrt(252)
            sortino_ratio = excess_return.mean() / downside_risk if downside_risk != 0 else np.nan

            # Calculate Beta
            if benchs_daily_returns is not None:
                market_returns = benchs_daily_returns["AQR MULTI-ASSET FUND I"].loc[oos_period_start:oos_period_end]
                aligned_portfolio_returns, aligned_market_returns = full_period_returns.align(market_returns, join='inner')
                if len(aligned_market_returns) > 1:
                    covariance = np.cov(aligned_portfolio_returns, aligned_market_returns)[0, 1]
                    market_variance = aligned_market_returns.var()
                    beta = covariance / market_variance if market_variance != 0 else np.nan
                else:
                    beta = np.nan
            else:
                beta = np.nan

            # Maximum Drawdown
            cumulative_returns = (1 + full_period_returns).cumprod()
            max_drawdown = (cumulative_returns / cumulative_returns.cummax() - 1).min() * 100

            # Cumulative Return
            cumulative_return = (1 + full_period_returns).prod()

            # Store final metrics
            final_metrics.append({
                "Selected Classes": selected_classes,
                "Final Annualized Return": annualized_return,
                "Final Annualized Volatility": annualized_volatility,
                "Sharpe Ratio": sharpe_ratio,
                "Beta": beta,
                "Sortino Ratio": sortino_ratio,
                "Max Drawdown (%)": max_drawdown,
                "Cumulative Return": cumulative_return,
                "Total Fees": fees
            })

    # Compute yearly metrics
    yearly_results = []
    for year, data in yearly_metrics.items():
        yearly_returns = pd.Series(data["returns"])
        yearly_annualized_return = np.mean(yearly_returns) * 252
        yearly_annualized_volatility = np.std(yearly_returns) * np.sqrt(252)
        yearly_results.append({
            "Year": year,
            "Annualized Return": yearly_annualized_return,
            "Annualized Volatility": yearly_annualized_volatility
        })

    # Create DataFrames
    final_metrics_df = pd.DataFrame(final_metrics)
    yearly_metrics_df = pd.DataFrame(yearly_results)

    return final_metrics_df, yearly_metrics_df



# ----------------------------- Daily Returns with Fees -----------------------------

def calculate_daily_returns_with_fees(erc_portfolios_df, equal_weighted_portfolios, start_date="2017-01-31",
                                      end_date="2024-10-28", months=rebalancing_freq_months):
    # Create a date range from start_date to end_date
    date_range = pd.date_range(start=start_date, end=end_date, freq='B')  # 'B' ensures weekdays only

    # Initialize an empty list to store daily returns for all portfolios
    portfolio_daily_returns = {selected_classes: [] for selected_classes in
                               erc_portfolios_df["Selected Classes"].unique()}

    # Loop through each unique portfolio composition
    for selected_classes in erc_portfolios_df["Selected Classes"].unique():
        # Filter the portfolio compositions for the current selected_classes
        portfolio_group = erc_portfolios_df[erc_portfolios_df["Selected Classes"] == selected_classes]

        # Create a list to store the daily returns for this specific portfolio
        daily_returns = pd.Series(index=date_range, dtype=float)

        # Loop through each row in the portfolio group
        for _, row in portfolio_group.iterrows():
            rebalance_date = row["Rebalance Date"]
            weights_erc = np.array(row["Weights"])
            fees = row["Fees at Rebalancing"]

            # Combine the daily returns for the selected asset classes from the equally weighted portfolios
            combined_daily_returns = pd.concat(
                [equal_weighted_portfolios[cls]["Daily Equal Weighted Return"] for cls in selected_classes],
                axis=1
            )
            combined_daily_returns.columns = selected_classes

            # Define the rebalancing period (from rebalance_date to the next rebalance)
            next_rebalance_date = rebalance_date + pd.DateOffset(months=rebalancing_freq_months)
            rebalancing_period = pd.date_range(rebalance_date, next_rebalance_date, freq='B')

            # Apply the fee adjustment: fee per day for the period until the next rebalance
            fee_per_day = fees / len(rebalancing_period)

            # Loop through each day in the rebalancing period
            for date in rebalancing_period:
                if date in combined_daily_returns.index:
                    # Calculate the daily return for the portfolio: weighted sum of the asset class returns
                    daily_portfolio_return = (combined_daily_returns.loc[date] @ weights_erc)

                    # Adjust for fees by subtracting the fee per day from the return
                    daily_portfolio_return_with_fee = daily_portfolio_return - fee_per_day

                    # Store the daily return with fee adjustment
                    daily_returns[date] = daily_portfolio_return_with_fee

        # Store the daily returns for this portfolio
        portfolio_daily_returns[selected_classes] = daily_returns

    # Create a DataFrame from the daily returns of all portfolios
    portfolio_daily_returns_df = pd.DataFrame(portfolio_daily_returns)

    return portfolio_daily_returns_df







# ----------------------------- Portfolio Analysis and Visualization -----------------------------

def analyze_and_plot_portfolios(final_metrics_df: pd.DataFrame):
    """
    Analyze portfolios based on annualized returns and plot the relationship
    between annualized return and volatility.

    Args:
        final_metrics_df (pd.DataFrame): DataFrame containing portfolio performance metrics.

    Returns:
        dict: Contains lists of positive and negative annualized return portfolios.
    """
    # Filter portfolios with positive and negative Annualized Return
    positive_annualized_return = final_metrics_df[final_metrics_df['Final Annualized Return'] > 0]
    negative_annualized_return = final_metrics_df[final_metrics_df['Final Annualized Return'] < 0]

    # List of portfolio names with positive Annualized Return
    positive_portfolios = positive_annualized_return['Selected Classes'].tolist()

    # List of portfolio names with negative Annualized Return
    negative_portfolios = negative_annualized_return['Selected Classes'].tolist()

    # Print the results
    print("Portfolios with Positive Annualized Return:")
    print(positive_portfolios)
    print(f"Count: {len(positive_portfolios)}")
    print("\nPortfolios with Negative Annualized Return:")
    print(negative_portfolios)
    print(f"Count: {len(negative_portfolios)}")

    # Extract the necessary columns: 'Annualized Return' and 'Annualized Volatility'
    x = final_metrics_df['Final Annualized Volatility']
    y = final_metrics_df['Final Annualized Return']
    portfolio_names = final_metrics_df['Selected Classes']
    colors = ['red' if return_val < 0 else 'blue' for return_val in y]

    # Plotting
    plt.figure(figsize=(15, 6))
    plt.scatter(x, y, c=colors, marker='o')

    # Label the axes
    plt.xlabel('Annualized Volatility')
    plt.ylabel('Annualized Return')

    # Title of the plot
    plt.title('Annualized Return vs. Annualized Volatility for Each Portfolio')

    # Show grid for easier visualization
    plt.grid(True)

    # Show the plot
    plt.tight_layout()
    plt.show()

    return {
        "positive_portfolios": positive_portfolios,
        "negative_portfolios": negative_portfolios
    }


def generate_erc_portfolio(equal_weighted_portfolios, start_date, end_date, rebalancing_freq_months, investor_selected_assets):
    """
    Generate an ERC portfolio based on investor preferences.

    Args:
        equal_weighted_portfolios (dict): Equal-weighted portfolios for asset classes.
        start_date (str): Start date for portfolio analysis.
        end_date (str): End date for portfolio analysis.
        rebalancing_freq_months (int): Frequency of portfolio rebalancing (in months).
        investor_selected_assets (list): List of asset classes selected by the investor.

    Returns:
        pd.DataFrame: DataFrame containing ERC portfolio details for the selected configuration.
    """
    selected_classes = investor_selected_assets
    num_assets = len(selected_classes)
    all_portfolios = []

    # Combine returns for selected asset classes
    combined_daily_returns = pd.concat(
        [equal_weighted_portfolios[cls]["Daily Equal Weighted Return"] for cls in selected_classes],
        axis=1
    )
    combined_daily_returns.columns = selected_classes

    # Define rebalancing dates starting from start_date
    rebalancing_dates = pd.date_range(
        start=start_date,
        end=end_date,
        freq=pd.DateOffset(months=rebalancing_freq_months)
    )

    for rebalance_date in rebalancing_dates:
        # Rolling 3-year window
        window_start_date = rebalance_date - pd.DateOffset(years=3)

        if window_start_date >= combined_daily_returns.index.min():
            # Get in-sample window returns
            window_returns = combined_daily_returns.loc[window_start_date:rebalance_date].dropna()

            if not window_returns.empty and window_returns.shape[1] == num_assets:
                cov_matrix = calculate_cov_matrix(window_returns) * 252

                # Optimize ERC weights
                x0_ew = np.ones(num_assets) / num_assets
                lin_constraint = LinearConstraint(np.ones(num_assets), 1, 1)
                bounds = [(0, 1) for _ in range(num_assets)]

                res = minimize(
                    objective_erc,
                    x0_ew,
                    args=(cov_matrix,),
                    method='SLSQP',
                    constraints=[lin_constraint],
                    bounds=bounds
                )

                if res.success:
                    weights_erc = res.x / np.sum(res.x)

                    # Calculate fees for the portfolio
                    fees = sum(
                        weights_erc[i] * fees_per_sector.get(selected_classes[i], 0)
                        for i in range(num_assets)
                    )

                    # Out-of-sample 6-month returns
                    oos_period_start = rebalance_date + pd.Timedelta(days=1)
                    oos_period_end = rebalance_date + pd.DateOffset(months=6)
                    oos_returns = combined_daily_returns.loc[oos_period_start:oos_period_end].dropna()

                    if not oos_returns.empty:
                        oos_portfolio_returns = oos_returns @ weights_erc
                        annualized_return = np.mean(oos_portfolio_returns) * 252 - fees  # Subtract fees
                        annualized_volatility = np.std(oos_portfolio_returns) * np.sqrt(252)

                        abs_risk_contrib = risk_contribution(weights_erc, cov_matrix)

                        # Store portfolio data
                        portfolio_data = {
                            "Rebalance Date": rebalance_date,
                            "Net of Fees Portfolio Return (Annualized)": annualized_return,
                            "Portfolio Volatility (Annualized)": annualized_volatility,
                            "Fees at Rebalancing": fees,
                            "Absolute Risk Contribution": abs_risk_contrib.tolist(),
                            "Weights": weights_erc.tolist(),
                            "Selected Classes": selected_classes
                        }
                        all_portfolios.append(portfolio_data)

    # Create DataFrame
    erc_portfolio_df = pd.DataFrame(all_portfolios)
    return erc_portfolio_df

def calculate_yearly_and_final_metrics_with_fees_for_portfolio(
    erc_portfolio_df,
    equal_weighted_portfolios,
    selected_assets,
    months=6,  # Default to 6 months if not provided
    risk_free_rate_series=None,
    benchs_daily_returns=None
):
    """
    Calculate yearly and final portfolio metrics, accounting for fees and selected assets.

    Args:
        erc_portfolio_df (pd.DataFrame): DataFrame containing ERC portfolio details.
        equal_weighted_portfolios (dict): Equal-weighted portfolios by category.
        selected_assets (list): List of investor-selected asset classes.
        months (int): Rebalancing period in months.
        risk_free_rate_series (pd.Series, optional): Risk-free rate series for calculating excess returns.
        benchs_daily_returns (pd.DataFrame, optional): Benchmark daily returns.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Final metrics DataFrame and yearly metrics DataFrame.
    """
    yearly_metrics = {}
    aggregated_returns = []
    fees_total = 0

    for _, row in erc_portfolio_df.iterrows():
        selected_classes = row["Selected Classes"]
        rebalance_date = row["Rebalance Date"]
        weights_erc = np.array(row["Weights"])
        fees = row["Fees at Rebalancing"]
        fees_total += fees

        # Check if selected classes align with investor-selected assets
        if not set(selected_classes).issubset(set(selected_assets)):
            continue

        # Combine the daily returns for the selected asset classes
        combined_daily_returns = pd.concat(
            [equal_weighted_portfolios[cls]["Daily Equal Weighted Return"] for cls in selected_classes],
            axis=1
        )
        combined_daily_returns.columns = selected_classes

        # Define the OOS period for this rebalance date
        oos_period_start = rebalance_date + pd.Timedelta(days=1)
        oos_period_end = rebalance_date + pd.DateOffset(months=months)

        # Get the OOS returns for this rebalance period
        oos_returns = combined_daily_returns.loc[oos_period_start:oos_period_end].dropna()

        if not oos_returns.empty:
            # Compute portfolio returns for this period
            portfolio_returns = oos_returns @ weights_erc

            # Apply fees: Adjust the returns by subtracting fees equally over the period
            daily_fee_adjustment = fees / len(portfolio_returns)
            adjusted_returns = portfolio_returns - daily_fee_adjustment

            aggregated_returns.append(adjusted_returns)

            # Aggregate yearly metrics
            for date, ret in portfolio_returns.items():
                year = date.year
                if year not in yearly_metrics:
                    yearly_metrics[year] = {"returns": []}
                yearly_metrics[year]["returns"].append(ret)

    if aggregated_returns:
        # Combine all returns across all periods
        full_period_returns = pd.concat(aggregated_returns)

        # Calculate overall metrics for the portfolio
        total_return = (1 + full_period_returns).prod() - 1
        annualized_return = np.mean(full_period_returns) * 252
        annualized_volatility = np.std(full_period_returns) * np.sqrt(252)

        # Risk-free rate (average over the entire period)
        if risk_free_rate_series is not None:
            rf_full = risk_free_rate_series.loc[full_period_returns.index].mean() / 100
        else:
            rf_full = 0

        # Sharpe Ratio
        excess_return = full_period_returns - rf_full / 252
        sharpe_ratio = excess_return.mean() / full_period_returns.std() if full_period_returns.std() != 0 else np.nan

        # Sortino Ratio
        downside_risk = np.sqrt(((full_period_returns[full_period_returns < 0]) ** 2).mean()) * np.sqrt(252)
        sortino_ratio = excess_return.mean() / downside_risk if downside_risk != 0 else np.nan

        # Calculate Beta
        if benchs_daily_returns is not None:
            market_returns = benchs_daily_returns["AQR MULTI-ASSET FUND I"].loc[full_period_returns.index]
            aligned_portfolio_returns, aligned_market_returns = full_period_returns.align(market_returns, join="inner")
            if len(aligned_market_returns) > 1:
                covariance = np.cov(aligned_portfolio_returns, aligned_market_returns)[0, 1]
                market_variance = aligned_market_returns.var()
                beta = covariance / market_variance if market_variance != 0 else np.nan
            else:
                beta = np.nan
        else:
            beta = np.nan

        # Maximum Drawdown
        cumulative_returns = (1 + full_period_returns).cumprod()
        max_drawdown = (cumulative_returns / cumulative_returns.cummax() - 1).min() * 100

        # Cumulative Return
        cumulative_return = (1 + full_period_returns).prod()

        # Store the overall portfolio metrics
        final_metrics = {
            "Final Annualized Return (net of Fees)": annualized_return,
            "Final Annualized Volatility": annualized_volatility,
            "Sharpe Ratio": sharpe_ratio,
            "Beta": beta,
            "Sortino Ratio": sortino_ratio,
            "Max Drawdown (%)": max_drawdown,
            "Cumulative Return": cumulative_return,
            "Total Fees": fees_total
        }

    # Compute yearly metrics
    yearly_results = []
    for year, data in yearly_metrics.items():
        yearly_returns = pd.Series(data["returns"])
        yearly_annualized_return = np.mean(yearly_returns) * 252
        yearly_annualized_volatility = np.std(yearly_returns) * np.sqrt(252)

        # Risk-free rate for the year
        if risk_free_rate_series is not None:
            yearly_rf = risk_free_rate_series.loc[str(year)].mean() / 100
        else:
            yearly_rf = 0

        # Sharpe Ratio
        excess_yearly_return = yearly_returns - yearly_rf / 252
        yearly_sharpe_ratio = (
            excess_yearly_return.mean() / yearly_returns.std() if yearly_returns.std() != 0 else np.nan
        )

        # Sortino Ratio
        downside_risk = np.sqrt(((yearly_returns[yearly_returns < 0]) ** 2).mean()) * np.sqrt(252)
        yearly_sortino_ratio = (
            excess_yearly_return.mean() / downside_risk if downside_risk != 0 else np.nan
        )

        # Maximum Drawdown
        cumulative_returns = (1 + yearly_returns).cumprod()
        yearly_max_drawdown = (cumulative_returns / cumulative_returns.cummax() - 1).min() * 100

        yearly_results.append({
            "Year": year,
            "Annualized Return (net of Fees)": yearly_annualized_return,
            "Annualized Volatility": yearly_annualized_volatility,
            "Sharpe Ratio": yearly_sharpe_ratio,
            "Sortino Ratio": yearly_sortino_ratio,
            "Max Drawdown (%)": yearly_max_drawdown
        })

    # Create DataFrames
    final_metrics_df = pd.DataFrame([final_metrics])
    yearly_metrics_df = pd.DataFrame(yearly_results)

    return final_metrics_df, yearly_metrics_df


def calculate_daily_returns_with_fees(
        erc_portfolio_df,
        equal_weighted_portfolios,
        start_date,
        end_date,
        months=6  # Default rebalancing frequency to 6 months if not provided
):
    """
    Calculate daily portfolio returns adjusted for fees.

    Args:
        erc_portfolio_df (pd.DataFrame): DataFrame containing ERC portfolio details.
        equal_weighted_portfolios (dict): Equal-weighted portfolios by category.
        start_date (str): Start date for the calculation.
        end_date (str): End date for the calculation.
        months (int): Rebalancing frequency in months.

    Returns:
        pd.DataFrame: DataFrame with daily portfolio returns adjusted for fees.
    """
    # Create a date range from start_date to end_date
    date_range = pd.date_range(start=start_date, end=end_date, freq='B')  # 'B' ensures weekdays only

    # Initialize a Series to store daily returns for the single portfolio
    daily_returns = pd.Series(index=date_range, dtype=float)

    # Loop through each row in the portfolio DataFrame
    for _, row in erc_portfolio_df.iterrows():
        rebalance_date = row["Rebalance Date"]
        weights_erc = np.array(row["Weights"])
        fees = row["Fees at Rebalancing"]
        selected_classes = row["Selected Classes"]

        # Combine the daily returns for the selected asset classes
        combined_daily_returns = pd.concat(
            [equal_weighted_portfolios[cls]["Daily Equal Weighted Return"] for cls in selected_classes],
            axis=1
        )
        combined_daily_returns.columns = selected_classes

        # Define the rebalancing period (from rebalance_date to the next rebalance)
        next_rebalance_date = rebalance_date + pd.DateOffset(months=months)
        rebalancing_period = pd.date_range(rebalance_date, next_rebalance_date, freq='B')

        # Apply the fee adjustment: fee per day for the period until the next rebalance
        fee_per_day = fees / len(rebalancing_period)

        # Loop through each day in the rebalancing period
        for date in rebalancing_period:
            if date in combined_daily_returns.index:
                # Calculate the daily return for the portfolio: weighted sum of the asset class returns
                daily_portfolio_return = (combined_daily_returns.loc[date] @ weights_erc)

                # Adjust for fees by subtracting the fee per day from the return
                daily_portfolio_return_with_fee = daily_portfolio_return - fee_per_day

                # Store the daily return with fee adjustment
                daily_returns[date] = daily_portfolio_return_with_fee

    # Return the daily returns as a DataFrame
    daily_returns_df = daily_returns.to_frame(name="Portfolio Daily Return")

    return daily_returns_df






def align_daily_returns(portfolio_daily_returns_df, benchs_daily_returns):
    """
    Align the benchmark and portfolio daily returns based on the common date range.

    Args:
        portfolio_daily_returns_df (pd.DataFrame): DataFrame of portfolio daily returns.
        benchs_daily_returns (pd.DataFrame): DataFrame of benchmark daily returns.

    Returns:
        Tuple[pd.DataFrame, pd.Series]: Aligned daily returns and benchmark returns.
    """
    common_start_date = max(portfolio_daily_returns_df.index.min(), benchs_daily_returns.index.min())
    common_end_date = min(portfolio_daily_returns_df.index.max(), benchs_daily_returns.index.max())

    aligned_daily_returns_df = portfolio_daily_returns_df[common_start_date:common_end_date].dropna()
    aligned_market_returns = benchs_daily_returns[common_start_date:common_end_date].dropna()

    return aligned_daily_returns_df, aligned_market_returns


def plot_cumulative_and_drawdown(aligned_daily_returns_df, aligned_market_returns, title):
    """
    Plot cumulative returns and drawdowns for portfolios and the benchmark.

    Args:
        aligned_daily_returns_df (pd.DataFrame): DataFrame of aligned daily returns for portfolios.
        aligned_market_returns (pd.Series): Series of aligned benchmark daily returns.
        title (str): Title for the plots.

    Returns:
        matplotlib.figure.Figure: The matplotlib figure containing the plots.
    """
    fig, axes = plt.subplots(2, 1, figsize=(20, 15))  # Create subplots for cumulative returns and drawdowns

    # Subplot 1: Cumulative Returns
    axes[0].set_title(f"{title} - Cumulative Returns", fontsize=16)
    for portfolio in aligned_daily_returns_df.columns:
        cumulative_returns = (1 + aligned_daily_returns_df[portfolio]).cumprod() - 1
        axes[0].plot(cumulative_returns.index, cumulative_returns, label=portfolio)
    benchmark_cumulative_returns = (1 + aligned_market_returns).cumprod() - 1
    axes[0].plot(benchmark_cumulative_returns.index, benchmark_cumulative_returns, linestyle='--', color='navy',
                 linewidth=4, label="Benchmark")
    axes[0].grid(True)
    axes[0].set_ylabel("Cumulative Return (%)")
    axes[0].legend()

    # Subplot 2: Drawdowns
    axes[1].set_title(f"{title} - Drawdowns", fontsize=16)
    for portfolio in aligned_daily_returns_df.columns:
        cumulative_returns = (1 + aligned_daily_returns_df[portfolio]).cumprod()
        drawdown = (cumulative_returns / cumulative_returns.cummax()) - 1
        axes[1].plot(drawdown.index, drawdown, label=f"{portfolio} Drawdown")

    # Benchmark drawdown
    benchmark_cumulative_returns_raw = (1 + aligned_market_returns).cumprod()
    benchmark_drawdown = (benchmark_cumulative_returns_raw / benchmark_cumulative_returns_raw.cummax()) - 1
    axes[1].plot(benchmark_drawdown.index, benchmark_drawdown, linestyle='--', color='navy', linewidth=4,
                 label="Benchmark Drawdown")

    axes[1].grid(True)
    axes[1].set_ylabel("Drawdown (%)")
    axes[1].legend()

    # Tight layout for better spacing
    plt.tight_layout()
    return fig

def plot_asset_allocation_over_time_for_portfolio(weights_over_time_df, asset_classes):
    """
    Plot the asset allocation over time for a given portfolio using an area plot with custom colors.

    Args:
    weights_over_time_df (pd.DataFrame): DataFrame where each column represents an asset class and rows represent weights over time.
    asset_classes (list): List of asset classes included in the portfolio.
    """
    # Define custom color palette inside the function
    custom_colors = [
        "#afeeee",  # paleturquoise
        "#008080",  # teal
        "#00ffff",  # aqua
        "#5f9ea0",  # cadetblue
        "#add8e6",  # lightblue
        "#00bfff",  # deepskyblue
        "#4682b4",  # steelblue
        "#1e90ff",  # dodgerblue
        "#2f4f4f"   # darkslategray
    ]

    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(15, 6))

    # Plotting area plot for asset allocation over time using custom colors
    weights_over_time_df.plot.area(
        stacked=True, alpha=0.8, color=custom_colors[:len(weights_over_time_df.columns)], ax=ax
    )

    ax.set_title("Asset Allocation Over Time for Portfolio", fontsize=14, fontweight='bold')
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Portfolio Weight (%)", fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(title="Asset Classes", loc='upper left')
    plt.tight_layout()

    # Return the figure for rendering in Streamlit
    return fig



def display_asset_allocation_over_time_for_portfolio(erc_portfolios_df, investor_selected_assets):
    """
    Extract asset weights for a specific portfolio over time and display the allocation as an area plot.

    Args:
        erc_portfolios_df (pd.DataFrame): ERC portfolio DataFrame containing portfolio details.
        investor_selected_assets (list): List of asset classes selected by the investor.
    """
    # Filtering the portfolios to get the specific portfolio based on selected asset classes
    portfolio_data = erc_portfolios_df[
        erc_portfolios_df['Selected Classes'].apply(lambda x: set(x) == set(investor_selected_assets))
    ]

    if portfolio_data.empty:
        print("No matching portfolio found with the given asset classes.")
        return

    # Extracting the weights over time for the portfolio
    weights_over_time = []

    for _, row in portfolio_data.iterrows():
        rebalance_date = row['Rebalance Date']
        weights = row['Weights']  # Assuming 'Weights' contains weights of all asset classes

        if isinstance(weights, list):
            weights_series = pd.Series(weights, index=row['Selected Classes'], name=rebalance_date)
            weights_over_time.append(weights_series)

    # Combine into a DataFrame for plotting
    weights_over_time_df = pd.DataFrame(weights_over_time)
    weights_over_time_df.index = pd.to_datetime(weights_over_time_df.index)  # Set the index as rebalance dates for proper time representation

    # Plot the asset allocation over time for the specific portfolio
    plot_asset_allocation_over_time_for_portfolio(weights_over_time_df, investor_selected_assets)


def extract_weights_over_time(erc_portfolio_df):
    """
    Extract asset weights over time from the ERC portfolio DataFrame.

    Args:
        erc_portfolio_df (pd.DataFrame): DataFrame containing ERC portfolio details, including weights.

    Returns:
        pd.DataFrame: DataFrame with asset weights over time, indexed by rebalance dates.
    """
    weights_over_time = []

    for _, row in erc_portfolio_df.iterrows():
        rebalance_date = row['Rebalance Date']
        weights = row['Weights']
        selected_classes = row['Selected Classes']

        if isinstance(weights, list):
            weights_series = pd.Series(weights, index=selected_classes, name=rebalance_date)
            weights_over_time.append(weights_series)

    if weights_over_time:
        weights_over_time_df = pd.DataFrame(weights_over_time)
        weights_over_time_df.index = pd.to_datetime(weights_over_time_df.index)  # Ensure proper datetime index
        return weights_over_time_df
    else:
        return pd.DataFrame()

def plot_risk_contribution_over_time_for_portfolio(risk_contribution_df, asset_classes):
    """
    Plot the risk contribution of asset classes over time for a specific portfolio using a stacked bar plot with custom colors.

    Args:
        risk_contribution_df (pd.DataFrame): DataFrame where each column represents an asset class and rows represent risk contribution over time.
        asset_classes (list): List of asset classes included in the portfolio.

    Returns:
        matplotlib.figure.Figure: The figure object for the plot.
    """
    # Define custom colors
    custom_colors = [
        "#afeeee", "#008080", "#00ffff", "#5f9ea0", "#add8e6", "#00bfff", "#4682b4", "#1e90ff", "#2f4f4f"
    ]

    # Create the figure and axes
    fig, ax = plt.subplots(figsize=(12, 6))

    # Plotting stacked bar chart for risk contribution over time using custom colors
    risk_contribution_df.plot(
        kind='bar',
        stacked=True,
        color=custom_colors[:len(risk_contribution_df.columns)],
        alpha=0.8,
        width=0.85,
        ax=ax
    )

    ax.set_title("Risk Contribution of Asset Classes Over Time", fontsize=14, fontweight='bold')
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Risk Contribution (%)", fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(title="Asset Classes", loc='upper left')

    plt.tight_layout()

    return fig



def display_risk_contribution_over_time_for_portfolio(erc_portfolios_df, investor_selected_assets):
    """
    Extract risk contribution data for a specific portfolio over time and display it as a stacked bar plot.
    The user can specify which portfolio by selecting asset classes.

    Args:
        erc_portfolios_df (pd.DataFrame): DataFrame containing ERC portfolio details.
        investor_selected_assets (list): List of asset classes selected by the investor.

    Returns:
        None
    """
    # Filtering the portfolios to get the specific portfolio based on selected asset classes
    portfolio_data = erc_portfolios_df[erc_portfolios_df['Selected Classes'].apply(lambda x: set(x) == set(investor_selected_assets))]

    if portfolio_data.empty:
        print("No matching portfolio found with the given asset classes.")
        return

    # Extracting the risk contributions over time for the portfolio
    risk_contributions_over_time = []

    for _, row in portfolio_data.iterrows():
        rebalance_date = row['Rebalance Date']
        risk_contributions = row['Absolute Risk Contribution']  # Assuming 'Absolute Risk Contribution' contains the risk contribution of each asset class

        if isinstance(risk_contributions, list):
            risk_contribution_series = pd.Series(risk_contributions, index=row['Selected Classes'], name=rebalance_date)
            risk_contributions_over_time.append(risk_contribution_series)

    # Combine into a DataFrame for plotting
    risk_contributions_over_time_df = pd.DataFrame(risk_contributions_over_time)
    risk_contributions_over_time_df.index = pd.to_datetime(risk_contributions_over_time_df.index)  # Set the index as rebalance dates for proper time representation

    # Plot the risk contribution over time for the specific portfolio
    plot_risk_contribution_over_time_for_portfolio(risk_contributions_over_time_df, investor_selected_assets)

def extract_risk_contributions_over_time(erc_portfolio_df):
    """
    Extract risk contribution data for asset classes over time from the ERC portfolio DataFrame.

    Args:
        erc_portfolio_df (pd.DataFrame): DataFrame containing portfolio data with 'Absolute Risk Contribution'.

    Returns:
        pd.DataFrame: A DataFrame where rows represent rebalancing dates, columns represent asset classes,
                      and values represent their risk contribution over time.
    """
    risk_contributions_over_time = []

    for _, row in erc_portfolio_df.iterrows():
        rebalance_date = row['Rebalance Date']
        risk_contributions = row['Absolute Risk Contribution']
        selected_classes = row['Selected Classes']

        # Ensure risk contributions align with selected classes
        if isinstance(risk_contributions, list) and isinstance(selected_classes, list):
            risk_contribution_series = pd.Series(
                risk_contributions, index=selected_classes, name=rebalance_date
            )
            risk_contributions_over_time.append(risk_contribution_series)

    # Combine all risk contributions into a DataFrame
    risk_contributions_over_time_df = pd.DataFrame(risk_contributions_over_time)
    risk_contributions_over_time_df.index = pd.to_datetime(risk_contributions_over_time_df.index)

    return risk_contributions_over_time_df

import streamlit as st

def select_and_display_portfolio_performance_with_period(
    erc_portfolios_df,
    predefined_selection=None,
    predefined_dates=None,
    risk_free_rate_series=None,
    investment_amount_chf=None
):
    """
    Select a portfolio based on predefined asset classes and dates, display performance metrics interactively,
    and calculate fees in Swiss francs.

    Args:
        erc_portfolios_df (pd.DataFrame): DataFrame containing ERC portfolio data.
        predefined_selection (list): Predefined list of asset classes selected by the investor.
        predefined_dates (tuple): Predefined start and end dates for the investment period.
        risk_free_rate_series (pd.Series, optional): Risk-free rate series for Sharpe ratio calculation.
        investment_amount_chf (float, optional): Total investment amount in Swiss francs.
    """
    # Check for required columns in the DataFrame
    required_columns = ["Rebalance Date", "Net of Fees Portfolio Return (Annualized)",
                        "Portfolio Volatility (Annualized)", "Fees at Rebalancing", "Selected Classes"]
    missing_columns = [col for col in required_columns if col not in erc_portfolios_df.columns]
    if missing_columns:
        st.error(f"Missing required columns in portfolio data: {', '.join(missing_columns)}")
        return

    # Ensure the dates are converted to datetime64[ns]
    start_date, end_date = predefined_dates
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)

    selected_classes_set = set(predefined_selection)

    # Filter matching portfolios
    matching_portfolios = erc_portfolios_df[
        (erc_portfolios_df['Selected Classes'].apply(set) == selected_classes_set) &
        (erc_portfolios_df['Rebalance Date'] >= start_date) &
        (erc_portfolios_df['Rebalance Date'] <= end_date)
    ]

    if matching_portfolios.empty:
        st.warning("No portfolio matches your selection or falls within the specified date range.")
        return

    # Calculate metrics for visualization
    performance_data = []
    total_fees_chf = 0  # Initialize total fees in Swiss francs
    cumulative_return = 1  # Initialize cumulative return for overall calculation

    for _, row in matching_portfolios.iterrows():
        rebalance_date = row["Rebalance Date"]
        portfolio_return = row.get("Net of Fees Portfolio Return (Annualized)", 0)
        portfolio_volatility = row.get("Portfolio Volatility (Annualized)", 0)
        fees_percent = row.get("Fees at Rebalancing", 0)

        # Validate investment amount and fees_percent
        if investment_amount_chf is None or fees_percent is None:
            st.warning(f"Missing data for fees or investment amount at {rebalance_date}. Skipping this period.")
            continue

        # Calculate fees in Swiss francs
        fees_chf = fees_percent * investment_amount_chf
        total_fees_chf += fees_chf

        # Update cumulative return
        cumulative_return *= (1 + portfolio_return / 252) ** 252

        # Calculate Sharpe ratio with fallback
        if risk_free_rate_series is not None:
            try:
                risk_free_rate = risk_free_rate_series.loc[rebalance_date]
            except KeyError:
                risk_free_rate = risk_free_rate_series.mean()  # Default to mean if not found

            sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility if portfolio_volatility != 0 else None
        else:
            sharpe_ratio = None

        performance_data.append({
            "Rebalance Date": rebalance_date,
            "Portfolio Return (%)": portfolio_return * 100,
            "Portfolio Volatility (%)": portfolio_volatility * 100,
            "Sharpe Ratio": sharpe_ratio,
            "Fees (CHF)": fees_chf,
        })

    # Convert to DataFrame
    performance_df = pd.DataFrame(performance_data)

    # --- Interactive Plot 1: Returns ---
    fig_returns = go.Figure()
    fig_returns.add_trace(go.Bar(
        x=performance_df["Rebalance Date"],
        y=performance_df["Portfolio Return (%)"],
        name="Portfolio Return (%)",
        marker=dict(color="blue"),
        text=performance_df["Portfolio Return (%)"].round(2),
        textposition="auto"
    ))
    fig_returns.update_layout(
        title="Portfolio Returns for Each Rebalancing Period",
        xaxis_title="Rebalance Date",
        yaxis_title="Return (%)",
        template="plotly_white",
        hovermode="x unified",
        legend_title="Legend"
    )

    # --- Interactive Plot 2: Fees ---
    fig_fees = go.Figure()
    fig_fees.add_trace(go.Bar(
        x=performance_df["Rebalance Date"],
        y=performance_df["Fees (CHF)"],
        name="Fees (CHF)",
        marker=dict(color="orange"),
        text=performance_df["Fees (CHF)"].round(2),
        textposition="auto"
    ))
    fig_fees.update_layout(
        title="Fees (CHF) for Each Rebalancing Period",
        xaxis_title="Rebalance Date",
        yaxis_title="Fees (CHF)",
        template="plotly_white",
        hovermode="x unified",
        legend_title="Legend"
    )

    # Display in Streamlit
    st.plotly_chart(fig_returns, use_container_width=True)
    st.plotly_chart(fig_fees, use_container_width=True)

    # Compute and display overall metrics
    avg_annualized_return = performance_df["Portfolio Return (%)"].mean() / 100
    avg_annualized_volatility = performance_df["Portfolio Volatility (%)"].mean() / 100
    overall_sharpe_ratio = (avg_annualized_return - risk_free_rate_series.mean()) / avg_annualized_volatility \
        if avg_annualized_volatility != 0 else None

    st.markdown("#### Overall Metrics")
    st.write(f"**Cumulative Return:** {((cumulative_return - 1) * 100):.2f}%")
    st.write(f"**Average Annualized Return:** {avg_annualized_return:.2%}")
    st.write(f"**Average Annualized Volatility:** {avg_annualized_volatility:.2%}")
    st.write(f"**Overall Sharpe Ratio:** {overall_sharpe_ratio:.2f}")
    st.write(f"**Total Fees (CHF):** {total_fees_chf:,.2f}")

    # Show performance data in a table for reference
    st.markdown("#### Performance Data Summary")
    st.dataframe(performance_df.style.format({
        "Portfolio Return (%)": "{:.2f}",
        "Portfolio Volatility (%)": "{:.2f}",
        "Sharpe Ratio": "{:.2f}",
        "Fees (CHF)": "{:,.2f}"
    }))










# Generate the interactive bar chart for returns comparison
def plot_returns_comparison(erc_portfolios_df, benchmark_returns):
    """
    Create an interactive bar chart comparing portfolio returns and benchmark returns.

    Args:
        erc_portfolios_df (pd.DataFrame): Portfolio performance metrics at rebalancing dates.
        benchmark_returns (pd.Series): Daily returns of the benchmark.

    Returns:
        plotly.graph_objects.Figure: A Plotly figure object for the returns comparison.
    """
    dates = []
    portfolio_returns = []
    benchmark_returns_list = []

    for _, row in erc_portfolios_df.iterrows():
        rebalance_date = row["Rebalance Date"]
        portfolio_return = row["Net of Fees Portfolio Return (Annualized)"]

        # Filter benchmark returns for the rebalancing period
        benchmark_period = benchmark_returns.loc[
                           rebalance_date: rebalance_date + pd.DateOffset(months=6)
                           ].dropna()

        if not benchmark_period.empty:
            benchmark_return = benchmark_period.mean() * 252

            # Append data for the plot
            dates.append(rebalance_date.date())
            portfolio_returns.append(portfolio_return * 100)
            benchmark_returns_list.append(benchmark_return * 100)

    # Create the Plotly bar chart
    fig = go.Figure()

    # Add Portfolio Returns
    fig.add_trace(go.Bar(
        x=dates,
        y=portfolio_returns,
        name="Portfolio Returns",
        marker=dict(color="steelblue"),
    ))

    # Add Benchmark Returns
    fig.add_trace(go.Bar(
        x=dates,
        y=benchmark_returns_list,
        name="Benchmark Returns",
        marker=dict(color="lightcoral"),
    ))

    # Customize layout
    fig.update_layout(
        title="Portfolio vs. Benchmark Returns at Each Rebalancing Date",
        xaxis_title="Rebalance Date",
        yaxis_title="Annualized Returns (%)",
        barmode="group",  # Side-by-side bars
        template="plotly_white",
        legend=dict(title="Legend"),
    )

    return fig


def plot_interactive_cumulative_and_drawdowns(aligned_daily_returns_df, aligned_market_returns):
    """
    Plot interactive cumulative returns and drawdowns for portfolios and the benchmark.

    Args:
        aligned_daily_returns_df (pd.DataFrame): DataFrame of aligned daily returns for portfolios.
        aligned_market_returns (pd.Series): Series of aligned benchmark daily returns.

    Returns:
        Tuple[plotly.graph_objects.Figure, plotly.graph_objects.Figure]: Interactive cumulative returns and drawdowns plots.
    """
    # Calculate cumulative returns and drawdowns
    cumulative_portfolio = (1 + aligned_daily_returns_df).cumprod() - 1
    cumulative_benchmark = (1 + aligned_market_returns).cumprod() - 1

    drawdown_portfolio = (1 + aligned_daily_returns_df).cumprod() / (
            1 + aligned_daily_returns_df).cumprod().cummax() - 1
    drawdown_benchmark = (1 + aligned_market_returns).cumprod() / (
            1 + aligned_market_returns).cumprod().cummax() - 1

    # --- Plot 1: Cumulative Returns ---
    fig_cumulative = go.Figure()

    # Add cumulative returns for the portfolio
    for portfolio in cumulative_portfolio.columns:
        fig_cumulative.add_trace(go.Scatter(
            x=cumulative_portfolio.index,
            y=cumulative_portfolio[portfolio],
            mode='lines',
            name=f'{portfolio} ',
            line=dict(color='blue', width=2)
        ))

    # Add cumulative returns for the benchmark
    fig_cumulative.add_trace(go.Scatter(
        x=cumulative_benchmark.index,
        y=cumulative_benchmark,
        mode='lines',
        name='Benchmark (AQR MULTI-ASSET FUND I)',
        line=dict(color='orange', width=2, dash='dot')  # Dashed orange line for the benchmark
    ))

    # Customize layout for cumulative returns
    fig_cumulative.update_layout(
        title="Cumulative Returns",
        xaxis_title="Date",
        yaxis_title="Cumulative Return (%)",
        legend_title="Legend",
        template="plotly_white",
        hovermode="x unified"
    )

    # --- Plot 2: Drawdowns ---
    fig_drawdowns = go.Figure()

    # Add drawdowns for the portfolio
    for portfolio in drawdown_portfolio.columns:
        fig_drawdowns.add_trace(go.Scatter(
            x=drawdown_portfolio.index,
            y=drawdown_portfolio[portfolio],
            mode='lines',
            name=f'{portfolio} Drawdown',
            line=dict(color='blue', width=2)
        ))

    # Add drawdowns for the benchmark
    fig_drawdowns.add_trace(go.Scatter(
        x=drawdown_benchmark.index,
        y=drawdown_benchmark,
        mode='lines',
        name='Benchmark (AQR MULTI-ASSET FUND I) Drawdown',
        line=dict(color='orange', width=2, dash='dot')  # Dashed orange line for the benchmark drawdown
    ))

    # Customize layout for drawdowns
    fig_drawdowns.update_layout(
        title="Drawdowns",
        xaxis_title="Date",
        yaxis_title="Drawdown (%)",
        legend_title="Legend",
        template="plotly_white",
        hovermode="x unified"
    )

    return fig_cumulative, fig_drawdowns
