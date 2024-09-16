# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import plotly.express as px
from datetime import datetime, timedelta
from sklearn.decomposition import PCA
from scipy.optimize import minimize
from scipy import stats
import statsmodels.api as sm
import pandas_datareader as pdr
import warnings
from PIL import Image
import pytz

warnings.filterwarnings('ignore')
image = Image.open('PortfolioOptimizationApp/optGraph.jpg')
st.title('Portfolio Optimization & Analysis')
st.image(image, caption='', use_column_width=True)

def fetch_factor_data(start_date, end_date):
    ff_factors = pdr.get_data_famafrench('F-F_Research_Data_Factors_daily', start=start_date, end=end_date)[0]
    ff_factors.index = pd.to_datetime(ff_factors.index).tz_localize(None)  # Ensure tz-naive
    ff_factors = ff_factors / 100  # Convert to decimal format
    return ff_factors

def calculate_performance(weights, returns, risk_free_rate):
    portfolio_return = (returns * weights).sum(axis=1)
    excess_return = portfolio_return - risk_free_rate
    trading_days = len(returns)
    years = trading_days / 252  # Calculate the fraction of a year
    annualization_factor = np.sqrt(252 / trading_days)

    total_return = (1 + portfolio_return).prod() - 1
    annualized_return = (1 + total_return) ** (1 / years) - 1
    volatility = portfolio_return.std() * np.sqrt(252 / trading_days)
    sharpe_ratio = excess_return.mean() / excess_return.std() * np.sqrt(252 / trading_days)
    sortino_ratio = excess_return.mean() / excess_return[excess_return < 0].std() * np.sqrt(252 / trading_days)
    max_drawdown = calculate_max_drawdown(portfolio_return)
    calmar_ratio = annualized_return / abs(max_drawdown)

    return {
        'Total Return': total_return,
        'Annualized Return': annualized_return,
        'Volatility': volatility,
        'Sharpe Ratio': sharpe_ratio,
        'Sortino Ratio': sortino_ratio,
        'Max Drawdown': max_drawdown,
        'Calmar Ratio': calmar_ratio
    }
def calculate_max_drawdown(returns):
    cumulative_returns = (1 + returns).cumprod()
    peak = cumulative_returns.expanding(min_periods=1).max()
    drawdown = (cumulative_returns / peak) - 1
    return drawdown.min()

def minimum_variance_portfolio(returns):
    if isinstance(returns, pd.Series):
        returns = returns.to_frame()
    n = returns.shape[1]
    trading_days = len(returns)
    args = (returns,)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0, 1) for _ in range(n))
    result = minimize(lambda weights, returns: (returns * weights).sum(axis=1).std() * np.sqrt(252 / trading_days),
                      n*[1./n,], args=args, method='SLSQP', bounds=bounds, constraints=constraints)
    return pd.Series(result.x, index=returns.columns)

def maximum_sharpe_ratio_portfolio(returns, risk_free_rate):
    if isinstance(returns, pd.Series):
        returns = returns.to_frame()
    n = returns.shape[1]
    trading_days = len(returns)
    args = (returns, risk_free_rate)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0, 1) for _ in range(n))
    result = minimize(lambda weights, returns, rf: -((returns * weights).sum(axis=1).mean() - rf) /
                      ((returns * weights).sum(axis=1).std() * np.sqrt(252 / trading_days)),
                      n*[1./n,], args=args, method='SLSQP', bounds=bounds, constraints=constraints)
    return pd.Series(result.x, index=returns.columns)

def risk_parity_portfolio(returns):
    if isinstance(returns, pd.Series):
        returns = returns.to_frame()
    n = returns.shape[1]
    trading_days = len(returns)
    target_risk = 1/n
    def objective(weights):
        portfolio_vol = (returns * weights).sum(axis=1).std() * np.sqrt(252 / trading_days)
        asset_contribs = weights * (returns.cov() * (252 / trading_days)).dot(weights) / portfolio_vol
        return np.sum((asset_contribs - target_risk)**2)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0, 1) for _ in range(n))
    result = minimize(objective, n*[1./n,], method='SLSQP', bounds=bounds, constraints=constraints)
    return pd.Series(result.x, index=returns.columns)

def black_litterman_portfolio(returns, market_caps, risk_aversion=2.5, tau=0.05):
    if isinstance(returns, pd.Series):
        returns = returns.to_frame()
    trading_days = len(returns)
    Sigma = returns.cov() * (252 / trading_days)
    Pi = risk_aversion * Sigma.dot(market_caps)
    weights = np.linalg.inv(tau * Sigma + Sigma).dot(tau * Sigma.dot(market_caps) + Pi)
    return pd.Series(weights / weights.sum(), index=returns.columns)

def momentum_portfolio(returns, lookback=12):
    if isinstance(returns, pd.Series):
        returns = returns.to_frame()
    momentum = returns.iloc[-lookback:].mean()
    weights = momentum / momentum.sum()
    return weights

def factor_analysis(portfolio_returns, factor_returns):
    aligned_data = pd.concat([portfolio_returns, factor_returns], axis=1).dropna()
    portfolio_returns = aligned_data.iloc[:, 0]
    factor_returns = aligned_data.iloc[:, 1:]
    X = sm.add_constant(factor_returns)
    model = sm.OLS(portfolio_returns, X).fit()
    return model

def bootstrap_sharpe_ratio(returns, weights, num_simulations=10000):
    portfolio_returns = (returns * weights).sum(axis=1)
    trading_days = len(returns)
    sharpe_ratios = []
    for _ in range(num_simulations):
        sample = np.random.choice(portfolio_returns, size=len(portfolio_returns), replace=True)
        sharpe = np.sqrt(252 / trading_days) * sample.mean() / sample.std()
        sharpe_ratios.append(sharpe)
    return np.mean(sharpe_ratios), np.percentile(sharpe_ratios, [2.5, 97.5])

def stress_test(weights, returns, scenarios):
    portfolio_returns = (returns * weights).sum(axis=1)
    results = {}
    for name, scenario in scenarios.items():
        stressed_returns = portfolio_returns * scenario
        results[name] = stressed_returns.sum()
    return results    

# Input section
st.header('Portfolio Setup')
ticker_input = st.text_input('Enter stock tickers (comma-separated)', '')
stock_tickers = [stock.strip() for stock in ticker_input.split(",")]

# Reorganized date inputs
st.subheader('Date Range Selection')
col1, col2 = st.columns(2)
with col1:
    analysis_start = st.date_input('Analysis Start Date', datetime(2021, 1, 1).date(), key='analysis_start')
    backtest_start = st.date_input('Backtest Start Date', datetime(2023, 1, 1).date(), key='backtest_start')

with col2:
    analysis_end = st.date_input('Analysis End Date', datetime(2022, 12, 31).date(), key='analysis_end')
    backtest_end = st.date_input('Backtest End Date', datetime(2023, 12, 31).date(), key='backtest_end')

# Convert date inputs to tz-naive datetime objects
analysis_start = datetime.combine(analysis_start, datetime.min.time())
analysis_end = datetime.combine(analysis_end, datetime.min.time())
backtest_start = datetime.combine(backtest_start, datetime.min.time())
backtest_end = datetime.combine(backtest_end, datetime.min.time())

if st.button('Run Analysis'):
    try:
        # Data fetching and preprocessing
        @st.cache_data
        def fetch_data(tickers, start, end):
            data = yf.download(tickers, start=start, end=end, progress=False)
            data.index = data.index.tz_localize(None)  # Remove timezone info
            return data['Adj Close']

        data = fetch_data(stock_tickers, analysis_start, backtest_end)
        data = data.dropna(axis=1)
        returns = data.pct_change().dropna()
        data.to_csv("stock_data.csv")
        st.download_button(
        label="Download stock data as CSV",
        data=data,
        file_name="stock_data.csv",
        mime="text/csv",
)

        analysis_returns = returns[(returns.index >= analysis_start) & (returns.index <= analysis_end)]
        backtest_returns = returns[(returns.index >= backtest_start) & (returns.index <= backtest_end)]

        @st.cache_data
        def get_risk_free_rate(start, end):
            try:
                tnx = yf.Ticker("^TNX")
                history = tnx.history(start=start, end=end)
                history.index = history.index.tz_localize(None)  # Remove timezone info
                return history['Close'].iloc[-1] / 100 / 252
            except Exception as e:
                st.warning(f"Unable to fetch risk-free rate: {e}. Using default value of 0.02/252.")
                return 0.02 / 252

        risk_free_rate = get_risk_free_rate(analysis_start, backtest_end)

        @st.cache_data
        def get_market_cap(ticker):
            try:
                return yf.Ticker(ticker).info.get('marketCap', 1e9)
            except:
                return 1e9

        market_caps = pd.Series({ticker: get_market_cap(ticker) for ticker in data.columns})
        market_caps = market_caps[data.columns]
        market_caps = market_caps / market_caps.sum()

        # Portfolio creation
        portfolios = {
            'Eigenportfolio': pd.Series(PCA(n_components=1).fit(analysis_returns).components_[0], index=data.columns),
            'Equal-Weight': pd.Series(1/len(data.columns), index=data.columns),
            'Minimum Variance': minimum_variance_portfolio(analysis_returns),
            'Maximum Sharpe Ratio': maximum_sharpe_ratio_portfolio(analysis_returns, risk_free_rate),
            'Risk Parity': risk_parity_portfolio(analysis_returns),
            'Black-Litterman': black_litterman_portfolio(analysis_returns, market_caps),
            'Momentum': momentum_portfolio(analysis_returns)
        }

        for name, weights in portfolios.items():
            portfolios[name] = weights / weights.sum()

        results = {name: calculate_performance(weights, backtest_returns, risk_free_rate) for name, weights in portfolios.items()}

        # Plotting
        st.header('Portfolio Performance Comparison')
        fig, ax = plt.subplots(figsize=(12, 6))
        for name, weights in portfolios.items():
            cumulative_return = (1 + (backtest_returns * weights).sum(axis=1)).cumprod()
            ax.plot(cumulative_return.index, cumulative_return, label=f'{name} ({results[name]["Total Return"]:.2%})')
        ax.set_title('Portfolio Performance Comparison')
        ax.set_xlabel('Date')
        ax.set_ylabel('Cumulative Return')
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)

        st.header('Factor Analysis')
        factor_data = fetch_factor_data(backtest_start, backtest_end)
        for name, weights in portfolios.items():
            st.subheader(f'{name}')
            portfolio_returns = (backtest_returns * weights).sum(axis=1)
            factor_model = factor_analysis(portfolio_returns, factor_data)
            
            # Convert the summary table to a DataFrame for better presentation
            summary_df = pd.read_html(factor_model.summary().tables[1].as_html(), header=0, index_col=0)[0]
            st.dataframe(summary_df)

        # Bootstrap Analysis
        st.header('Bootstrap Analysis')
        bootstrap_results = []
        for name, weights in portfolios.items():
            mean_sharpe, ci = bootstrap_sharpe_ratio(backtest_returns, weights)
            bootstrap_results.append({
                'Strategy': name,
                'Mean Sharpe Ratio': mean_sharpe,
                'CI Lower': ci[0],
                'CI Upper': ci[1]
            })
        bootstrap_df = pd.DataFrame(bootstrap_results)
        st.dataframe(bootstrap_df)

        # Stress Testing
        st.header('Stress Test Results')
        scenarios = {
            'Market Crash': 0.7,
            'Economic Boom': 1.3,
            'High Volatility': 1.0
        }
        stress_test_results = {name: stress_test(weights, backtest_returns, scenarios) for name, weights in portfolios.items()}
        stress_test_df = pd.DataFrame(stress_test_results).T
        st.dataframe(stress_test_df)

        # Summary statistics
        st.header('Portfolio Performance Summary')
        summary_df = pd.DataFrame(results).T
        st.dataframe(summary_df)

        # Best strategy
        st.header('Best Strategy')
        best_strategy = max(results, key=lambda x: results[x]['Total Return'])
        st.write(f"Best Strategy: {best_strategy}")
        st.write(f"Best Strategy Total Return: {results[best_strategy]['Total Return']:.4%}")
        st.write(f"Best Strategy Sharpe Ratio: {results[best_strategy]['Sharpe Ratio']:.4f}")
        
        st.subheader('Best Strategy Weights')
        best_weights = portfolios[best_strategy]
        weight_df = pd.DataFrame(best_weights.sort_values(ascending=False)).reset_index()
        weight_df.columns = ['Stock', 'Weight']
        weight_df['Weight'] = weight_df['Weight'].apply(lambda x: f"{x:.4f}")
        st.dataframe(weight_df)

        st.header('Understanding Factor Analysis')
        with st.expander("Read more about interpreting factor analysis"):
            st.write("""
            Factor analysis helps us understand how different market factors influence portfolio returns. Here's how to interpret the results:
        
            1. **Coefficients**: These show the sensitivity of the portfolio to each factor. A positive coefficient means the portfolio tends to move in the same direction as the factor, while a negative coefficient indicates an inverse relationship.
        
            2. **t-statistic**: This measures the statistical significance of each factor. Generally, a t-statistic greater than 2 or less than -2 is considered significant.
        
            3. **P-value**: This is the probability that the observed relationship between the factor and portfolio returns occurred by chance. A p-value less than 0.05 is typically considered statistically significant.
        
            4. **R-squared**: This indicates the proportion of variance in portfolio returns explained by the factors. A higher R-squared suggests that the factors do a better job of explaining portfolio performance.
        
            5. **Factors**:
               - Mkt-RF: Excess return of the market portfolio over the risk-free rate
               - SMB: Small Minus Big, the return difference between small and large cap stocks
               - HML: High Minus Low, the return difference between value and growth stocks
        
            A significant positive coefficient for Mkt-RF indicates the portfolio tends to move with the market. Significant SMB or HML coefficients suggest exposure to size or value factors, respectively.
        
            Remember, while factor analysis provides insights into portfolio behavior, it's based on historical data and may not predict future performance.
            """)

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.error("Please check your inputs and try again.")
