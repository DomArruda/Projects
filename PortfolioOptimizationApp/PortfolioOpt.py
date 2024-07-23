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

warnings.filterwarnings('ignore')
image = Image.open('PortfolioOptimizationApp/optGraph.jpg')
st.title('Python Portfolio Optimization')
st.image(image,caption = '', use_column_width = True)
    
def fetch_factor_data(start_date, end_date):
    ff_factors = pdr.get_data_famafrench('F-F_Research_Data_Factors_daily', start=start_date, end=end_date)[0]
    ff_factors.index = pd.to_datetime(ff_factors.index)
    ff_factors = ff_factors / 100  # Convert to decimal format
    return ff_factors

def calculate_performance(weights, returns, risk_free_rate):
    portfolio_return = (returns * weights).sum(axis=1)
    excess_return = portfolio_return - risk_free_rate
    trading_days = len(returns)
    annualization_factor = np.sqrt(252 / trading_days)

    total_return = (1 + portfolio_return).prod() - 1
    annualized_return = (1 + total_return) ** (252 / trading_days) - 1
    volatility = portfolio_return.std() * np.sqrt(252)
    sharpe_ratio = excess_return.mean() / excess_return.std() * annualization_factor
    sortino_ratio = excess_return.mean() / excess_return[excess_return < 0].std() * annualization_factor
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
    n = returns.shape[1]
    args = (returns,)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0, 1) for _ in range(n))
    result = minimize(lambda weights, returns: (returns * weights).sum(axis=1).std() * np.sqrt(252),
                      n*[1./n,], args=args, method='SLSQP', bounds=bounds, constraints=constraints)
    return result.x

def maximum_sharpe_ratio_portfolio(returns, risk_free_rate):
    n = returns.shape[1]
    args = (returns, risk_free_rate)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0, 1) for _ in range(n))
    result = minimize(lambda weights, returns, rf: -((returns * weights).sum(axis=1).mean() - rf) /
                      ((returns * weights).sum(axis=1).std() * np.sqrt(252)),
                      n*[1./n,], args=args, method='SLSQP', bounds=bounds, constraints=constraints)
    return result.x

def risk_parity_portfolio(returns):
    n = returns.shape[1]
    target_risk = 1/n
    def objective(weights):
        portfolio_vol = (returns * weights).sum(axis=1).std() * np.sqrt(252)
        asset_contribs = weights * (returns.cov() * 252).dot(weights) / portfolio_vol
        return np.sum((asset_contribs - target_risk)**2)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0, 1) for _ in range(n))
    result = minimize(objective, n*[1./n,], method='SLSQP', bounds=bounds, constraints=constraints)
    return result.x

def black_litterman_portfolio(returns, market_caps, risk_aversion=2.5, tau=0.05):
    Sigma = returns.cov() * 252
    Pi = risk_aversion * Sigma.dot(market_caps)
    weights = np.linalg.inv(tau * Sigma + Sigma).dot(tau * Sigma.dot(market_caps) + Pi)
    return weights / weights.sum()

def momentum_portfolio(returns, lookback=12):
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
    sharpe_ratios = []
    for _ in range(num_simulations):
        sample = np.random.choice(portfolio_returns, size=len(portfolio_returns), replace=True)
        sharpe = np.sqrt(252) * sample.mean() / sample.std()
        sharpe_ratios.append(sharpe)
    return np.mean(sharpe_ratios), np.percentile(sharpe_ratios, [2.5, 97.5])

def stress_test(weights, returns, scenarios):
    portfolio_returns = (returns * weights).sum(axis=1)
    results = {}
    for name, scenario in scenarios.items():
        stressed_returns = portfolio_returns * scenario
        results[name] = stressed_returns.sum()
    return results    
    
from datetime import date
from datetime import timedelta


def create_portfolio(tick_list, start_date, end_date, future_date = None): 
  if '-' in start_date: 
    start_date.replace('-', '/')
  if '-' in end_date: 
    end_date.replace('-', '/')
  if future_date != None:
    future_date.replace('-', '/')
    future_date = datetime.strptime(future_date, '%m/%d/%Y') 

  start_date = datetime.strptime(start_date, '%m/%d/%Y')
  end_date = datetime.strptime(end_date, '%m/%d/%Y')

st.title('Advanced Portfolio Optimization')

# Input section
st.header('Portfolio Setup')
ticker_input = st.text_input('Enter stock tickers (comma-separated)', 'WMT, DIS, KO, NFLX, MTCH, TGT, OXY, TDG, NOC, HWM, QCOM, META, AMZN')
stock_tickers = [stock.strip() for stock in ticker_input.split(",")]

col1, col2 = st.columns(2)
with col1:
    analysis_start = st.date_input('Analysis Start Date', datetime(2021, 1, 1))
    analysis_end = st.date_input('Analysis End Date', datetime(2022, 12, 31))
with col2:
    backtest_start = st.date_input('Backtest Start Date', datetime(2023, 1, 1))
    backtest_end = st.date_input('Backtest End Date', datetime(2023, 12, 31))

if st.button('Run Analysis'):
    try:
        # Data fetching and preprocessing
        yf.pdr_override()
        data = pdr.get_data_yahoo(stock_tickers, start=analysis_start, end=backtest_end)['Adj Close']
        data = data.dropna(axis=1)
        returns = data.pct_change().dropna()

        analysis_returns = returns.loc[analysis_start:analysis_end]
        backtest_returns = returns.loc[backtest_start:backtest_end]

        risk_free_rate = yf.Ticker("^TNX").history(start=analysis_start, end=backtest_end)['Close'].iloc[-1] / 100 / 252

        market_caps = pd.Series({ticker: yf.Ticker(ticker).info.get('marketCap', 1e9) for ticker in data.columns})
        market_caps = market_caps / market_caps.sum()

        # Portfolio creation
        portfolios = {
            'Eigenportfolio': pd.Series(PCA(n_components=1).fit(analysis_returns).components_[0], index=data.columns),
            'Equal-Weight': pd.Series(1/len(data.columns), index=data.columns),
            'Minimum Variance': pd.Series(minimum_variance_portfolio(analysis_returns), index=data.columns),
            'Maximum Sharpe Ratio': pd.Series(maximum_sharpe_ratio_portfolio(analysis_returns, risk_free_rate), index=data.columns),
            'Risk Parity': pd.Series(risk_parity_portfolio(analysis_returns), index=data.columns),
            'Black-Litterman': pd.Series(black_litterman_portfolio(analysis_returns, market_caps), index=data.columns),
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
        ax.set_title('Portfolio Performance Comparison (2023)')
        ax.set_xlabel('Date')
        ax.set_ylabel('Cumulative Return')
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)

        # Factor Analysis
        st.header('Factor Analysis')
        factor_data = fetch_factor_data(backtest_start, backtest_end)
        for name, weights in portfolios.items():
            portfolio_returns = (backtest_returns * weights).sum(axis=1)
            factor_model = factor_analysis(portfolio_returns, factor_data)
            st.subheader(f'{name}')
            st.write(factor_model.summary().tables[1])

        # Bootstrap Analysis
        st.header('Bootstrap Analysis')
        for name, weights in portfolios.items():
            mean_sharpe, ci = bootstrap_sharpe_ratio(backtest_returns, weights)
            st.subheader(f'{name}')
            st.write(f"Mean Sharpe Ratio: {mean_sharpe:.4f}")
            st.write(f"95% Confidence Interval: ({ci[0]:.4f}, {ci[1]:.4f})")

        # Stress Testing
        st.header('Stress Test Results')
        scenarios = {
            'Market Crash': 0.7,
            'Economic Boom': 1.3,
            'High Volatility': 1.0
        }
        stress_test_results = {name: stress_test(weights, backtest_returns, scenarios) for name, weights in portfolios.items()}
        for name, scenario_results in stress_test_results.items():
            st.subheader(f'{name}')
            for scenario, result in scenario_results.items():
                st.write(f"  {scenario}: {result:.2%}")

        # Summary statistics
        st.header('Portfolio Performance Summary')
        for name, result in results.items():
            st.subheader(f'{name}')
            for metric, value in result.items():
                st.write(f"  {metric}: {value:.4f}")

        # Best strategy
        best_strategy = max(results, key=lambda x: results[x]['Sharpe Ratio'])
        st.header('Best Strategy')
        st.write(f"Best Strategy: {best_strategy}")
        st.write(f"Best Strategy Sharpe Ratio: {results[best_strategy]['Sharpe Ratio']:.4f}")
        st.write(f"Best Strategy Total Return: {results[best_strategy]['Total Return']:.4%}")

        st.subheader('Best Strategy Weights')
        best_weights = portfolios[best_strategy]
        weight_df = pd.DataFrame(best_weights.sort_values(ascending=False)).reset_index()
        weight_df.columns = ['Stock', 'Weight']
        weight_df['Weight'] = weight_df['Weight'].apply(lambda x: f"{x:.4f}")
        st.table(weight_df)

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.error("Please check your inputs and try again.")
