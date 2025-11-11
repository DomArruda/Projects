# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.optimize import minimize
import statsmodels.api as sm
import pandas_datareader as pdr
import warnings
from PIL import Image
from datetime import datetime

warnings.filterwarnings('ignore')
image = Image.open('PortfolioOptimizationApp/optGraph.jpg')

st.set_page_config(
    page_title="Portfolio Optimization & Analysis",
    page_icon="📈"  #stonks!
)

st.title('Portfolio Optimization & Analysis')
st.image(image, caption='', use_container_width=True)


# ----------------------------- Data helpers -----------------------------
@st.cache_data
def fetch_factor_data(start_date, end_date):
    ff = pdr.get_data_famafrench('F-F_Research_Data_Factors_daily', start=start_date, end=end_date)[0]
    ff.index = pd.to_datetime(ff.index).tz_localize(None)
    ff = ff / 100.0  # to decimals; columns: Mkt-RF, SMB, HML, RF
    return ff

@st.cache_data
def fetch_data(tickers, start, end):
    data = yf.download(tickers, start=start, end=end, progress=False)
    data.index = data.index.tz_localize(None)
    return data['Close']

@st.cache_data
def get_risk_free_rate_scalar(start, end):
    # Fallback only if Fama-French is unavailable in the window
    try:
        tnx = yf.Ticker("^TNX").history(start=start, end=end)
        tnx.index = tnx.index.tz_localize(None)
        # ^TNX is 10× percent; 45.00 => 4.5% => 0.045
        return (tnx['Close'].iloc[-1] / 1000.0) / 252.0
    except Exception:
        return 0.02 / 252.0

def align_rf(rf, idx):
    if isinstance(rf, pd.Series):
        return rf.reindex(idx).ffill().bfill().fillna(0.0)
    return pd.Series(float(rf), index=idx)

# ----------------------------- Metrics -----------------------------
def calculate_max_drawdown(returns):
    cum = (1 + returns).cumprod()
    peak = cum.cummax()
    dd = cum / peak - 1.0
    return dd.min()

def calculate_performance(weights, returns, risk_free):
    port = (returns * weights).sum(axis=1)
    rf = align_rf(risk_free, port.index)
    excess = port - rf
    total_return = (1.0 + port).prod() - 1.0
    years = len(port) / 252.0 if len(port) else 0.0
    annualized_return = (1.0 + total_return)**(1.0 / max(years, 1e-9)) - 1.0 if years > 0 else np.nan
    vol = port.std(ddof=1) * np.sqrt(252.0) if len(port) > 1 else np.nan
    s_den = excess.std(ddof=1)
    sharpe = (excess.mean() / s_den) * np.sqrt(252.0) if s_den and s_den > 0 else np.nan
    downside = excess[excess < 0]
    d_den = downside.std(ddof=1) if len(downside) > 1 else 0
    sortino = (excess.mean() / d_den) * np.sqrt(252.0) if d_den and d_den > 0 else np.nan
    mdd = calculate_max_drawdown(port)
    calmar = (annualized_return / abs(mdd)) if mdd and mdd != 0 else np.nan
    return {'Total Return': total_return, 'Annualized Return': annualized_return, 'Volatility': vol,
            'Sharpe Ratio': sharpe, 'Sortino Ratio': sortino, 'Max Drawdown': mdd, 'Calmar Ratio': calmar}

# ----------------------------- Portfolio constructors -----------------------------
def minimum_variance_portfolio(returns):
    R = returns if isinstance(returns, pd.DataFrame) else returns.to_frame()
    n = R.shape[1]; x0 = np.full(n, 1.0/n)
    bounds = tuple((0, 1) for _ in range(n))
    cons = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0},)
    def obj(w):
        p = (R * w).sum(axis=1)
        return p.std(ddof=1) * np.sqrt(252.0)
    res = minimize(obj, x0, method='SLSQP', bounds=bounds, constraints=cons)
    return pd.Series(res.x, index=R.columns)

def maximum_sharpe_ratio_portfolio(returns, risk_free):
    R = returns if isinstance(returns, pd.DataFrame) else returns.to_frame()
    n = R.shape[1]; x0 = np.full(n, 1.0/n)
    bounds = tuple((0, 1) for _ in range(n))
    cons = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0},)
    rf = align_rf(risk_free, R.index)
    def neg_sharpe(w):
        p = (R * w).sum(axis=1); ex = p - rf
        s = ex.std(ddof=1)
        return - (ex.mean() / s) * np.sqrt(252.0) if s and s > 0 else 1e6
    res = minimize(neg_sharpe, x0, method='SLSQP', bounds=bounds, constraints=cons)
    return pd.Series(res.x, index=R.columns)

def risk_parity_portfolio(returns):
    R = returns if isinstance(returns, pd.DataFrame) else returns.to_frame()
    n = R.shape[1]; Sigma = (R.cov().values) * 252.0
    def port_vol(w): return np.sqrt(w @ Sigma @ w)
    def risk_contrib(w): v = port_vol(w); return (w * (Sigma @ w)) / v
    target = np.full(n, 1.0/n); x0 = np.full(n, 1.0/n)
    bounds = tuple((0, 1) for _ in range(n))
    cons = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0},)
    def obj(w): rc = risk_contrib(w); return ((rc - target)**2).sum()
    res = minimize(obj, x0, method='SLSQP', bounds=bounds, constraints=cons)
    return pd.Series(res.x, index=R.columns)

def black_litterman_portfolio(returns, market_caps, risk_aversion=2.5, tau=0.05):
    # Placeholder without explicit views: revert to market-cap weights
    w = market_caps.copy()
    if w.sum() <= 0 or w.isna().all():
        w = pd.Series(1.0/len(returns.columns), index=returns.columns)
    return w / w.sum()

def eigen_portfolio(returns):
    R = returns if isinstance(returns, pd.DataFrame) else returns.to_frame()
    comp = PCA(n_components=1).fit(R).components_[0]
    w = pd.Series(comp, index=R.columns).clip(lower=0)
    return (w / w.sum()) if w.sum() > 0 else pd.Series(1.0/len(w), index=w.index)

def momentum_portfolio(returns, lookback=252):
    R = returns if isinstance(returns, pd.DataFrame) else returns.to_frame()
    lookback = min(lookback, len(R))
    mom = (1.0 + R.iloc[-lookback:]).prod() - 1.0
    w = mom.clip(lower=0)
    return (w / w.sum()) if w.sum() > 0 else pd.Series(1.0/len(w), index=w.index)

# ----------------------------- Factor analysis & bootstrap -----------------------------
def factor_analysis(portfolio_returns, factor_df):
    aligned = pd.concat([portfolio_returns, factor_df[['Mkt-RF', 'SMB', 'HML', 'RF']]], axis=1).dropna()
    y = aligned.iloc[:, 0] - aligned['RF']  # excess portfolio return
    X = sm.add_constant(aligned[['Mkt-RF', 'SMB', 'HML']])
    model = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": 5})
    return model

def bootstrap_sharpe_ratio(returns, weights, risk_free, num_simulations=2000):
    port = (returns * weights).sum(axis=1)
    rf = align_rf(risk_free, port.index)
    ex = (port - rf).values
    n = len(ex)
    if n == 0: return np.nan, (np.nan, np.nan)
    rng = np.random.default_rng(); vals = []
    for _ in range(num_simulations):
        sample = rng.choice(ex, size=n, replace=True)
        s = sample.std(ddof=1)
        if s and s > 0:
            vals.append((sample.mean() / s) * np.sqrt(252.0))
    return (np.mean(vals), np.percentile(vals, [2.5, 97.5])) if vals else (np.nan, (np.nan, np.nan))

def stress_test(weights, returns, scenarios):
    port = (returns * weights).sum(axis=1)
    out = {}
    for name, sc in scenarios.items():
        if isinstance(sc, str) and sc.startswith('oneday_'):
            shock = float(sc.split('_')[1]) / 100.0  # e.g. -10 => -0.10
            stressed = port.copy()
            if len(stressed) > 0:
                stressed.iloc[0] = stressed.iloc[0] + shock
            out[name] = (1.0 + stressed).prod() - 1.0
        else:
            out[name] = (1.0 + port * float(sc)).prod() - 1.0
    return out

# ----------------------------- UI: Inputs -----------------------------
st.header('Portfolio Setup')
ticker_input = st.text_input('Enter stock tickers (comma-separated)', '')
stock_tickers = [t.strip().upper() for t in ticker_input.split(",") if t.strip()]

st.subheader('Date Range Selection')
col1, col2 = st.columns(2)
with col1:
    analysis_start = st.date_input('Analysis Start Date', datetime(2021, 1, 1).date(), key='analysis_start')
    backtest_start = st.date_input('Backtest Start Date', datetime(2023, 1, 1).date(), key='backtest_start')
with col2:
    analysis_end = st.date_input('Analysis End Date', datetime(2022, 12, 31).date(), key='analysis_end')
    backtest_end = st.date_input('Backtest End Date', datetime(2023, 12, 31).date(), key='backtest_end')

analysis_start = datetime.combine(analysis_start, datetime.min.time())
analysis_end = datetime.combine(analysis_end, datetime.min.time())
backtest_start = datetime.combine(backtest_start, datetime.min.time())
backtest_end = datetime.combine(backtest_end, datetime.min.time())

# ----------------------------- Run -----------------------------
if st.button('Run Analysis'):
    try:
        if not stock_tickers:
            st.warning("Please enter at least one valid ticker."); st.stop()
        if analysis_end <= analysis_start or backtest_end <= backtest_start:
            st.error("End date must be after start date for both analysis and backtest."); st.stop()

        data = fetch_data(stock_tickers, analysis_start, backtest_end).dropna(axis=1, how='all')
        if data.empty: st.error("No price data retrieved for given tickers/dates."); st.stop()
        returns = data.pct_change().dropna(how='all')
        st.dataframe(data)

        analysis_returns = returns[(returns.index >= analysis_start) & (returns.index <= analysis_end)].dropna(how='any', axis=1)
        backtest_returns = returns[(returns.index >= backtest_start) & (returns.index <= backtest_end)].dropna(how='any', axis=1)
        common_cols = analysis_returns.columns.intersection(backtest_returns.columns)
        analysis_returns, backtest_returns = analysis_returns[common_cols], backtest_returns[common_cols]
        if len(common_cols) == 0: st.error("No overlapping tickers after cleaning."); st.stop()

        # Risk-free series from Fama-French (preferred), with fallback scalar if missing
        factor_all = fetch_factor_data(analysis_start, backtest_end)
        rf_series = factor_all['RF'] if not factor_all.empty else None
        if rf_series is None or rf_series.empty:
            rf_scalar = get_risk_free_rate_scalar(analysis_start, backtest_end)
        else:
            rf_scalar = None

        # Market caps (fallback equal-weight if unavailable)
        @st.cache_data
        def get_market_cap(ticker):
            try:
                t = yf.Ticker(ticker)
                mc = getattr(t, "fast_info", None)
                val = getattr(mc, "market_cap", None) if mc else None
                return val if val and val > 0 else t.info.get('marketCap', None)
            except Exception:
                return None
        mkt_caps = pd.Series({t: get_market_cap(t) for t in common_cols}).astype('float')
        if mkt_caps.isna().all() or mkt_caps.le(0).all():
            mkt_caps = pd.Series(1.0/len(common_cols), index=common_cols)
        else:
            mkt_caps = mkt_caps.fillna(mkt_caps.median()).clip(lower=1.0).loc[common_cols]
            mkt_caps = mkt_caps / mkt_caps.sum()

        # Build portfolios on analysis window
        rf_for_anal = rf_series if rf_scalar is None else rf_scalar
        portfolios = {
            'Eigenportfolio': eigen_portfolio(analysis_returns),
            'Equal-Weight': pd.Series(1/len(common_cols), index=common_cols),
            'Minimum Variance': minimum_variance_portfolio(analysis_returns),
            'Maximum Sharpe Ratio': maximum_sharpe_ratio_portfolio(analysis_returns, rf_for_anal),
            'Risk Parity': risk_parity_portfolio(analysis_returns),
            'Black-Litterman': black_litterman_portfolio(analysis_returns, mkt_caps),
            'Momentum': momentum_portfolio(analysis_returns, lookback=min(252, len(analysis_returns)))
        }
        portfolios = {k: (w / w.sum()).reindex(common_cols).fillna(0.0) for k, w in portfolios.items()}

        # Evaluate on backtest
        rf_for_back = rf_series if rf_scalar is None else rf_scalar
        results = {name: calculate_performance(weights, backtest_returns, rf_for_back) for name, weights in portfolios.items()}

        # Plot cumulative performance
        st.header('Portfolio Performance Comparison')
        fig, ax = plt.subplots(figsize=(12, 6))
        for name, weights in portfolios.items():
            cum = (1 + (backtest_returns * weights).sum(axis=1)).cumprod()
            ax.plot(cum.index, cum, label=f'{name} ({results[name]["Total Return"]:.2%})')
        ax.set_title('Portfolio Performance Comparison'); ax.set_xlabel('Date'); ax.set_ylabel('Cumulative Return')
        ax.legend(); ax.grid(True); st.pyplot(fig)

        # Factor Analysis (Fama-French 3 factors)
        st.header('Factor Analysis')
        ff_back = factor_all[(factor_all.index >= backtest_start) & (factor_all.index <= backtest_end)]
        for name, weights in portfolios.items():
            st.subheader(f'{name}')
            port_ret = (backtest_returns * weights).sum(axis=1)
            if ff_back.empty:
                st.info("Fama-French data unavailable for this window."); continue
            model = factor_analysis(port_ret, ff_back)
            out = pd.DataFrame({
                'coef': [round(val,3) for val in model.params],
                't': [round(val,3) for val in model.tvalues],
                'pval': [round(val,3) for val in model.pvalues]
            })
            out.loc['R-squared', ['coef','t','pval']] = [model.rsquared, np.nan, np.nan]
            st.dataframe(out)

        # Bootstrap Analysis
        st.header('Bootstrap Analysis')
        bootstrap_rows = []
        for name, weights in portfolios.items():
            mean_s, ci = bootstrap_sharpe_ratio(backtest_returns, weights, rf_for_back)
            bootstrap_rows.append({'Strategy': name, 'Mean Sharpe Ratio': mean_s, 'CI Lower': ci[0], 'CI Upper': ci[1]})
        st.dataframe(pd.DataFrame(bootstrap_rows))

        # Stress Testing
        st.header('Stress Test Results')
        scenarios = {'Market Crash (-10% day-1)': 'oneday_-10', 'Economic Boom (×1.3 daily)': 1.3, 'High Volatility (no drift change)': 1.0}
        stress_test_results = {name: stress_test(weights, backtest_returns, scenarios) for name, weights in portfolios.items()}
        st.dataframe(pd.DataFrame(stress_test_results).T)

        # Summary + Best Strategy
        st.header('Portfolio Performance Summary')
        summary_df = pd.DataFrame(results).T
        st.dataframe(summary_df)

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
            Factor analysis estimates how market factors explain your portfolio's excess returns:
            1. **Coefficients**: Sensitivity to factors; positive = co-moves, negative = opposite move.
            2. **t-statistics / p-values**: Statistical significance (|t|>2, p<0.05 often used).
            3. **R-squared**: Share of variance explained by the model.
            4. **Factors**:
               - Mkt-RF: Market excess return
               - SMB: Size (small minus big)
               - HML: Value (high minus low)
            Historical results don’t guarantee future performance.
            """)

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.error("Please check your inputs and try again.")
