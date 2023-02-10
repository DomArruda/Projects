

import pandas as pd
import yfinance as yf
import plotly.express as plx
import datetime
import time
import requests
import io
import torch
import streamlit as st
import pandas_datareader as pdr
from datetime import datetime
import pypfopt
from pypfopt.expected_returns import mean_historical_return
from pypfopt.risk_models import CovarianceShrinkage
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices
from pypfopt import HRPOpt
from pypfopt.efficient_frontier import EfficientCVaR
from PIL import Image
from random import randint
from pypfopt import HRPOpt
image = Image.open('PortfolioOptimizationApp/optGraph.jpg')
st.title('Python Portfolio Optimization')
st.image(image,caption = '', use_column_width = True)
    
    
from datetime import date
from datetime import timedelta


def create_portfolio(tick_list, start_date, end_date): 
  if '-' in start_date: 
    start_date.replace('-', '/')
  if '-' in end_date: 
    end_date.replace('-', '/')
  start_date = datetime.strptime(start_date, '%m/%d/%Y')
  end_date = datetime.strptime(end_date, '%m/%d/%Y')

  df_list = []
  for i in tick_list: 
    stock = yf.download(i, start = start_date, end = end_date, progress = False)
    if len(stock) == 0: 
      continue 

    else: 
      stock.rename( columns = {'Close': i}, inplace= True)
      stock = stock[i]
      df_list.append(stock)


  portfolio_list = pd.concat(df_list, axis = 1)
  portfolio_list.dropna(inplace= True)
  return portfolio_list





#Mean Variance Optimization


def MVO_opt(portfolio):
   
    mu = mean_historical_return(portfolio)
    S = CovarianceShrinkage(portfolio).ledoit_wolf()
    
    ef = EfficientFrontier(mu, S)
       
    weights = ef.max_sharpe()
    cleaned_weights = ef.clean_weights()
    
    cleaned_weights = pd.DataFrame().append(dict(cleaned_weights), ignore_index = True).T.reset_index()
    cleaned_weights.columns = ['Ticker', 'Allocation']
    st.text('')
    st.markdown('**Portfolio Performance**')
    st.write('Expected Annual Return (Discrete Allocation):           ',str(round(ef.portfolio_performance()[0] * 100, 3)) + '%')
    st.write('Annual Volatility:           ',str(round(ef.portfolio_performance()[1] * 100, 3)) + '%')
    st.write('Sharpe Ratio:           ',str(round(ef.portfolio_performance()[2], 3)))
    st.text('')
    st.text('')
    
    
    
    latest_prices = get_latest_prices(portfolio)
    da = DiscreteAllocation(weights, latest_prices, total_portfolio_value=port_value)
    allocation, leftover = da.greedy_portfolio()
    allocation = pd.DataFrame().append(dict(allocation), ignore_index = True).T.reset_index()
    allocation.columns = ['Ticker', 'Number of Stocks']
    
    st.markdown("**Discrete stock allocation:**")
    st.text('')
    allocation.sort_values(by = ['Number of Stocks'], inplace = True)
    st.dataframe(allocation)
    st.text('')
    st.write("Funds remaining (MVO): ${:.2f}".format(leftover))
    
    
    st.markdown("**Non-Discrete Allocation**") 
    st.dataframe(cleaned_weights)
    #have to fix from here
   

    


# %%

#Hierarchal Risk Parity:
def HRP(portfolio):
    mu = mean_historical_return(portfolio)
    S = CovarianceShrinkage(portfolio).ledoit_wolf()
    returns = portfolio.pct_change().dropna()
    
    hrp = HRPOpt(returns)
    hrp_weights = hrp.optimize()
    
    hrp.portfolio_performance(verbose=True)
    st.text('')
    st.markdown('**Portfolio Performance**')
    st.write('Expected Annual Return (Discrete Allocation):           ',str(round(hrp.portfolio_performance()[0] * 100, 3)) + '%')
    st.write('Annual Volatility:           ',str(round(hrp.portfolio_performance()[1] * 100, 3)) + '%')
    st.write('Sharpe Ratio:           ',str(round(hrp.portfolio_performance()[2], 3)))
    st.text('')
    st.text('')
    
    latest_prices = get_latest_prices(portfolio)
    da_hrp = DiscreteAllocation(hrp_weights, latest_prices, total_portfolio_value= port_value)
    allocation, leftover = da_hrp.greedy_portfolio()
    allocation = pd.DataFrame().append(dict(allocation), ignore_index = True).T.reset_index()
    allocation.columns = ['Ticker', 'Number of Stocks']
    print("Discrete allocation (HRP):", allocation)
    print("Funds remaining (HRP): ${:.2f}".format(leftover))
    
    
    
    st.markdown("**Discrete stock allocation:**")
    st.text('')
    allocation.sort_values(by = ['Number of Stocks'], inplace = True)
    st.dataframe(allocation)
    st.text('')
    st.write("Funds remaining (HRP): ${:.2f}".format(leftover))
    
    st.markdown("**Non-Discrete Allocation**") 
    hrp_weights_temp =   pd.DataFrame().append(dict(hrp_weights), ignore_index = True).T.reset_index()
    hrp_weights_temp.columns = ['Ticker', 'Percent Allocation']
    ND_weights = hrp_weights_temp.copy()
    ND_weights['Latest Prices'] = list(latest_prices) 
    ND_weights['Number of Stocks'] = (ND_weights['Percent Allocation'] * port_value)/ND_weights['Latest Prices']
    ND_weights.drop(['Percent Allocation', 'Latest Prices'] , axis = 1, inplace = True)
    ND_weights.sort_values(by = ['Number of Stocks'] , inplace = True)
    
    st.dataframe(ND_weights)
   
    
           
             



#Mean Conditional Value at Risk



def MCV(portfolio): 

    mu = mean_historical_return(portfolio)
    S = CovarianceShrinkage(portfolio).ledoit_wolf()
    returns = portfolio.pct_change().dropna()
    ef_cvar = EfficientCVaR(mu, S)
    cvar_weights = ef_cvar.min_cvar()
    cleaned_weights = ef_cvar.clean_weights()
    ef_cvar.portfolio_performance(verbose = True)
    st.write('Expected Annual Return (Discrete Allocation):           ',str(round(ef_cvar.portfolio_performance()[0] * 100, 3)) + '%')
    st.write('Annual Volatility:           ',str(round(ef_cvar.portfolio_performance()[1] * 100, 3)) + '%')
    
   
    
    latest_prices = get_latest_prices(portfolio)
    da_cvar = DiscreteAllocation(cvar_weights, latest_prices, total_portfolio_value=port_value)
    
    allocation, leftover = da_cvar.greedy_portfolio()
    allocation = pd.DataFrame().append(dict(allocation), ignore_index = True).T.reset_index()
    allocation.columns = ['Ticker', 'Number of stocks']
    st.write("Discrete allocation:")
    st.text('')
    st.dataframe(allocation)
    st.write(("Funds remaining: ${:.2f}".format(leftover)))
    st.text('')



ticker_str = st.text_input('Input your list of tickers. Format must follow: "Ticker1, Ticker2, Ticker3..."')
stock_button = st.button('Not sure what to pick? Download S&P 500 stock tickers')
if stock_button != False: 
    URL = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    ticker_list = pd.read_html(URL)[0]['Symbol'].tolist()
    st.markdown('**Tickers for S&P 500**')
    st.text('')
    st.markdown('**Click on the arrow to view stock tickers. After, click on top clipboard to copy all ticker symbols**')
    st.write(ticker_list)
stock_list = [i.strip() for i in list(ticker_str.split(','))]
stock_list = [i.strip('[') for i in stock_list] 
stock_list = [i.strip(']')  for i in stock_list] 
stock_list = [i.strip('"') for i in stock_list] 
stock_list = [i.replace('"', '') for i in stock_list] 
stock_list = [i.upper() for i in stock_list]
selected_stocks = st.multiselect('Chosen Stocks: ', options  = stock_list, default = stock_list)



start_date = st.text_input('Write your stock start date in month/day/year format', (datetime.today() - timedelta(days = 2 * 366)).strftime('%m/%d/%Y'))
end_date = st.text_input('Write your stock end date in month/day/year format', (datetime.today() - timedelta(days = 1)).strftime('%m/%d/%Y'))



if "" not in selected_stocks  and start_date != False and end_date != False:
    try:
        portfolio = create_portfolio(selected_stocks, start_date, end_date)
        
        #st.write(type(portfolio))
        if portfolio is None:
            st.text('')
            st.write('Invalid Stock Selection')
            st.stop()
            
        #portfolio.to_csv("portfolio.csv")
        #portfolio = pd.read_csv("portfolio.csv")    
        #portfolio.index = portfolio['Date']
        #portfolio.drop(['Date'], inplace = True, axis = 1)
        st.write(portfolio)
        fig = plx.imshow(portfolio.corr(method = 'spearman').round(2), title = 'Stock Correlations:', text_auto = True)
        st.plotly_chart(fig)
        port_value = st.text_input('What amount do you plan on investing in your portfolio?')
        if port_value == '' or port_value is None: 
            st.stop()
        else:
            port_value = float(port_value)
            opt_list = ['None Selected', 'Hierarchical Risk Parity', 'Mean Conditional Value at Risk'] 
            choice = st.selectbox('Choose Which Optimization Technique You Would Like To Use: ', opt_list)
            
            #'Mean Variance Optimization' - Need to fix this one up...
            if choice == opt_list[1]: 
                try:
                    HRP(portfolio)
                except: 
                    st.text('Error Occured: Please try again')
            elif choice == opt_list[2]:
                try:
                    MCV(portfolio)
                except: 
                    st.text('Error Occured - Please try again')
    except: 
            st.text('')
      
else: 
    st.stop()

    
