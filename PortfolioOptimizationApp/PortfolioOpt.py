# -*- coding: utf-8 -*-
"""
Created on Sun Mar 12 09:29:19 2023
@author: darruda
"""



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
import numpy as np 
image = Image.open('PortfolioOptimizationApp/optGraph.jpg')
st.title('Python Portfolio Optimization')
st.image(image,caption = '', use_column_width = True)
    
    
    
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




  df_list = []

  if future_date != None:
    for i in tick_list: 
      stock = yf.download(i, start = start_date, end = future_date, progress = False)
      if len(stock) == 0: 
        continue 

      else: 
        stock.rename( columns = {'Close': i}, inplace= True)
        stock = stock[i]
        df_list.append(stock)

    portfolio_list = pd.concat(df_list, axis = 1)
    portfolio_list.dropna(inplace= True)
    portfolio = portfolio_list

    og_portfolio = portfolio
    portfolio = og_portfolio[og_portfolio.index <= end_date]
    portfolio_end = og_portfolio[og_portfolio.index >= end_date]

    return (portfolio, portfolio_end)

  else:
    for i in tick_list: 
      stock = yf.download(i, start = start_date, end = end_date, progress = False)
      if len(stock) == 0: 
        continue 

      else: 
        stock.rename( columns = {'Close': i}, inplace= True)
        stock = stock[i]
        df_list.append(stock)

      

    portfolio_list = pd.concat(df_list, axis = 1)
    print(portfolio_list)
    portfolio_list.dropna(inplace= True)
    portfolio = portfolio_list

    return portfolio




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
def HRP(portfolio, port_value, future_portfolio = None):
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
  allocation_dict = allocation
  allocation = pd.DataFrame( tuple(allocation.items())) 
  allocation.columns = ['Ticker', 'Number of Stocks']
  print("\nDiscrete allocation (HRP):", allocation)
  print("\nFunds remaining (HRP): ${:.2f}".format(leftover))


  st.markdown("**Discrete stock allocation:**")
  st.text('')
  allocation.sort_values(by = ['Number of Stocks'], inplace = True)
  st.dataframe(allocation)
  st.text('')
  st.write("\nFunds remaining (HRP): ${:.2f}".format(leftover))

  st.markdown("**Non-Discrete Allocation**") 



  hrp_weights_df = pd.DataFrame(tuple(hrp_weights.items()), columns = ['Ticker', 'Percent Allocation'])
  ND_weights = hrp_weights_df.copy()
  ND_weights_og = dict(ND_weights)
  ND_weights['Latest Prices'] = list(latest_prices) 
  ND_weights['Number of Stocks'] = (ND_weights['Percent Allocation'] * port_value)/ND_weights['Latest Prices']
  ND_weights.drop(['Percent Allocation', 'Latest Prices'] , axis = 1, inplace = True)
  ND_weights.sort_values(by = ['Number of Stocks'] , inplace = True)

  if future_portfolio is not None: 
    discrete_purchases = {stock_name:(portfolio[stock_name].iloc[-1] * discrete_allocation)  for stock_name, discrete_allocation in allocation_dict.items() if stock_name in list(portfolio.columns)}
    initial_value_DA = sum(discrete_purchases.values())
    discrete_end_values = {stock_name: discrete_purchases[stock_name] * (future_portfolio[stock_name].iloc[-1]/portfolio[stock_name].iloc[-1]) for stock_name in discrete_purchases.keys() if stock_name in list(portfolio.columns) }
    end_value_DA = sum(discrete_end_values.values())
    percent_change_DA = ((end_value_DA - initial_value_DA)/initial_value_DA) * 100 
    print(f'\n\nPortfolio By End of Test Date (DA): ${end_value_DA:.0f}')
    print(f'\nPercent Change: (DA) {percent_change_DA:.2f}%')



  print('\n\n\nNon-Discrete Allocation:\n')
  print(ND_weights)
    
           
             



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
    
def hedgeify(portfolio, corr, lower_bound, upper_bound):
  corr_matrix = portfolio.corr(method = corr)
  #.rename_axis(None).rename_axis(None, axis = 1)
  correlationData = pd.DataFrame(corr_matrix.stack().reset_index()).rename(columns = {'level_0': 'Stock1', 'level_1': 'Stock2', 0: 'Correlation'}) 
  correlationData['Same Stock'] = correlationData['Stock1'] == correlationData['Stock2']
  correlationData = correlationData[ correlationData['Same Stock'] != True]

  
  correlationData = correlationData[ (correlationData['Correlation'] >= lower_bound) & (correlationData['Correlation'] <= upper_bound)]

  allStocks = list(set(correlationData['Stock1'].to_list() + correlationData['Stock2'].to_list()))


  if 'Date' in list(portfolio.columns): 
    portfolio = portfolio[allStocks + 'Date']
  else: 
    portfolio= portfolio[allStocks]

  return portfolio 

#%%

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



start_date = st.text_input('Write your stock start date in month/day/year format', (datetime.today() - timedelta(days = 2 * 365 + 1)).strftime('%m/%d/%Y'))
end_date = st.text_input('Write your stock end date in month/day/year format', (datetime.today() - timedelta(days = 1)).strftime('%m/%d/%Y'))
future_date = st.text_input('Write a backtest date (after end date) in month/day/year format (OPTIONAL)', (datetime.today() - timedelta(days = 1)).strftime('%m/%d/%Y'))
                        


if ("" not in selected_stocks)  and (start_date != False) and (end_date != False):
    try:
        future_portfolio = None
        if (future_date == False) or (future_date == '') or (end_date == future_date): 
            future_date = None 

        portfolio = create_portfolio(selected_stocks, start_date, end_date, future_date)
        if len(portfolio) == 2 and not isinstance(portfolio, pd.DataFrame):
            portfolio, future_portfolio = portfolio 
        
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
        portfolioData = portfolio.to_csv(index = True).encode('utf-8')
        st.download_button('Click Here To Download Stock Data', 
                       portfolioData, 'StockData.csv')
        
        correlation_types =   ['Pearson', 'Kendall', 'Spearman']
    
        corr_option = st.selectbox( 'Pick Correlation Method: ',correlation_types)
          
  
        fig = plx.imshow(portfolio.corr(method = corr_option.lower()).round(2), title = f'Stock Correlations - {corr_option.title()}:', text_auto = True)
        st.plotly_chart(fig)
     
        correlation_values = np.arange(start = -1.00, stop = 1.01, step = 0.01)
        correlation_values = [round(i,2) for i in correlation_values]
        lower_bound, upper_bound = ( st.select_slider('Filter Portfolio By Correlation Range', options = correlation_values,
                         value = (min(correlation_values), max(correlation_values))))
        lower_bound, upper_bound = float(lower_bound), float(upper_bound)
         
        if lower_bound != min(correlation_values) or upper_bound != max(correlation_values):
            portfolio = hedgeify(portfolio, corr = corr_option.lower(), lower_bound = lower_bound, upper_bound= upper_bound) 
            st.write(portfolio)
            portfolioData = portfolio.to_csv(index = True).encode('utf-8')
            st.download_button('Click Here To Download Filtered Stock Data', 
                           portfolioData, 'StockData.csv')
            fig = plx.imshow(portfolio.corr(method = corr_option.lower()).round(2), title = f'Filtered Portfolio Stock Correlations - {corr_option.title()}:', text_auto = True)
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
                    HRP(portfolio, port_value, future_portfolio)
                except Exception as e: 
                    st.text(e)
                    st.text('Error Occured: Please try again')
            elif choice == opt_list[2]:
                try:
                    MCV(portfolio)
                except Exception as e_: 
                    st.text(e_)
                    #st.text('Error Occured - Please try again')
    except Exception as e : 
            st.text(e)
      
else: 
    st.stop()
