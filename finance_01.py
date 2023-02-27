import streamlit as st
import pandas_datareader.data as pdr
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import yfinance as yf
yf.pdr_override()

import plotly.offline as py
#import plotly.io as io
#io.renderers.default='browser'
import plotly.graph_objs as go
#from plotly.subplots import make_subplots

py.init_notebook_mode(connected = True)
#fig = make_subplots(rows = 2)

st.set_page_config(page_title='Stock Analysis App', page_icon='🖖', layout="wide", initial_sidebar_state="expanded", menu_items=None)

Menu = ['Home', 'Login']
user_list = ['admin', 'fabi']
pass_list = ['123', 'fabi']

#login = True
choice = st.sidebar.selectbox('', Menu)
if choice == 'Home':
    st.subheader("""Stock Analysis""")
    st.write('Select Login to use this app.')
    login = False
elif choice == 'Login':
    login = False
    username = st.sidebar.text_input('Username')
    password = st.sidebar.text_input('Password', type = 'password')
    if username in user_list and password in pass_list:
        login = True
        st.success('Logged in as {}'.format(username))
        
    else:
        login = False
        st.warning('Enter a valid username and password.')
  

def simple_config_plot(fig, title):
    title = {'text': title, 'xanchor':'center', 'yanchor':'top','y':0.99, 'x':0.5,}
    fig.update_layout(title = title, font_family="Courier New", font_color="blue", xaxis_rangeslider_visible = False, width = 1150, height = 420, xaxis_showgrid = True, xaxis_gridwidth = 1, xaxis_gridcolor = '#E8E8E8', xaxis_linecolor = 'black', xaxis_tickfont = dict(size=18), yaxis_showgrid = True, yaxis_gridwidth = 1, yaxis_tickfont = dict(size=16), yaxis_gridcolor = '#E8E8E8', yaxis_linecolor = 'black', plot_bgcolor='rgba(0,0,0,0)', margin=dict(l=0, r=10, b=50, t=40), showlegend = True)
    
def simple_plot(data, title):
    fig = go.Figure(data = data)
    #fig.append_trace(data,1,1)
    simple_config_plot(fig, title)
    #fig.show(renderer='browser')
    return fig
    
def get(tickers, start_date, end_date):
    def data(ticker):
        return (pdr.get_data_yahoo(ticker, start = start_date, end = end_date))
    datas = map(data,tickers)
    return(pd.concat(datas, keys = tickers, names=['Ticker', 'Date']))

def get_ema(window,prices):
    kk = (2 / (window + 1))
    ma = prices.rolling(window = window).mean().dropna()
    
    datam = pd.DataFrame(index = ma.index)
    datam['Price'] = prices
    datam['EMA'] = np.NaN
    
    datam.EMA[0] = ma[1]
    
    for i in range(1, len(datam)):
        datam.EMA[i] = (datam.Price[i] * kk) + ((1 - kk)*datam.EMA[i-1])
    return datam

if login == True:
    ### SETUP ###
    tickers_list = ['VALE3.SA', 'PETR3.SA', 'PETR4.SA', '^BVSP', 'CMIG4.SA', 'ITSA4.SA', 'VIIA3.SA','CPLE6.SA', 'MGLU3.SA', 'BBDC4.SA', 'BBDC3.SA', 'B3SA3.SA', 'WEGE3.SA', 'ELET3.SA', 'ITUB4.SA', 'BBAS3.SA', 'JBSS3.SA', 'CIEL3.SA', 'RADL3.SA', 'BEEF3.SA', 'ABEV3.SA', 'LREN3.SA', 'TIMS3.SA', 'HYPE3.SA', 'GGBR4.SA', 'MULT3.SA', 'RRRP3.SA', 'UGPA3.SA','PETZ3.SA', 'RAIZ4.SA', 'NTCO3.SA', 'EGIE3.SA', 'BRAP4.SA','TAEE11.SA','PRIO3.SA','CCRO3.SA','RAIL3.SA','CYRE3.SA','ENEV3.SA','USIM5.SA','ALPA4.SA','BRKM5.SA','ARZZ3.SA', 'AZUL4.SA','CSNA3.SA', 'RDOR3.SA','MRVE3.SA','SOMA3.SA', 'GOAU4.SA', 'EMBR3.SA', 'VIVT3.SA','GOLL4.SA','TOTS3.SA'] #ALL of Them
    
    tickers_list.sort()
    
    #tickers = ['VALE3.SA', 'PETR3.SA', 'ALPA4.SA', 'B3SA3.SA', 'AZUL4.SA', 'BBDC3.SA', 'BBDC4.SA', 'CCRO3.SA', 'EGIE3.SA', 'GOLL4.SA', 'LREN3.SA', 'MRVE3.SA', 'RAIZ4.SA', 'RDOR3.SA', 'SOMA3.SA', 'VIIA3.SA', 'VIVT3.SA'] #cheapest ones
    
    #tickers = ['VALE3.SA', 'PETR3.SA']
    
    #tickers = ['ITSA4.SA', 'CMIG4.SA', 'ELET3.SA', 'VIIA3.SA'] #purchased


    st.sidebar.write("""# Stock Analysis""")
    stock = st.sidebar.selectbox('Select a stock', tickers_list)
    tickers = [stock]

    x = 1 # how many years from now
    x = st.sidebar.slider('Period of time (years)', 0.2, 7.0, 5.0)
    end_date = datetime.now() - timedelta(days=1)
    start_date = end_date - timedelta(days=int(365*x))
    
    st.sidebar.write("Start date: " + str(start_date.date()))
    
    all_data = get(tickers, start_date, end_date)
    
    #for stock in tickers:   
    #     globals()[stock[:-3]] = pdr.get_data_yahoo(stock, start_date, end_date)    
      
    # for stock in tickers:
    #     trace = go.Candlestick(x = all_data.loc[stock].index, open = all_data.loc[stock].Open, high = all_data.loc[stock].High, low = all_data.loc[stock].Low, close = all_data.loc[stock].Close)
    #     data = [trace]
    #     simple_plot(data, str(stock))
      
    trace = go.Candlestick(x = all_data.loc[stock].index, open = all_data.loc[stock].Open, high = all_data.loc[stock].High, low = all_data.loc[stock].Low, close = all_data.loc[stock].Close, name = 'Price')
    
    volume_n = all_data.loc[stock].Volume
    highest_p = all_data.loc[stock].High
    
    #volume_f = 0.4 * max(highest_p) * volume_n / max(volume_n)
    volume_f = volume_n
    
    # trace_vol = go.Scatter(x = all_data.loc[stock].index, y = volume_f, name = 'Volume', line = dict(color='black'), opacity=1)
    
    trace_vol = go.Bar(x = all_data.loc[stock].index, y = volume_f, name = 'Volume', marker_color = 'black')
    
    window1 = 12
    window2 = 26
    window1 = st.sidebar.slider('Moving Average #1', 7, 15, 12)
    window2 = st.sidebar.slider('Moving Average #2', 15, 30, 26)
    ckMA1 = st.sidebar.checkbox('Moving Avg')
    ckEMA1 = st.sidebar.checkbox('Exp Moving Avg')
    ckMACD = st.sidebar.checkbox('MACD')
    ckHiLo = st.sidebar.checkbox('HiLo')
    ckBollinger = st.sidebar.checkbox('Bollinger')
    ckIfr = st.sidebar.checkbox('IFR')
    ckObv = st.sidebar.checkbox('OBV')
    ckDividends = st.sidebar.checkbox('Dividends')
    
    k1 = ( 2 / (window1 + 1) )
    k2 = ( 2 / (window2 + 1) )
    
    MA1 = all_data.loc[stock].Close.rolling(window = window1).mean().dropna()
    MA2 = all_data.loc[stock].Close.rolling(window = window2).mean().dropna()
    
    trace_avg1 = go.Scatter(x = MA1.index, y = MA1, name = 'MA'+ str(window1), 
                           line = dict(color='#d06539'), opacity=1)
    
    trace_avg2 = go.Scatter(x = MA2.index, y = MA2, name = 'MA'+ str(window2), 
                           line = dict(color='#0032ac'), opacity=1)
    
    ema_data1 = pd.DataFrame(index = MA1.index)
    ema_data1['Price'] = all_data.loc[stock].dropna().Close
    ema_data1['MA'] = MA1
    ema_data1['EMA'] = np.NaN
    ema_data1.EMA[0] = ema_data1.MA[1]
    
    for i in range(1, len(ema_data1)):
        ema_data1.EMA[i] = (ema_data1.Price[i] * k1) + ((1 - k1) * ema_data1.EMA[i-1])
        
    ema_data2 = pd.DataFrame(index = MA2.index)
    ema_data2['Price'] = all_data.loc[stock].dropna().Close
    ema_data2['MA'] = MA2
    ema_data2['EMA'] = np.NaN
    ema_data2.EMA[0] = ema_data2.MA[1]
    
    auxm = all_data.loc[stock].dropna()
    mm1 = get_ema(window1, auxm.Close)
    mm2 = get_ema(window2, auxm.Close)
    mm_macd = mm1.EMA - mm2.EMA
    mm_signal = get_ema(9, mm_macd.dropna()).EMA
    hist_macd = mm_macd - mm_signal
    
    for i in range(1, len(ema_data2)):
        ema_data2.EMA[i] = (ema_data2.Price[i] * k2) + ((1 - k2) * ema_data2.EMA[i-1])
    
    trace_ema1 = go.Scatter(x = ema_data1.index, y = ema_data1.EMA, name = 'Exp MA'+ str(window1), line = dict(color='#d06539'), opacity=0.5)
    
    trace_ema2 = go.Scatter(x = ema_data2.index, y = ema_data2.EMA, name = 'Exp MA'+ str(window2), line = dict(color='#0032ac'), opacity=0.5)
    
    trace_macd = go.Scatter(x = mm_macd.index, y = mm_macd, name = 'MACD', line = dict(color='#17BECF'), opacity=1)
    
    trace_signal = go.Scatter(x = mm_signal.index, y = mm_signal, name = 'Signal', line = dict(color='#B22222'), opacity=1)
    
    trace_hist_macd = go.Scatter(x = hist_macd.index, y = hist_macd, name = 'Signal', fill = 'tozeroy')
    
    HighS = all_data.loc[stock].High.rolling(window = 8).mean().dropna()
    LowS = all_data.loc[stock].Low.rolling(window = 8).mean().dropna()
    
    trace_high = go.Scatter(x = HighS.index, y = HighS, name = 'High Avg', opacity = 1, line = dict(color='#cfc74d'))
    
    trace_low = go.Scatter(x = LowS.index, y = LowS, name = 'Low Avg', opacity = 1, line = dict(color='#cfc74d'))
    
    boll = all_data.loc[stock].Close.rolling(window = 20).mean().dropna()
    bollstdv = all_data.loc[stock].Close.rolling(window = 20).std().dropna()
    
    bollh = boll + bollstdv.apply(lambda x: (x * 2))
    bolll = boll - bollstdv.apply(lambda x: (x * 2))
    
    trace_bollh = go.Scatter(x = bollh.index, y = bollh, name = 'Boll. High', opacity = 1, line = dict(color='#17BECF'))
    
    trace_bolll = go.Scatter(x = bolll.index, y = bolll, name = 'Boll. Low', opacity = 1, line = dict(color='#17BECF'))
    
    trace_bollm = go.Scatter(x = boll.index, y = boll, name = 'Avg', opacity = 1, line = dict(color='#0d0303'))
    
    stock_ifr = all_data.loc[stock].Close
    ifr = pd.DataFrame(index = stock_ifr.index)
    ifr_changes = stock_ifr.diff()
    ifr['gain'] = ifr_changes.clip(lower=0)
    ifr['loss'] = ifr_changes.clip(upper=0).abs()
    
    ifr['gainAvg'] = np.NaN
    ifr['lossAvg'] = np.NaN
    
    windowi = 14
    ifr.gainAvg[:windowi] = ifr.gain[:windowi].mean()
    ifr.lossAvg[:windowi] = ifr.loss[:windowi].mean()
    ifr.gainAvg[windowi] = ifr.iloc[0:windowi].gain.mean()
    ifr.lossAvg[windowi] = ifr.iloc[0:windowi].loss.mean()
    
    for i in range(windowi+1,len(ifr)):
        # ifr.gainAvg[i] = (ifr.gainAvg[i-1]*(windowi - 1) + ifr.gain[i])/windowi
        # ifr.lossAvg[i] = (ifr.lossAvg[i-1]*(windowi - 1) - ifr.loss[i])/windowi
        ifr.gainAvg[i] = ifr.gain[i-windowi:i].sum() / windowi
        ifr.lossAvg[i] = ifr.loss[i-windowi:i].sum() / windowi
        
    ifr['value'] = 100 - (100/(1 + (ifr.gainAvg / ifr.lossAvg)))
    ifr['h70'] = 70
    ifr['h30'] = 30
    
    trace_ifr = go.Scatter(x = ifr.index, y = ifr.value, opacity = 1, showlegend = True)
    trace_h70 = go.Scatter(x = ifr.index, y = ifr.h70, opacity = 0.7, line=dict(color='rgb(255, 0, 0)', dash='dash'), showlegend = False)
    trace_h30 = go.Scatter(x = ifr.index, y = ifr.h30, opacity = 0.7, line=dict(color='rgb(255, 0, 0)', dash='dash'), showlegend = False)
    
    data = [trace]
    #data = [trace, trace_avg1, trace_avg2, trace_ema1, trace_ema2]
    
    if ckMA1 == True:
        data.append(trace_avg1)
        data.append(trace_avg2)
        
    if ckEMA1 == True:
        data.append(trace_ema1)
        data.append(trace_ema2)
    
    if ckHiLo == True:
        data.append(trace_high)
        data.append(trace_low)
        
    if ckBollinger == True:
        data.append(trace_bollh)
        data.append(trace_bolll)
        #data.append(trace_bollm)
    
    fig = simple_plot(data, str(stock))
    fig_voll = simple_plot([trace_vol], '')
    fig_voll.update_layout(width = 1150, height = 150)
        
    st.plotly_chart(fig, use_container_width = False)
    st.plotly_chart(fig_voll, use_container_width = False)
    
    if ckMACD == True:
        data1 = [trace_macd, trace_signal, trace_hist_macd]
        fig1 = simple_plot(data1, 'MACD')
        fig1.update_layout(width = 1150, height = 280)
        st.plotly_chart(fig1, use_container_width = False)
    
    if ckIfr == True:
        data2 = [trace_ifr, trace_h70, trace_h30]
        fig2 = simple_plot(data2, 'Relative Force Index')
        fig2.update_layout(width = 1150, height = 280)
        st.plotly_chart(fig2, use_container_width = False)
    
    
    stock_obv = all_data.loc[stock]
    obv = pd.DataFrame(index = stock_obv.index)
    obv_changes = stock_obv.Close - stock_obv.Open
    obv['open'] = stock_obv.Open
    obv['close'] = stock_obv.Close
    obv['volume'] = np.where(obv_changes > 0, stock_obv.Volume, stock_obv.Volume * (-1))
    obv['volume_sum'] = obv.volume.cumsum()
    
    trace_obv = go.Scatter(x = obv.index, y = obv.volume_sum, opacity = 1, name='OBV', showlegend = True)
    
    if ckObv == True:
        data_obv = [trace_obv]
        fig_obv = simple_plot(data_obv, 'OBV')
        fig_obv.update_layout(width = 1150, height = 280)
        st.plotly_chart(fig_obv, use_container_width = False)
        
    ### Dividends
    
    sinfo = yf.Ticker(str(stock))
    data_d = sinfo.dividends.resample('Y').sum()
    data_d = data_d.reset_index()
    data_d['Year'] = data_d['Date'].dt.year
    data_d = data_d[data_d['Year'] >= start_date.year]
    data_d = data_d.reset_index()
    trace_dividends = go.Bar(x = data_d['Year'], y = data_d['Dividends'], marker_color = 'blue', name = 'Dividends')
    
    if ckDividends == True:
        datad = [trace_dividends]
        fig_dividends = simple_plot(datad, '')
        fig_dividends.update_layout(width = 1150, height = 280)
        st.plotly_chart(fig_dividends, use_container_width = False)
    
    ### algorithm for opportunities
    
    start_date_min = end_date - timedelta(days=int(365*4)) # ultimos 4 anos
    #end_date_min = datetime.now() - timedelta(days=30)
    threshold = 1.12
    
    if st.button('Check for opportunities'):
        all_data_full = get(tickers_list, start_date, end_date)
    
        datac = all_data_full.reset_index()
        
        datacc = datac[datac['Date']<datetime.today()-timedelta(days=30)]
        
        datac = datac.set_index(['Date','Ticker']).sort_index()
        datacc = datacc.set_index(['Date','Ticker']).sort_index()
        
        close = datac['Close']
        close = close.reset_index().pivot(index = 'Date', columns = 'Ticker', values = 'Close')
        
        close_min = datacc['Close']
        close_min = close_min.reset_index().pivot(index = 'Date', columns = 'Ticker', values = 'Close')
            
        for tick in tickers_list:
            aux = pd.DataFrame([])
            aux[tick] = close_min[tick]
            min_value = aux[tick].min()
            if close[tick][-1] <= (min_value * threshold):
                data_min = pd.DataFrame([])
                data_min = aux.loc[aux[tick] == min_value]
                st.write("""### Check """ + str(tick) + ". Current price R\$ " + str(round(close[tick][-1],2)) + " close to " + str(data_min.index[0].date()) + " price: R\$ " + str(round(min_value,2)) + ".")







