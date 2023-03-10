import streamlit as st
import pandas_datareader.data as pdr
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import yfinance as yf
yf.pdr_override()

import requests
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout

#import plotly.offline as py
#import plotly.io as io
#io.renderers.default='browser'
import plotly.graph_objs as go

from pmdarima.arima import auto_arima
#from plotly.subplots import make_subplots

#py.init_notebook_mode(connected = True)
#fig = make_subplots(rows = 2)

st.set_page_config(page_title='Stock Analysis App', page_icon='🖖', layout="wide", initial_sidebar_state="auto", menu_items=None)

hide_menu_style = """
        <style>
        #MainMenu {visibility: hidden;}
        </style>
        """
st.markdown(hide_menu_style, unsafe_allow_html=True)

Menu = ['Home', 'Login']
user_list = ['admin', 'fabi', 'eude', 'kempt']
pass_list = ['123', 'fabi', 'eude', 'kempt']

# login = True
# username = 'admin'

st.sidebar.subheader("""Stock Analysis App""")
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
    fig.update_layout(title = title, font_family="Courier New", font_color="blue", xaxis_rangeslider_visible = False, width = 1000, height = 420, xaxis_showgrid = True, xaxis_gridwidth = 1, xaxis_gridcolor = '#E8E8E8', xaxis_linecolor = 'black', xaxis_tickfont = dict(size=14), yaxis_showgrid = True, yaxis_gridwidth = 1, yaxis_tickfont = dict(size=14), yaxis_gridcolor = '#E8E8E8', yaxis_linecolor = 'black', plot_bgcolor='rgba(0,0,0,0)', margin=dict(l=0, r=10, b=50, t=40), showlegend = True)
    
    #xaxis={'type': 'category'}
    
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

# def create_df(df, steps = 1):
#     dataX, dataY = [], []
#     for i in range(len(df)-steps-1):
#         a = df[i:(i+steps), 0]
#         dataX.append(a)
#         dataY.append(df[i+steps, 0])
#     return np.array(dataX), np.array(dataY)

def create_df(data, steps, features):
    X = []
    Y = []
    for i in range(steps, len(data)):
        X.append(data[i-steps:i, features])
        Y.append(data[i, 0])
    X, Y = np.array(X), np.array(Y)
    return X, Y

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
    # tickers_list = ['VALE3.SA', 'PETR3.SA', 'PETR4.SA', '^BVSP', 'CMIG4.SA', 'ITSA4.SA', 'VIIA3.SA','CPLE6.SA', 'MGLU3.SA', 'BBDC4.SA', 'BBDC3.SA', 'B3SA3.SA', 'WEGE3.SA', 'ELET3.SA', 'ITUB4.SA', 'BBAS3.SA', 'JBSS3.SA', 'CIEL3.SA', 'RADL3.SA', 'BEEF3.SA', 'ABEV3.SA', 'LREN3.SA', 'TIMS3.SA', 'HYPE3.SA', 'GGBR4.SA', 'RRRP3.SA', 'UGPA3.SA','PETZ3.SA', 'RAIZ4.SA', 'NTCO3.SA', 'EGIE3.SA','BRAP4.SA','TAEE11.SA','PRIO3.SA','CCRO3.SA','RAIL3.SA','CYRE3.SA','ENEV3.SA','USIM5.SA','ALPA4.SA','BRKM5.SA','ARZZ3.SA', 'AZUL4.SA','CSNA3.SA', 'RDOR3.SA','MRVE3.SA','SOMA3.SA', 'GOAU4.SA', 'EMBR3.SA', 'VIVT3.SA','GOLL4.SA','TOTS3.SA', 'CVCB3.SA', 'HAPV3.SA', 'BRFS3.SA','BPAC11.SA', 'RENT3.SA', 'ENBR3.SA', 'EQTL3.SA', 'BBSE3.SA', 'VBBR3.SA', 'ASAI3.SA', 'MULT3.SA', 'KLBN11.SA', 'ALUP11.SA', 'SBSP3.SA'] #ALL of Them
    # tickers_list = [*set(tickers_list)]
    # tickers_list.sort()
    
    #tickers = ['VALE3.SA', 'PETR3.SA', 'ALPA4.SA', 'B3SA3.SA', 'AZUL4.SA', 'BBDC3.SA', 'BBDC4.SA', 'CCRO3.SA', 'EGIE3.SA', 'GOLL4.SA', 'LREN3.SA', 'MRVE3.SA', 'RAIZ4.SA', 'RDOR3.SA', 'SOMA3.SA', 'VIIA3.SA', 'VIVT3.SA'] #cheapest ones
    
    #tickers = ['VALE3.SA', 'PETR3.SA']
    
    #tickers = ['ITSA4.SA', 'CMIG4.SA', 'ELET3.SA', 'VIIA3.SA'] #purchased
    
    url = 'https://github.com/fsarraci/finance_app/blob/main/stocks_list.xlsx?raw=true'
    file = requests.get(url)
    df_stocks_list = pd.read_excel(file.content)
    
    tickers_list = df_stocks_list['ticker'].to_list()
    tickers_list = [*set(tickers_list)]
    tickers_list.sort()

    st.sidebar.write("""Stock Analysis""" + ' - ' + str(len(tickers_list)) + ' tickers')
    stock = st.sidebar.selectbox('Select a stock', tickers_list)
    tickers = [stock]
    aux_stock = df_stocks_list.loc[df_stocks_list['ticker'] == tickers[0]]
    aux_stock = aux_stock.reset_index()
    aux_stock.drop(aux_stock.columns[[0, 1]], axis=1, inplace=True)
    
    x = 1 # how many years from now
    x = st.sidebar.slider('Period of time (years)', 0.15, 5.0, 1.0)
    end_date = datetime.now() - timedelta(days=1)
    start_date = end_date - timedelta(days=int(365*x))
    st.write(str(aux_stock['full_name'][0]))
    st.sidebar.write("Start date: " + str(start_date.date() + timedelta(days=1)))
    
    all_data = get(tickers, start_date, end_date)

    # aux_dat = all_data.reset_index()
    # aux_dat['Date'] = aux_dat['Date'].dt.strftime('%d-%m-%Y')
    # all_data = aux_dat.set_index(['Ticker', 'Date'])   
    
    #for stock in tickers:   
    #     globals()[stock[:-3]] = pdr.get_data_yahoo(stock, start_date, end_date)    
      
    # for stock in tickers:
    #     trace = go.Candlestick(x = all_data.loc[stock].index, open = all_data.loc[stock].Open, high = all_data.loc[stock].High, low = all_data.loc[stock].Low, close = all_data.loc[stock].Close)
    #     data = [trace]
    #     simple_plot(data, str(stock))
      
    trace = go.Candlestick(x = all_data.loc[stock].index, open = all_data.loc[stock].Open, high = all_data.loc[stock].High, low = all_data.loc[stock].Low, close = all_data.loc[stock].Close, name = 'Price', line=dict(width=1.5))
    
    volume_n = all_data.loc[stock].Volume
    highest_p = all_data.loc[stock].High
    
    #volume_f = 0.4 * max(highest_p) * volume_n / max(volume_n)
    volume_f = volume_n
    
    # trace_vol = go.Scatter(x = all_data.loc[stock].index, y = volume_f, name = 'Volume', line = dict(color='black'), opacity=1)
    
    trace_vol = go.Bar(x = all_data.loc[stock].index, y = volume_f, name = 'Volume', marker_color = 'black')
    
    chkFib = st.sidebar.checkbox('Fibonacci')
    if chkFib == True:
        xfib = x + 0.05 - st.slider('Fibonacci Period', 0.1, x, min(0.45,x))       
                
    window1 = 12
    window2 = 26
    ckMA1 = st.sidebar.checkbox('Moving Avg')
    ckEMA1 = st.sidebar.checkbox('Exp Moving Avg')
    window1 = st.sidebar.slider('Moving Average #1', 7, 15, 12)
    window2 = st.sidebar.slider('Moving Average #2', 15, 30, 26)
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
    if chkFib == True:
        end_date_fib = datetime.now() - timedelta(days=1)
        start_date_fib = end_date - timedelta(days=int(365*xfib))
        all_data_fib = get(tickers, start_date_fib, end_date_fib)
        fib_min = min(all_data_fib['Low'])
        fib_max = max(all_data_fib['High'])
        diff_fib = fib_max - fib_min
        all_data_fib = all_data_fib.reset_index()
        l0 = fib_max
        l1 = fib_max - diff_fib * 0.236
        l2 = fib_max - diff_fib * 0.382
        l3 = fib_max - diff_fib * 0.5
        l4 = fib_max - diff_fib * 0.618
        l5 = fib_min
        all_data_fib['l0'] = l0
        all_data_fib['l1'] = l1
        all_data_fib['l2'] = l2
        all_data_fib['l3'] = l3
        all_data_fib['l4'] = l4
        all_data_fib['l5'] = l5
        x01 = [start_date_fib, start_date_fib, end_date, end_date, start_date_fib]
        y01 = [l0, l1, l1, l0]
        y12 = [l1, l2, l2, l1]
        y23 = [l2, l3, l3, l2]
        y34 = [l3, l4, l4, l3]
        y45 = [l4, l5, l5, l4]

        esp = 1
        trace_l01 = go.Scatter(x = x01, y = y01, opacity = 0.15, line = dict(color='red', dash='dash', width=esp), fill = 'toself', showlegend = False)
        trace_l12 = go.Scatter(x = x01, y = y12, opacity = 0.15, line = dict(color='yellow', dash='dash', width=esp), fill = 'toself', showlegend = False)
        trace_l23 = go.Scatter(x = x01, y = y23, opacity = 0.15, line = dict(color='green', dash='dash', width=esp), fill = 'toself', showlegend = False)
        trace_l34 = go.Scatter(x = x01, y = y34, opacity = 0.15, line = dict(color='cyan', dash='dash', width=esp), fill = 'toself', showlegend = False)
        trace_l45 = go.Scatter(x = x01, y = y45, opacity = 0.15, line = dict(color='blue', dash='dash', width=esp), fill = 'toself', showlegend = False)
        
        trace_l0 = go.Scatter(x = all_data_fib['Date'], y = all_data_fib['l0'], opacity = 0.7, line = dict(color='rgb(0, 0, 255)', dash='dash', width=esp), showlegend = False)
        trace_l1 = go.Scatter(x = all_data_fib['Date'], y = all_data_fib['l1'], opacity = 0.7, line = dict(color='rgb(0, 0, 255)', dash='dash', width=esp), showlegend = False)
        trace_l2 = go.Scatter(x = all_data_fib['Date'], y = all_data_fib['l2'], opacity = 0.7, line = dict(color='rgb(0, 0, 255)', dash='dash', width=esp), showlegend = False)
        trace_l3 = go.Scatter(x = all_data_fib['Date'], y = all_data_fib['l3'], opacity = 0.7, line = dict(color='rgb(0, 0, 255)', dash='dash', width=esp), showlegend = False)
        trace_l4 = go.Scatter(x = all_data_fib['Date'], y = all_data_fib['l4'], opacity = 0.7, line = dict(color='rgb(0, 0, 255)', dash='dash', width=esp), showlegend = False)
        trace_l5 = go.Scatter(x = all_data_fib['Date'], y = all_data_fib['l5'], opacity = 0.7, line = dict(color='rgb(0, 0, 255)', dash='dash', width=esp), showlegend = False)
        data.append(trace_l0)
        data.append(trace_l1)
        data.append(trace_l2)
        data.append(trace_l3)
        data.append(trace_l4)
        data.append(trace_l5)
        data.append(trace_l01)
        data.append(trace_l12)
        data.append(trace_l23)
        data.append(trace_l34)
        data.append(trace_l45)
        
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
    fig_voll.update_layout(width = 1000, height = 150)
        
    st.plotly_chart(fig, use_container_width = False)
    st.plotly_chart(fig_voll, use_container_width = False)
    
    if ckMACD == True:
        data1 = [trace_macd, trace_signal, trace_hist_macd]
        fig1 = simple_plot(data1, 'MACD')
        fig1.update_layout(width = 1000, height = 280)
        st.plotly_chart(fig1, use_container_width = False)
    
    if ckIfr == True:
        data2 = [trace_ifr, trace_h70, trace_h30]
        fig2 = simple_plot(data2, 'Relative Force Index')
        fig2.update_layout(width = 1000, height = 280)
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
        fig_obv.update_layout(width = 1000, height = 280)
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
        fig_dividends.update_layout(width = 1000, height = 280)
        st.plotly_chart(fig_dividends, use_container_width = False)
    
########## algorithm for opportunities
    
    if username == 'admin':  
        start_date_min = end_date - timedelta(days=int(365*4)) # ultimos 4 anos
        #end_date_min = datetime.now() - timedelta(days=30)
        threshold = 1.12
            
        if st.button('Check for opportunities'):
            
            tickers_list = [*set(tickers_list)]
            t1list = tickers_list
            #t1list.pop()
            all_data_full = get(t1list, start_date, end_date)
                    
            datac = all_data_full.reset_index()
            
            datacc = datac[datac['Date']<datetime.today()-timedelta(days=30)]
            
            data_p_plot = datac[datac['Date']==datac['Date'][len(datac)-1]]
            data_p_plot = data_p_plot.sort_values(by = ['Close'])
            data_p_plot = data_p_plot.reset_index()
            data_p_plot = data_p_plot.drop('index', axis=1)
            
            datac = datac.set_index(['Date','Ticker']).sort_index()
            datacc = datacc.set_index(['Date','Ticker']).sort_index()
            
            close = datac['Close']
            close = close.reset_index().pivot(index = 'Date', columns = 'Ticker', values = 'Close')
            
            close_min = datacc['Close']
            close_min = close_min.reset_index().pivot(index = 'Date', columns = 'Ticker', values = 'Close')
            
            df_table = pd.DataFrame(columns=['Ticker','Current Price','Min Value History', 'Date Min Value', 'Delta Price'])
            i = 0
            for tick in tickers_list:
                aux = pd.DataFrame([])
                aux[tick] = close_min[tick]
                min_value = aux[tick].min()
                if close[tick][-1] <= (min_value * threshold):
                    data_min = pd.DataFrame([])
                    data_min = aux.loc[aux[tick] == min_value]
                    df_table.at[i, 'Ticker'] = str(tick)
                    df_table.at[i, 'Current Price'] = round(close[tick][-1],2)
                    df_table.at[i, 'Min Value History'] = round(min_value,2)
                    df_table.at[i, 'Date Min Value'] = data_min.index[0].date()
                    df_table.at[i, 'Delta Price'] = round(close[tick][-1],2) - round(min_value,2)
                    i += 1
            df_table = df_table.sort_values('Delta Price')
            df_table = df_table.reset_index()
            df_table = df_table.drop('index', axis=1)
            
            st.write('Opportunities - Current Price close to historical price')
            st.dataframe(df_table.style.format({'Current Price': '{:.2f}', 'Min Value History': '{:.2f}', 'Delta Price': '{:.2f}'}))
            
            df_table.to_excel("opportunities.xlsx")
            data_p_plot.to_excel("prices.xlsx")
            
            y = data_p_plot['Ticker']
            x = data_p_plot['Close']
            
            trace_prices = go.Bar(x = x, y = y, marker=dict(color='blue'), name = '', orientation='h')
            
            fig_prices = simple_plot([trace_prices], 'Stock Prices')
            fig_prices.update_layout(width = 1000, height = 4500)
                
            st.plotly_chart(fig_prices, use_container_width = False)
        
####




###########################
    if username == 'admin':    
        if st.button("COMBINED - DL + ARIMA"):
            
            df_acao_fec = all_data.reset_index()
            df_acao_fec = df_acao_fec[['Date', 'Close']]
            df_acao_fec = df_acao_fec[:-1]
            dates = df_acao_fec['Date']
            df_acao_fec = df_acao_fec.set_index(['Date'])
                   
            model_arima = auto_arima(df_acao_fec,start_p=1, start_q=1,
                               max_p=3, max_q=3, m=12,
                               start_P=0, seasonal=True,
                               d=1, D=1, trace=True,
                               error_action='ignore',  
                               suppress_warnings=True)
           
            n_future = 9
            forecast = model_arima.predict(n_periods = n_future)
            
            list_output_prev_arima = forecast.tolist()
            
            predict_dates = pd.date_range(list(dates)[-1] + pd.DateOffset((1)), periods=n_future, freq='b').tolist()
            #predict_dates
            
            forecast_dates =[]
            for i in predict_dates:
                forecast_dates.append(i.date())    
            
            df_forecast = pd.DataFrame({'data':np.array(forecast_dates), 'close':list_output_prev_arima})
            df_forecast['data'] = pd.to_datetime(df_forecast['data'])
            df_forecast = df_forecast.set_index(['data'])
            #df_forecast
            
            trace_arima = go.Scatter(x = df_forecast.index, y = df_forecast.close, opacity = 1, line=dict(color='rgb(155, 100, 255)'), mode = 'lines', name='ARIMA Forecast', showlegend = True)
            
            
            qtd_linhas = len(df_acao_fec)
    
            qtd_linhas_treino = round(0.9 * qtd_linhas)
            qtd_linhas_teste = qtd_linhas - qtd_linhas_treino
    
            #normalizando os dados
            scaler = StandardScaler()        
            df_scaled = scaler.fit_transform(df_acao_fec)
            
            #separa treino e teste
            train = df_scaled[:qtd_linhas_treino]
            test = df_scaled[qtd_linhas_treino: qtd_linhas_treino + qtd_linhas_teste]
            
            #gerando dados de treino e teste
            steps = 15
            features = [0] # indices of Close, Volume, and Delta columns
            X_train, Y_train = create_df(train, steps, features)
            X_test, Y_test = create_df(test, steps, features)
            
            #gerando os dados que o modelo espera
            X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], len(features))
            X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], len(features))
    
            n_lstm = 100
               
            #montando a rede
            model = Sequential()
            model.add(LSTM(n_lstm, return_sequences=True, input_shape=(steps,len(features))))
            model.add(LSTM(n_lstm, return_sequences=True))
            model.add(Dropout(0.2))
            model.add(LSTM(n_lstm, return_sequences=True))
            model.add(Dropout(0.2))
            model.add(LSTM(n_lstm, return_sequences=True))
            model.add(Dropout(0.2))
            model.add(LSTM(n_lstm, return_sequences=True))
            model.add(Dropout(0.2))
            model.add(LSTM(n_lstm, return_sequences=True))
            model.add(Dropout(0.2))
            model.add(LSTM(n_lstm, return_sequences=True))
            model.add(Dropout(0.2))
            model.add(LSTM(n_lstm, return_sequences=True))
            model.add(Dropout(0.2))
            model.add(LSTM(n_lstm, return_sequences=True))
            model.add(Dropout(0.2))
            model.add(LSTM(n_lstm, return_sequences=True))
            model.add(Dropout(0.2))
            model.add(LSTM(n_lstm))
            model.add(Dropout(0.2))
            model.add(Dense(1))
            
            model.compile(optimizer = 'adam', loss='mse')
            
            model.summary()
            
            #treinamento do modelo
            validation = model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs = 30, batch_size=10, verbose = 2)
            
            df_training = pd.DataFrame()
            df_training['loss'] = validation.history['loss']
            df_training['val_loss'] = validation.history['val_loss']
            
            preva = model.predict(X_train)
            preva = scaler.inverse_transform(preva)
            
            prevb = model.predict(X_test)
            prevb = scaler.inverse_transform(prevb)
            
            dfa = pd.DataFrame()
            dfa['close'] = pd.DataFrame(np.array(preva).tolist())
            dfa.index = df_acao_fec[steps:len(preva)+steps].index
                    
            dfb = pd.DataFrame()
            dfb['close'] = pd.DataFrame(np.array(prevb).tolist())
            dfb.index = df_acao_fec[len(preva)+2*steps:len(preva)+2*steps+len(prevb)].index
               
            trace_preva = go.Scatter(x = dfa.index, y = dfa['close'], opacity = 1, line=dict(color='rgb(255, 0, 255)'), mode = 'lines', name='LSTM-Model_Train', showlegend = True)
            trace_prevb = go.Scatter(x = dfb.index, y = dfb['close'], opacity = 1, line=dict(color='rgb(255, 0, 255)'), mode = 'lines', name='LSTM-Model_Test', showlegend = True)
            
            trace_training = go.Scatter(x = df_training.index, y = df_training['loss'], opacity = 1, line=dict(color='rgb(255, 0, 0)'), name='training loss', showlegend = True)
            trace_training1 = go.Scatter(x = df_training.index, y = df_training['val_loss'], opacity = 1, line=dict(color='rgb(0, 0, 255)'), name='validation loss', showlegend = True)
            
            datatrain = [trace_training, trace_training1]             
            figtrain = simple_plot(datatrain, str(stock))
            st.plotly_chart(figtrain, use_container_width = False)
                   
            #previsao para os proximos 10 dias
            length_test = len(test)
            #length_test
            
            #pegar os ultimos dias que sao o tamanho do step
            days_input_steps = length_test - steps
            
            #transforma em array
            input_steps = test[days_input_steps:]
            input_steps = np.array(input_steps).reshape(1,-1)
            #input_steps
    
            #transformar em lista
            list_output_steps = list(input_steps)
            list_output_steps = list_output_steps[0].tolist()
            #list_output_steps
            
            pred_output =[]
            i = 0
            
            while(i<n_future):
                if(len(list_output_steps)>steps):
                    input_steps = np.array(list_output_steps[1:])
                    #print('{} dia. Valores de entrada -> {}'.format(i, input_steps))
                    input_steps = input_steps.reshape(1,-1)
                    input_steps = input_steps.reshape(1, steps, len(features))
                    
                    pred = model.predict(input_steps, verbose=0)
                    
                    #print('{} dia. Valor previsto -> {}'.format(i, pred))
                    list_output_steps.extend(pred[0].tolist())
                    list_output_steps = list_output_steps[1:]
                    pred_output.extend(pred.tolist())
                    i=i+1
                else:
                    input_steps = input_steps.reshape(1, steps, len(features))
                    pred = model.predict(input_steps, verbose=0)
                    #print(pred[0])
                    list_output_steps.extend(pred[0].tolist())
                    #print(len(list_output_steps))
                    pred_output.extend(pred.tolist())
                    i=i+1
            
            #transforma saida
            prev = scaler.inverse_transform(pred_output)
            prev = np.array(prev).reshape(1,-1)
            list_output_prev = list(prev)
            list_output_prev = prev[0].tolist()
            #list_output_prev
               
            df_forecast1 = pd.DataFrame(index = forecast_dates)
            df_forecast1['close'] = list_output_prev     
           
            trace_lstm = go.Scatter(x = df_forecast1.index, y = df_forecast1.close, opacity = 1, line=dict(color='rgb(255, 0, 255)'), mode = 'lines', name='LSTM Forecast', showlegend = True)
              
            df_forecast_combined = pd.DataFrame(index = forecast_dates)
            df_forecast_combined['close'] = (df_forecast['close'] + df_forecast1['close'])/2
            
            trace_combined = go.Scatter(x = df_forecast_combined.index, y = df_forecast_combined.close, opacity = 1, line=dict(color='rgb(0, 0, 255)'), name='Combined Prediction', mode = 'lines', showlegend = True)
            
            data = [trace, trace_arima, trace_lstm, trace_combined, trace_preva, trace_prevb]
            #data = [trace, trace_arima, trace_lstm, trace_preva, trace_prevb]                 
            fig = simple_plot(data, str(stock))
            st.plotly_chart(fig, use_container_width = False)


