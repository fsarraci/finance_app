import streamlit as st
import pandas_datareader.data as pdr
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import mplfinance as fplt
import yfinance as yf
import plotly
import plotly.graph_objs as go

yf.pdr_override()

st.set_page_config(page_title='Stock Analysis App', page_icon='🖖', layout="wide", initial_sidebar_state="auto", menu_items=None)

hide_menu_style = """
        <style>
        #MainMenu {visibility: hidden;}
        </style>
        """
st.markdown(hide_menu_style, unsafe_allow_html=True)

Menu = ['Home', 'Login']
user_list = ['admin', 'fabi', 'eude', 'user']
pass_list = ['123', 'fabi', 'eude', 'user']

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
    fig = plotly.graph_objs.Figure(data = data)
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

def ATR(DF, n):
    df = DF.copy() # making copy of the original dataframe
    df['H-L'] = abs(df['High'] - df['Low']) 
    df['H-PC'] = abs(df['High'] - df['Adj Close'].shift(1))# high -previous close
    df['L-PC'] = abs(df['Low'] - df['Adj Close'].shift(1)) #low - previous close
    df['TR'] = df[['H-L','H-PC','L-PC']].max(axis =1, skipna = False) # True range
    df['ATR'] = df['TR'].rolling(n).mean() # average –true range
    df = df.drop(['H-L','H-PC','L-PC'], axis =1) # dropping the unneccesary columns
    df.dropna(inplace = True) # droping null items
    return df

if login == True:
    ### SETUP ###
    # tickers_list = ['VALE3.SA', 'PETR3.SA', 'PETR4.SA', '^BVSP', 'CMIG4.SA', 'ITSA4.SA', 'VIIA3.SA','CPLE6.SA', 'MGLU3.SA', 'BBDC4.SA', 'BBDC3.SA', 'B3SA3.SA', 'WEGE3.SA', 'ELET3.SA', 'ITUB4.SA', 'BBAS3.SA', 'JBSS3.SA', 'CIEL3.SA', 'RADL3.SA', 'BEEF3.SA', 'ABEV3.SA', 'LREN3.SA', 'TIMS3.SA', 'HYPE3.SA', 'GGBR4.SA', 'RRRP3.SA', 'UGPA3.SA','PETZ3.SA', 'RAIZ4.SA', 'NTCO3.SA', 'EGIE3.SA','BRAP4.SA','TAEE11.SA','PRIO3.SA','CCRO3.SA','RAIL3.SA','CYRE3.SA','ENEV3.SA','USIM5.SA','ALPA4.SA','BRKM5.SA','ARZZ3.SA', 'AZUL4.SA','CSNA3.SA', 'RDOR3.SA','MRVE3.SA','SOMA3.SA', 'GOAU4.SA', 'EMBR3.SA', 'VIVT3.SA','GOLL4.SA','TOTS3.SA', 'CVCB3.SA', 'HAPV3.SA', 'BRFS3.SA','BPAC11.SA', 'RENT3.SA', 'EQTL3.SA', 'BBSE3.SA', 'VBBR3.SA', 'ASAI3.SA', 'MULT3.SA', 'KLBN11.SA', 'ALUP11.SA', 'SBSP3.SA'] #ALL of Them
    # tickers_list = [*set(tickers_list)]
    # tickers_list.sort()
    
    #tickers = ['VALE3.SA', 'PETR3.SA', 'ALPA4.SA', 'B3SA3.SA', 'AZUL4.SA', 'BBDC3.SA', 'BBDC4.SA', 'CCRO3.SA', 'EGIE3.SA', 'GOLL4.SA', 'LREN3.SA', 'MRVE3.SA', 'RAIZ4.SA', 'RDOR3.SA', 'SOMA3.SA', 'VIIA3.SA', 'VIVT3.SA'] #cheapest ones
    
    #tickers = ['VALE3.SA', 'PETR3.SA']
    
    #tickers = ['ITSA4.SA', 'CMIG4.SA', 'ELET3.SA', 'VIIA3.SA'] #purchased
    
    #url = 'https://github.com/fsarraci/finance_app/blob/main/stocks_list.xlsx?raw=true'
    #file = requests.get(url)
    #file = 'stocks_list.xlsx'
    #df_stocks_list = pd.read_excel(file)
       
    #tickers_list = df_stocks_list['ticker'].to_list()
    
    tickers_list = ['HAPV3.SA', 'MGLU3.SA', 'AZUL4.SA', 'GOLL4.SA', 'PETR4.SA', 'CVCB3.SA', 'B3SA3.SA', 'BBDC4.SA', 'OIBR3.SA', 'ITUB4.SA', 'ABEV3.SA', 'VALE3.SA', 'BRFS3.SA', 'ITSA4.SA', 'COGN3.SA', 'CIEL3.SA', 'FNOR11.SA', 'AMAR3.SA', 'PETR3.SA', 'LREN3.SA', 'CEAB3.SA', 'POMO4.SA', 'RENT3.SA', 'USIM5.SA', 'MRFG3.SA', 'ELET3.SA', 'GGBR4.SA', 'CSAN3.SA', 'BBAS3.SA', 'BEEF3.SA', 'CPLE6.SA', 'BPAC11.SA', 'PRIO3.SA', 'CSNA3.SA', 'GOAU4.SA', 'JBSS3.SA', 'MRVE3.SA', 'CMIG4.SA', 'EQTL3.SA', 'BBSE3.SA', 'VIVR3.SA', 'UGPA3.SA', 'RAIL3.SA', 'BBDC3.SA', 'BBDC3.SA', 'TOTS3.SA', 'EMBR3.SA', 'RADL3.SA', 'GRND3.SA', 'CCRO3.SA', 'LIGT3.SA', 'ENEV3.SA', 'MOVI3.SA', 'ECOR3.SA', 'ALPA4.SA', 'WEGE3.SA', 'SUZB3.SA', 'MULT3.SA', 'YDUQ3.SA', 'QUAL3.SA', 'ALUP11.SA', 'TEND3.SA', 'KLBN11.SA', 'GFSA3.SA', 'BPAN4.SA', 'CRFB3.SA', 'VAMO3.SA', 'ANIM3.SA', 'SMTO3.SA', 'BRAP4.SA', 'HYPE3.SA', 'ALSO3.SA', 'RCSL3.SA', 'POMO3.SA', 'RAPT4.SA', 'ELET6.SA', 'TRPL4.SA', 'CPFE3.SA', 'KLBN4.SA', 'CYRE3.SA', 'ARZZ3.SA', 'SBSP3.SA', 'STBP3.SA', 'EGIE3.SA', 'ITUB3.SA', 'MDIA3.SA', 'VIVT3.SA', 'ODPV3.SA', 'FLRY3.SA', 'SAPR4.SA', 'EZTC3.SA', 'DIRR3.SA', 'BRKM5.SA', 'GUAR3.SA', 'BRSR6.SA', 'JHSF3.SA', 'IRBR3.SA', 'SANB11.SA', 'PSSA3.SA', 'TSLA34.SA', 'INEP3.SA', 'VIVA3.SA', 'AALR3.SA', 'CPLE3.SA', 'LUPA3.SA', 'TAEE11.SA', 'HBOR3.SA', 'CSMG3.SA', 'ENAT3.SA', 'ENGI11.SA', 'CAML3.SA', 'RANI3.SA', 'POSI3.SA', 'NEOE3.SA', 'MYPK3.SA', 'PTBL3.SA', 'MELI34.SA', 'ABCB4.SA', 'EVEN3.SA', 'MILS3.SA', 'MXRF11.SA', 'MEAL3.SA', 'COCE5.SA', 'KEPL3.SA', 'SLCE3.SA', 'ROMI3.SA', 'VGIR11.SA', 'BMGB4.SA', 'RCSL4.SA', 'SAPR11.SA', 'ETER3.SA', 'PDGR3.SA', 'TUPY3.SA', 'SQIA3.SA', 'TASA4.SA', 'TRIS3.SA', 'FRAS3.SA', 'TAEE4.SA', 'SHUL4.SA', 'LOGG3.SA', 'JSLG3.SA', 'OIBR4.SA', 'INEP4.SA', 'USIM3.SA', 'DASA3.SA', 'VULC3.SA', 'PNVL3.SA', 'FESA4.SA', 'SEER3.SA', 'KLBN3.SA', 'AGRO3.SA', 'AZEV4.SA', 'SANB4.SA', 'SAPR3.SA', 'AMZO34.SA', 'SANB3.SA', 'TECN3.SA', 'SHOW3.SA', 'MSFT34.SA', 'VLID3.SA', 'AAPL34.SA', 'LEVE3.SA', 'UNIP6.SA', 'TAEE3.SA', 'GOAU3.SA', 'SGPS3.SA', 'BRPR3.SA', 'CMIG3.SA', 'ITSA3.SA', 'TPIS3.SA', 'TGMA3.SA', 'PFRM3.SA', 'LPSB3.SA', 'HCTR11.SA', 'BTCR11.SA', 'NFLX34.SA', 'ALUP4.SA', 'KNCR11.SA', 'GOGL34.SA']
    
    tickers_list = [*set(tickers_list)]
    tickers_list.sort()

    st.sidebar.write("""Stock Analysis""" + ' - ' + str(len(tickers_list)) + ' tickers')
    stock = st.sidebar.selectbox('Select a stock', tickers_list)
    tickers = [stock]
    #aux_stock = df_stocks_list.loc[df_stocks_list['ticker'] == tickers[0]]
    #aux_stock = aux_stock.reset_index()
    #aux_stock.drop(aux_stock.columns[[0, 1]], axis=1, inplace=True)
    
    x = 1 # how many years from now
    x = st.sidebar.slider('Period of time (years)', 0.15, 5.0, 1.5)
    end_date = datetime.now() - timedelta(days=0)
    start_date = end_date - timedelta(days=int(365*x))
    #st.write(str(aux_stock['full_name'][0]))
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
      
    trace = plotly.graph_objs.Candlestick(x = all_data.loc[stock].index, open = all_data.loc[stock].Open, high = all_data.loc[stock].High, low = all_data.loc[stock].Low, close = all_data.loc[stock].Close, name = 'Price', line=dict(width=1.5))
    
    volume_n = all_data.loc[stock].Volume
    highest_p = all_data.loc[stock].High
    
    #volume_f = 0.4 * max(highest_p) * volume_n / max(volume_n)
    volume_f = volume_n
    
    # trace_vol = go.Scatter(x = all_data.loc[stock].index, y = volume_f, name = 'Volume', line = dict(color='black'), opacity=1)
    
    trace_vol = plotly.graph_objs.Bar(x = all_data.loc[stock].index, y = volume_f, name = 'Volume', marker_color = 'black')
    
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
    ckRenko = st.sidebar.checkbox('Renko') 
    ckHiLo = st.sidebar.checkbox('HiLo')
    ckBollinger = st.sidebar.checkbox('Bollinger')
    ckIfr = st.sidebar.checkbox('IFR')
    ckObv = st.sidebar.checkbox('OBV')
    ckDividends = st.sidebar.checkbox('Dividends')
    
    k1 = ( 2 / (window1 + 1) )
    k2 = ( 2 / (window2 + 1) )
    
    MA1 = all_data.loc[stock].Close.rolling(window = window1).mean().dropna()
    MA2 = all_data.loc[stock].Close.rolling(window = window2).mean().dropna()
    
    trace_avg1 = plotly.graph_objs.Scatter(x = MA1.index, y = MA1, name = 'MA'+ str(window1), 
                           line = dict(color='#d06539'), opacity=1)
    
    trace_avg2 = plotly.graph_objs.Scatter(x = MA2.index, y = MA2, name = 'MA'+ str(window2), 
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
    
    trace_ema1 = plotly.graph_objs.Scatter(x = ema_data1.index, y = ema_data1.EMA, name = 'Exp MA'+ str(window1), line = dict(color='#d06539'), opacity=0.5)
    
    trace_ema2 = plotly.graph_objs.Scatter(x = ema_data2.index, y = ema_data2.EMA, name = 'Exp MA'+ str(window2), line = dict(color='#0032ac'), opacity=0.5)
    
    trace_macd = plotly.graph_objs.Scatter(x = mm_macd.index, y = mm_macd, name = 'MACD', line = dict(color='#17BECF'), opacity=1)
    
    trace_signal = plotly.graph_objs.Scatter(x = mm_signal.index, y = mm_signal, name = 'Signal', line = dict(color='#B22222'), opacity=1)
    
    trace_hist_macd = plotly.graph_objs.Scatter(x = hist_macd.index, y = hist_macd, name = 'Signal', fill = 'tozeroy')
    
    HighS = all_data.loc[stock].High.rolling(window = 8).mean().dropna()
    LowS = all_data.loc[stock].Low.rolling(window = 8).mean().dropna()
    
    trace_high = plotly.graph_objs.Scatter(x = HighS.index, y = HighS, name = 'High Avg', opacity = 1, line = dict(color='#cfc74d'))
    
    trace_low = plotly.graph_objs.Scatter(x = LowS.index, y = LowS, name = 'Low Avg', opacity = 1, line = dict(color='#cfc74d'))
    
    boll = all_data.loc[stock].Close.rolling(window = 20).mean().dropna()
    bollstdv = all_data.loc[stock].Close.rolling(window = 20).std().dropna()
    
    bollh = boll + bollstdv.apply(lambda x: (x * 2))
    bolll = boll - bollstdv.apply(lambda x: (x * 2))
    
    trace_bollh = plotly.graph_objs.Scatter(x = bollh.index, y = bollh, name = 'Boll. High', opacity = 1, line = dict(color='#17BECF'))
    
    trace_bolll = plotly.graph_objs.Scatter(x = bolll.index, y = bolll, name = 'Boll. Low', opacity = 1, line = dict(color='#17BECF'))
    
    trace_bollm = plotly.graph_objs.Scatter(x = boll.index, y = boll, name = 'Avg', opacity = 1, line = dict(color='#0d0303'))
    
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
    
    trace_ifr = plotly.graph_objs.Scatter(x = ifr.index, y = ifr.value, opacity = 1, showlegend = True)
    trace_h70 = plotly.graph_objs.Scatter(x = ifr.index, y = ifr.h70, opacity = 0.7, line=dict(color='rgb(255, 0, 0)', dash='dash'), showlegend = False)
    trace_h30 = plotly.graph_objs.Scatter(x = ifr.index, y = ifr.h30, opacity = 0.7, line=dict(color='rgb(255, 0, 0)', dash='dash'), showlegend = False)
    
    data = [trace]
    #data = [trace, trace_avg1, trace_avg2, trace_ema1, trace_ema2]
    if chkFib == True:
        end_date_fib = datetime.now() - timedelta(days=0)
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
        trace_l01 = plotly.graph_objs.Scatter(x = x01, y = y01, opacity = 0.15, line = dict(color='red', dash='dash', width=esp), fill = 'toself', showlegend = False)
        trace_l12 = plotly.graph_objs.Scatter(x = x01, y = y12, opacity = 0.15, line = dict(color='yellow', dash='dash', width=esp), fill = 'toself', showlegend = False)
        trace_l23 = plotly.graph_objs.Scatter(x = x01, y = y23, opacity = 0.15, line = dict(color='green', dash='dash', width=esp), fill = 'toself', showlegend = False)
        trace_l34 = plotly.graph_objs.Scatter(x = x01, y = y34, opacity = 0.15, line = dict(color='cyan', dash='dash', width=esp), fill = 'toself', showlegend = False)
        trace_l45 = plotly.graph_objs.Scatter(x = x01, y = y45, opacity = 0.15, line = dict(color='blue', dash='dash', width=esp), fill = 'toself', showlegend = False)
        
        trace_l0 = plotly.graph_objs.Scatter(x = all_data_fib['Date'], y = all_data_fib['l0'], opacity = 0.7, line = dict(color='rgb(0, 0, 255)', dash='dash', width=esp), showlegend = False)
        trace_l1 = plotly.graph_objs.Scatter(x = all_data_fib['Date'], y = all_data_fib['l1'], opacity = 0.7, line = dict(color='rgb(0, 0, 255)', dash='dash', width=esp), showlegend = False)
        trace_l2 = plotly.graph_objs.Scatter(x = all_data_fib['Date'], y = all_data_fib['l2'], opacity = 0.7, line = dict(color='rgb(0, 0, 255)', dash='dash', width=esp), showlegend = False)
        trace_l3 = plotly.graph_objs.Scatter(x = all_data_fib['Date'], y = all_data_fib['l3'], opacity = 0.7, line = dict(color='rgb(0, 0, 255)', dash='dash', width=esp), showlegend = False)
        trace_l4 = plotly.graph_objs.Scatter(x = all_data_fib['Date'], y = all_data_fib['l4'], opacity = 0.7, line = dict(color='rgb(0, 0, 255)', dash='dash', width=esp), showlegend = False)
        trace_l5 = plotly.graph_objs.Scatter(x = all_data_fib['Date'], y = all_data_fib['l5'], opacity = 0.7, line = dict(color='rgb(0, 0, 255)', dash='dash', width=esp), showlegend = False)
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
    
    if ckRenko == True:
        all_data_renko = all_data.reset_index('Ticker')
        bricks = round(ATR(all_data_renko,50)["ATR"][-1],2) 
        figrenko, ax =  fplt.plot(all_data_renko, type='renko',renko_params=dict(brick_size=bricks, atr_length=14), style='yahoo', title = "Renko Chart", mav=(10), volume=False, figsize =(20, 5), tight_layout=False, returnfig = True) #panel_ratios=(3,1)
        st.set_option('deprecation.showPyplotGlobalUse', False)   
        st.pyplot(figrenko)
        
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
    
    trace_obv = plotly.graph_objs.Scatter(x = obv.index, y = obv.volume_sum, opacity = 1, name='OBV', showlegend = True)
    
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
    trace_dividends = plotly.graph_objs.Bar(x = data_d['Year'], y = data_d['Dividends'], marker_color = 'blue', name = 'Dividends')
    
    if ckDividends == True:
        datad = [trace_dividends]
        fig_dividends = simple_plot(datad, '')
        fig_dividends.update_layout(width = 1000, height = 280)
        st.plotly_chart(fig_dividends, use_container_width = False)
    
########## algorithm for opportunities
    
    if username == 'admin' or username == 'user':  
        start_date_min = end_date - timedelta(days=int(365*4)) # ultimos 4 anos
        #end_date_min = datetime.now() - timedelta(days=30)
        threshold = 1.3
            
        if st.button('Check for Opportunities'):
            
            st.write('Running... It may take a while...')
            
            tickers_list = [*set(tickers_list)]
            t1list = tickers_list
            #t1list.pop()
            all_data_full = get(t1list, start_date, end_date)
                    
            datac = all_data_full.reset_index()
            
            datacc = datac[datac['Date']<datetime.today()-timedelta(days=0)]
            
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
                r_last = aux.iloc[-1].to_numpy()
                MAlast = all_data_full.loc[tick].Close.rolling(window = 10).mean().dropna()
                
                auxmv = pd.DataFrame([])
                auxmv = all_data_full.loc[tick].dropna()
                #print(tick)
                mm1v = get_ema(12, auxmv.Close)
                mm2v = get_ema(24, auxmv.Close)
                mm_macdv = mm1v.EMA - mm2v.EMA
                mm_signalv = get_ema(9, mm_macdv.dropna()).EMA
                hist_macdv = mm_macdv - mm_signalv
                
                if close[tick][-1] <= (min_value * threshold) and r_last[0] >= 0.99*MAlast[-1] and hist_macdv[-1] >= 0:
                    data_min = pd.DataFrame([])
                    data_min = aux.loc[aux[tick] == min_value]
                    df_table.at[i, 'Ticker'] = str(tick)
                    df_table.at[i, 'Current Price'] = round(close[tick][-1],2)
                    df_table.at[i, 'Min Value History'] = round(min_value,2)
                    df_table.at[i, 'Date Min Value'] = data_min.index[0].date()
                    df_table.at[i, 'Delta Price'] = round(close[tick][-1],2) - round(min_value,2)
                    i += 1
            
            st.write('Completed... Generating Results...')
            
            df_table = df_table.sort_values('Delta Price')
            df_table = df_table.reset_index()
            df_table = df_table.drop('index', axis=1)
            
            st.write('Opportunities - Current Price close to historical price')
            st.dataframe(df_table.style.format({'Current Price': '{:.2f}', 'Min Value History': '{:.2f}', 'Delta Price': '{:.2f}'}))
            
            #df_table.to_excel("opportunities.xlsx")
            #data_p_plot.to_excel("prices.xlsx")
            
            # y = data_p_plot['Ticker']
            # x = data_p_plot['Close']
            
            # trace_prices = go.Bar(x = x, y = y, marker=dict(color='blue'), name = '', orientation='h')
            
            # fig_prices = simple_plot([trace_prices], 'Stock Prices')
            # fig_prices.update_layout(width = 1000, height = 4500)
                
            # st.plotly_chart(fig_prices, use_container_width = False)
            
        
####


