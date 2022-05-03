import streamlit as st


def download_transform(ticker: str):
    '''
    ticker: string com o nome do ticker no Yahoo Finance
    
    return: Pandas Dataframe com todos as features disponíveis
    
    
    Função utilizada para baixar os dados do ticker utilizando
    o pacote yfinance.
    Utiliza-se um filtro para remover colunas que não agregam
    informação, ou seja não possuem valores distintos.
    '''
    

    import pandas as pd
    import yfinance as yf

    from ta.volatility import BollingerBands, AverageTrueRange
    from ta.momentum import RSIIndicator
    from ta.trend import CCIIndicator, SMAIndicator

    df = yf.download(ticker, start=pd.Timestamp.today(
    ) - pd.Timedelta(365*5, 'days'), end=pd.Timestamp.today()).reset_index()

    df['Close_shift'] = df.Close.shift(1)
    df['RETURN'] = (df.Close/df.Close_shift) - 1

    df['TARGET'] = [1 if x >= 0 else 0 for x in df.RETURN]

    # Criando Variáveis
    # Utilizarei alguns indicadores técnicos clássicos
    # Referência utilizada -> Leandro Guerra Outspoken Market

    df['FT_BB_OutUpperBand'] = BollingerBands(close=df.Close_shift,
                                           window=20,
                                           window_dev=2,
                                           fillna=False).bollinger_hband_indicator()

    df['FT_BB_OutLowerBand'] = BollingerBands(close=df.Close_shift,
                                           window=20,
                                           window_dev=2,
                                           fillna=False).bollinger_lband_indicator()

    df['RSI'] = RSIIndicator(close=df.Close_shift,
                             window=20,
                             fillna=False).rsi()

    df['FT_RSI_Overbough'] = [1 if x > 70 else 0 for x in df.RSI]
    df['FT_RSI_Oversold'] = [1 if x < 30 else 0 for x in df.RSI]

    df['FT_ATR'] = AverageTrueRange(close=df.Close_shift,
                      high=df.High.shift(1),
                      low=df.Low.shift(1),
                      fillna=False).average_true_range()

    df['CCI'] = CCIIndicator(high=df.High.shift(1),
                             low=df.Low.shift(1),
                             close=df.Close_shift).cci()
    df['FT_CCI_H'] = [1 if x > 120 else 0 for x in df.CCI]
    df['FT_CCI_L'] = [1 if x < -120 else 0 for x in df.CCI]

    df['FT_SMA_9'] = SMAIndicator(close=df.Close_shift,
                               window=9).sma_indicator()

    
    df = df.dropna()
    
    # Remove colunas que não trazen informação
    non_unique_cols = [c for c in df.columns if len(df[c].unique()) > 1]
    df = df[non_unique_cols]
    
    return df


def train_test_predict(df, FEATURES: list, TARGET: list):
    '''
    df: Pandas DataFrame com todas as features
    FEATURES: lista com as features que serão utilizadas para treinar
    o modelo
    TARGET: nome da coluna alvo
    
    return: Pandas DataFrame com as previsões, Modelo fitado da regressão
    logística 
    '''
    
    import pandas as pd
    from statsmodels.discrete.discrete_model import Logit
    import statsmodels.api as sm

    DF_LEN = len(df)

    TRAIN = df[:round(DF_LEN/2)]
    TEST = df[round(DF_LEN/2):]

    lr = Logit(exog=sm.add_constant(TRAIN[FEATURES]),
               endog=TRAIN[TARGET]).fit()
    
    df['PREDICT'] = [1 if x >=.5 else 0 for x in lr.predict(sm.add_constant(df[FEATURES]))]
    
    return df, lr


def model_prediction_return(df, prediction_col:str, target_col:str, return_col:str):
    '''
    df: Pandas Dataframe com a coluna de previsões realizadas pelo modelo
    prediction_col: string com o nome da coluna de previsões do modelo
    return_col: string com o nome da coluna dos retornos do modelo
    
    return: Pandas Dataframe com uma coluna de retorno das previsões do modelo
    '''
    
    
    import pandas as pd
    
    df.loc[:,'CORRECT_PREDICTION'] = [1 if x == y else 0 for x,y in zip(df[prediction_col], df[target_col])]
    df.loc[:,'MODEL_RETURN_CORRECTION'] = [abs(x) if y == 1 else -abs(x) for x,y in zip(df[return_col], df['CORRECT_PREDICTION'])]
    df.loc[:,'MODEL_RETURN_PLUS_ONE'] = df.loc[:,'MODEL_RETURN_CORRECTION'] + 1
    df.loc[:,'MODEL_RETURN'] = (df.loc[:,'MODEL_RETURN_PLUS_ONE'].cumprod()) - 1
    
    return df.drop(['CORRECT_PREDICTION','MODEL_RETURN_CORRECTION', 'MODEL_RETURN_PLUS_ONE'], axis=1)


def bnh_return(df, close_price_col):
    '''
    df: Pandas Dataframe com o preço de fechamento
    
    return: Pandas Dataframe com o retorno do ativo por Buy and Hold
    '''
    
    
    df.loc[:,'BNH_Return'] = (df.loc[:,close_price_col]/df.loc[:,close_price_col].iloc[0]) - 1
    
    return df


def metric_accuracy(df):
    '''
    df: Pandas Dataframe com o alvo e as previsões do modelo
    
    return: Acurácia em percentual do modelo
    '''
    
    from sklearn.metrics import accuracy_score
    
    acc_score = accuracy_score(df.TARGET, df.PREDICT)
    
    return acc_score * 100


def metric_max_dd(df, return_prediction_col):
    '''
    df: Pandas Dataframe com a coluna do retorno das previsões do modelo
    return_prediction_col: string com o nome da coluna de retorno das previsões
    
    return: drawdown maximo do modelo em percentual
    
    Referência utilizada
    https://quant.stackexchange.com/questions/55130/global-maximum-drawdown-and-maximum-drawdown-duration-implementation-in-python
    '''
    import pandas as pd
    

    highwatermarks = df[return_prediction_col].cummax()

    drawdowns = (1 + highwatermarks)/(1 + df[return_prediction_col]) - 1

    max_drawdown = max(drawdowns)
        
    return max_drawdown * 100
  
    
def model_last_decision(df, predict_col:str, date_col:str):
    '''
    df: Pandas Dataframe com a coluna de previsões do modelo e data 
    predict_col: string com o nome da coluna de previsões
    date_col: string com o nome da coluna de data
    
    return: previsão mais recente do modelo e a data da previsão formatada
    '''
    
    import pandas as pd
    
    last_date = df[date_col].max()
    decision = ['Buy' if x == 1 else 'Sell' for x in df[df[date_col] == last_date][predict_col]][0]
    
    formated_date = last_date.strftime('%d/%m/%Y')
    
    return decision, formated_date


def fig_returns(df):
    '''
    df: Pandas dataframe com o retorno por Buy and Hold e Retorno do modelo e Data
    
    return: Plotly figure, gráfico comparando o retorno por Buy and Hold ou por seguir
    o modelo
    
    Nota: As colunas estão Hard Coded como BNH_Return, MODEL_RETURN, Date
    '''
    
    
    import plotly.graph_objects as go
    

    return_fig = go.Figure(
    data=(
        go.Scatter(x=df.Date,
                   y=df.BNH_Return,
                   name='Retorno: Buy and Hold'
                   ),
        go.Scatter(x=df.Date,
                   y=df.MODEL_RETURN,
                   name='Retorno: Modelo')
        ),
    layout=go.Layout(
        title=go.layout.Title(text="Comparação entre os Retornos")
        )
    )
    
    
    return return_fig