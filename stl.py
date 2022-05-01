import streamlit as st
import fin_data



def main():
    
    
    st.set_page_config(page_title='SugFinance',
                   layout="centered", initial_sidebar_state="auto", menu_items=None)
    
    pair = st.sidebar.selectbox("Selecione o par", ["EUR-USD", "GBP-USD","BRL-USD"])
    
    translate = {"EUR-USD":'EURUSD=X',
                 "BRL-USD":'BRLUSD=X',
                 "GBP-USD":"GBPUSD=X"}
    
    return analysis(translate[pair])
    

@st.cache
def download_data(pair):
    
    df = fin_data.download_transform(pair)
    
    return df


def analysis(pair):
    
    cached_data = download_data(pair)
    
    df = cached_data[:]
    
    st.title('Indicador de Compra e Venda Forex')
    
    FEATURES = st.multiselect(
        'Features disponíveis',
        df.columns[df.columns.str.startswith('FT_')]
        )
    if st.button('Analisar'):
        df, model = fin_data.train_test_predict(df, FEATURES, 'TARGET')
        df = fin_data.model_prediction_return(df, 'PREDICT', 'TARGET', 'RETURN')
        df = fin_data.bnh_return(df, 'Close')
        fig = fin_data.fig_returns(df)
        
        m_col1, m_col2, m_col3 = st.columns(3)
        
        with m_col1:
            st.metric('Acurácia do Modelo', f'{fin_data.metric_accuracy(df):.2f}%', delta=None, delta_color="normal")
        
        with m_col2:
            st.metric('Drawdown Máximo do Modelo', f'{fin_data.metric_max_dd(df, "MODEL_RETURN"):.2f}%')
        
        with m_col3:
            st.metric(f'Decisão para {fin_data.model_last_decision(df, "PREDICT", "Date")[1]}', f'{fin_data.model_last_decision(df, "PREDICT", "Date")[0]}')
        
        
        
        
        st.plotly_chart(fig)
        
        st.header('Informações Estatísticas do Modelo')
        st.write(model.summary())
    
    
    
    


    
if __name__ == "__main__":
    main()