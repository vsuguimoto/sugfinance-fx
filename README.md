# Sugfinance - Forex

[Link para o APP](https://sugfinance-fx.herokuapp.com/)

Hello World!

Sugfinance é um HUB para treinamento de modelos de Machine Learning voltados para previsão de retorno dos pares. O alvo é prever se o preço do par irá subir ou cair para o dia seguinte.

O usuário pode escolher quais indicadores técnicos o mais agrada e treinar e validar um modelo de Regressão Logística em tempo real.

**Pares Disponíveis:**
- EUR-USD;
- GBP-USD;
- BRL-USD.


**Features Disponíveis:**

- FT_SMA_9: Média Móvel Aritmética Simples de 9 periodos;
- FT_BB_OutUpperBand: Preço fechou fora da Banda superior de uma Banda de Bolling de 20 períodos com 2 desvios padrões;
- FT_BB_OutLowerBand: Preço fechou fora da Banda inferior de uma Banda de Bolling de 20 períodos com 2 desvios padrões;
- FT_RSI_Overbough: *Relative Strength Index* de 20 períodos acima de 70 (Sobrecomprado);
- FT_RSI_Oversold: *Relative Strength Index* de 20 períodos abaixo de 30 (Sobrevendido);
- FT_CCI_H: *Commodity channel index* acima de 120;
- FT_CCI_L: *Commodity channel index* abaixo de -120.