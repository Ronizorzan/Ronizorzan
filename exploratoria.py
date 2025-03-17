#Análise Exploratória dos dados
import matplotlib.pyplot as plt
from utilidades import *
import const


#Recupera os dados do banco de dados        
df = fetch_data_from_db(const.consulta_sql)



#Transformação de Tipos
df['idade'] = df['idade'].astype(int)
df['valorsolicitado'] = df['valorsolicitado'].astype(float)
df['valortotalbem'] = df['valortotalbem'].astype(float)


#Colunas do dataframe
colunas_categoricas = ['profissao', 'tiporesidencia', 'escolaridade', 'score', 'estadocivil', 'produto']
colunas_numericas = ['tempoprofissao', 'renda', 'idade', 'dependentes', 'valorsolicitado', 'valortotalbem']


#Funções de Plotagem do arquivo utilidades
plot_bars(df, colunas_categoricas)
plot_boxplot(df, colunas_numericas)
plot_hist(df, colunas_numericas)



#Análise estatística
for coluna in df[colunas_numericas]:
    df[coluna].describe()


#Após análise exploratória foram encontradas inconsistências como valores faltantes, outliers, 
#Erros de Digitação entre outros em várias colunas do dataframe.
#Vamos tratar esses valores e visualizar os dados tratados no notebook "exploratoria.ipynb"