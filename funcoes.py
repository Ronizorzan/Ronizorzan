#Funções para tratamento de dados, padronização e codificação
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from fuzzywuzzy import process
import joblib
from time import time


#Função para monitoramento de desempenho
def monitor_funcoes(func):
    def contador_tempo(*args, **kwargs):
        hora_inicial = time()
        resultado = func(*args, **kwargs)
        hora_final = time()
        print("Função {} executada em {:.2f} segundos ".format(func.__name__, (hora_final - hora_inicial )))
        return resultado
    return contador_tempo


#Função para tratamento de nulos
@monitor_funcoes
def substitui_nulos(df):
    for coluna in df.columns:
        if df[coluna].dtype == 'object':
            moda = df[coluna].mode()[0]
            df.fillna({coluna: moda}, inplace = True)


        else:
            mediana = df[coluna].median()
            df.fillna({coluna: mediana}, inplace = True)
    
    return df


#Função para Correção de Erros de digitação
@monitor_funcoes
def corrigir_erros_digitacao(df, coluna, lista_valida):
    for i, valor in enumerate(df[coluna]):
        valor_str = str(valor) if pd.notnull(valor) else valor

        if valor_str not in lista_valida and pd.notnull(valor_str):
            correcao = process.extractOne(valor_str, lista_valida)[0]
            df.at[i, coluna] = correcao
    
    return df


#Função para tratamento de outliers
@monitor_funcoes
def tratar_outliers(df, coluna, minimo, maximo):
    mediana = df[(df[coluna] > minimo) & (df[coluna] < maximo)][coluna].median()
    df[coluna] = df[coluna].apply(lambda x : mediana if x < minimo or x > maximo else x)

    return df


#Função para padronização
@monitor_funcoes
def save_scalers(df, nome_colunas):
    for nome_coluna in nome_colunas:
        scaler = StandardScaler()
        df[nome_coluna] = scaler.fit_transform(df[[nome_coluna]])
        joblib.dump(scaler, f'./objects/scalers{nome_coluna}.joblib')

    return df


#Função para codificação
@monitor_funcoes
def save_encoders(df, nome_colunas):
    for nome_coluna in nome_colunas:
        labelencoder = LabelEncoder()
        df[nome_coluna] = labelencoder.fit_transform(df[nome_coluna])
        joblib.dump(labelencoder, f'./objects/encoders{nome_coluna}.joblib')

    return df


#Função para carregamento dos padronizadores
@monitor_funcoes
def load_scalers(df, nome_colunas):
    for nome_coluna in nome_colunas:
        arquivo_scaler = f'./objects/scalers{nome_coluna}.joblib'
        scaler = joblib.load(arquivo_scaler)
        df[nome_coluna] = scaler.transform(df[[nome_coluna]])

    return df


#Função para carregamento dos codificadores
@monitor_funcoes
def load_encoders(df, nome_colunas):
    for nome_coluna in nome_colunas:
        arquivo_encoder = f'./objects/encoders{nome_coluna}.joblib'
        encoder = joblib.load(arquivo_encoder)
        df[nome_coluna] = encoder.transform(df[nome_coluna])

    return df

# Função para aplicar estilos
def highlight_models(row):
    color_map = {
        'Baseline': 'background-color: red',
        'XGB': 'background-color: orange',
        'Redes Neurais': 'background-color: green'
    }
    return [color_map.get(row['Modelos'], '')] * len(row)





    



    





    