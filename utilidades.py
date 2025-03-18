#Funções de Plotagem e Conexão com o banco de dados
import pandas as pd
import psycopg2
import yaml
import const
from funcoes import *
import matplotlib.pyplot as plt
import seaborn as sns


#Recupera os dados diretamente do banco de dados
@monitor_funcoes
def fetch_data_from_db(sql_query):
    try:
        with open('config.yaml', 'r') as file:
            config = yaml.safe_load(file)

            con = psycopg2.connect(

            dbname = config['database_config']['dbname'],
            user = config['database_config']['user'],
            password = config['database_config']['password'],
            host = config['database_config']['host']

            )

            cursor = con.cursor()
            cursor.execute(sql_query)

            df = pd.DataFrame(cursor.fetchall(), columns = [desc[0] for desc in cursor.description])
            return df    
    
    finally:

        if 'cursor' in locals():
            cursor.close()

        if 'con' in locals():
            con.close()

    df = fetch_data_from_db(const.consulta_sql)


#Plotagem gráfico de Barras
@monitor_funcoes
def plot_bars(df, colunas_categorias, color):
    # Configuração global do tema
    sns.set_theme(style="darkgrid", context="talk")

    for coluna in colunas_categorias:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Agrupamento e cálculos
        agrupado = df[coluna].value_counts(normalize=False)
        percentuais = df[coluna].value_counts(normalize=True) * 100
        
        # Gráfico de barras
        bars = ax.bar(agrupado.index, agrupado.values, color=sns.color_palette(color))

        # Exibe contagens e porcentagens
        for bar, percentual in zip(bars, percentuais):
            height = bar.get_height()
            ax.annotate(f"{height}\n({percentual:.1f}%)",
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        textcoords="offset points", xytext=(0, 5),
                        ha="center", fontsize=10)

        # Customizações do gráfico
        sns.despine(top=True, right=True, left=False)
        plt.xticks(rotation=20, ha="right", fontsize=8)
        plt.yticks(fontsize=12)
        ax.set_xlabel(coluna, fontsize=14, fontweight="bold")
        ax.set_ylabel("Frequência", fontsize=14, fontweight="bold")
        ax.set_title(f"Distribuição de {coluna}", fontsize=16, fontweight="bold")
        
        # Ajuste layout
        plt.tight_layout()
        plt.show()





#Plotagem Histogramas
@monitor_funcoes
def plot_hist(df, colunas_numericas):    
    sns.set_theme("poster", "darkgrid") 
    fig, ax = plt.subplots()
    
    for coluna in df[colunas_numericas]:        
                    
        sns.histplot(data=df, x = coluna, kde=True, ax=ax)
        ax.set_title(f"Histograma de {coluna}", fontsize=16, fontweight="bold")
        ax.set_ylabel(f"Frequência de {coluna}", fontsize=12, fontweight="bold")
        ax.set_xlabel(coluna, fontsize=12, fontweight="bold")
        sns.despine(right=True, top=True)            
        plt.tight_layout()
        plt.show()


#Plotagem Boxplots e resumos
@monitor_funcoes
def plot_boxplot(df, colunas_numericas):
    for colunas in colunas_numericas:
        fig, ax = plt.subplots()
        sns.set_theme("poster", "darkgrid")

        sns.boxplot(df, y=colunas)
        plt.title(f"Boxplot de {colunas}", fontsize=16, fontweight="bold")
        plt.xlabel(f"Distribuição de {colunas} ", fontsize=12, fontweight="bold")
        plt.ylabel(f"Valor de {colunas} ", fontsize=12, fontweight="bold")
        plt.show()

        

