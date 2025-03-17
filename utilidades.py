#Funções de Plotagem e Conexão com o banco de dados
import pandas as pd
import psycopg2
import yaml
import const
import matplotlib.pyplot as plt
import seaborn as sns


#Recupera os dados diretamente do banco de dados
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


#Plotagem Gráfico de Barras
def plot_bars(df, colunas_categorias):
    for colunas in df[colunas_categorias]:
        fig, ax = plt.subplots()
        agrupado = df.groupby(colunas).size()
        bars = plt.bar(agrupado.index, agrupado.values.flatten(), width=0.9)
        sns.set_theme("paper", "whitegrid")

        for bar in bars:
            height = bar.get_height()
            ax.annotate("{}".format(height),
                        xy=(bar.get_x() + bar.get_width() /2, height),                         
                        textcoords="offset points", xytext=(0,3), fontsize=12,
                       va="bottom", ha="center")
            
        
        sns.despine(top=True, right=True, left=False)
        plt.xticks(rotation=45, ha="right")
        plt.xlabel(colunas, fontsize=15, fontweight="bold")
        plt.ylabel(f'Contagem de {colunas}', fontsize=15, fontweight="bold")        
        plt.title(f"Gráfico de barras de {colunas}", fontsize=20, fontweight="bold")
        plt.tight_layout()
        plt.show()



#Plotagem Histogramas
def plot_hist(df, colunas_numericas):
    for coluna in df[colunas_numericas]:
        
            
        fig, ax = plt.subplots()
        sns.set_theme("notebook")
        sns.histplot(data=df, x = coluna, kde=True, ax=ax)
        ax.set_title(f"Histograma de {coluna}", fontsize=18, fontweight="bold")
        ax.set_ylabel(f"Frequência de {coluna}", fontsize=14, fontweight="bold")
        ax.set_xlabel(coluna, fontsize=14, fontweight="bold")
        sns.despine(right=True, top=True)            
        plt.tight_layout()
        plt.show()


#Plotagem Boxplots e resumos
def plot_boxplot(df, colunas_numericas):
    for colunas in colunas_numericas:
        fig, ax = plt.subplots()
        sns.set_theme("paper")

        sns.boxplot(df, y=colunas)
        plt.title(f"Boxplot de {colunas}", fontsize=18, fontweight="bold")
        plt.xlabel(f"Distribuição de {colunas} ", fontsize=12, fontweight="bold")
        plt.ylabel(f"Valor de {colunas} ", fontsize=12, fontweight="bold")
        plt.show()

        

