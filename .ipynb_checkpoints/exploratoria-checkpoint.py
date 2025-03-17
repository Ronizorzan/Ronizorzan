import psycopg2
import matplotlib.pyplot as plt
import seaborn as sns
import yaml
import pandas as pd
import const


def fetch_data_from_db(sql_query):

    try:
        with open('config.yaml', 'r') as file:
            config = yaml.safe_load(file)



            con = psycopg2.connect(
            dbname = config['database_config']['dbname'],
            user = config['database_config']['user'],
            password =  config['database_config']['password'],
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



df['idade'] = df['idade'].astype(int)
df['valorsolicitado'] = df['valorsolicitado'].astype(float)
df['valortotalbem'] = df['valortotalbem'].astype(float)



colunas_categoricas = ['profissao', 'tiporesidencia', 'escolaridade', 'score', 'estadocivil', 'produto']
colunas_numericas = ['tempoprofissao', 'renda', 'idade', 'dependentes', 'valorsolicitado', 'valortotalbem']


def plot_bars(df, colunas_categorias):
    for colunas in df[colunas_categoricas]:
        fig, ax = plt.subplots()
        agrupado = df[colunas_categorias].value_counts()
        bars = plt.bar(agrupado.index, agrupado.values.flatten(), width=0.9)
        sns.set_theme("paper", "whitegrid")

        for bar in bars:
            height = bar.get_height()
            ax.annotate("{}".format(height),
                        xy=bar.get_x() + bar.get_height() /2, 
                        xycoords="offset points", xytext=(0,3))
            
        
        plt.xlabel(colunas, fontsize=15, fontweight="bold")
        plt.ylabel(f'Contagem de {colunas}')
        plt.title(f"Gráfico de barras de {colunas}", fontsize=20, fontweight="bold")
        plt.show()


def plot_hist_boxplot(df, colunas_numericas):
    for colunas in colunas_numericas:
        fig, ax = plt.subplots()
        sns.set_theme("paper", "whitegrid") 

        sns.histplot(df, colunas, kde=True, bins="auto")        
        plt.title(f"Histograma de {colunas}", fontsize=19, fontweight="bold")
        plt.xlabel(f"Distribuição de {colunas}", fontsize=15, fontweight="bold")
        plt.ylabel(f"Valor de {colunas}", fontsize=15, fontweight="bold")        
        plt.show()


        sns.boxplot(df, colunas)
        plt.title(f"Histograma de {colunas}", fontsize=19, fontweight="bold")
        plt.xlabel(f"Valor de {colunas}", fontsize=15, fontweight="bold")        
        plt.ylabel("Fraquência", fontsize=15, fontweight="bold")        
        plt.show()


        print("Distribuição de ", df[colunas].describe())


        nulos_por_coluna = df[colunas].isnull().sum()
        print(nulos_por_coluna)
          
          