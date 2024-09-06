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


for colunas in colunas_categoricas:
    df[colunas].value_counts().plot(kind = 'bar', color = 'green', figsize=(8,4))
    plt.xlabel(colunas)
    plt.ylabel('Contagem')
    plt.title(f"Gráfico de barras de {colunas}")
    plt.show()


for colunas in colunas_numericas:
    #plt.figure(figsize=(8,4))
    sns.boxplot(data = df, x = colunas)
    plt.xlabel(colunas)
    plt.title(f"Boxplot de {colunas}")
    plt.show()


    df[colunas].hist(bins = 10, color = 'blue')
    plt.xlabel(colunas)
    plt.ylabel("Fraquência")
    plt.title(f"Histograma de {colunas}")
    plt.show()


    print("Distribuição de ", df[colunas].describe())


    nulos_por_coluna = df[colunas].isnull().sum()
    print(nulos_por_coluna)
          