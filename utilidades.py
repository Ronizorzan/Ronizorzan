
#importação das bibliotecas
import pandas as pd
import psycopg2
import yaml
import const

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




