#Funções de Plotagem e Conexão com o banco de dados
import pandas as pd
import psycopg2
import yaml
import const
from funcoes import *
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
import numpy as np


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
    return fig


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

        
def calcular_metricas(matriz_confusao):
    """
    Calcula taxas de inadimplência, captação de bons clientes e aprovação
    com base na matriz de confusão fornecida.

    Parâmetros:
    matriz_confusao (numpy array): Matriz de confusão [ [VN, FP], [FN, VP] ]

    Retorna:
    dict: Dicionário com as métricas calculadas.
    """
    # Extraindo valores
    VN, FP = matriz_confusao[0][0], matriz_confusao[0][1]
    FN, VP = matriz_confusao[1][0], matriz_confusao[1][1]

    # Total de solicitações
    total_solicitacoes = VN + FP + FN + VP

    # Taxa de inadimplência real
    inadimplencia_real = (VN + FP) / total_solicitacoes * 100

    # Taxa de inadimplência prevista com o uso do modelo
    inadimplencia_prevista = FP / (VN + FP) * 100

    # Captação de bons clientes
    captacao_bons_clientes = VP /  (FN + VP) * 100

    # Taxa de aprovação
    taxa_aprovacao = (VP + FP) / (total_solicitacoes) * 100
    taxa_reprovacao = 100 - taxa_aprovacao

    resultado =  {
        "inadimplencia_sem_modelos": round(inadimplencia_real, 2),
        "inadimplencia_prevista": round(inadimplencia_prevista, 2),
        "captacao_bons_clientes": round(captacao_bons_clientes, 2),
        "taxa_aprovacao": round(taxa_aprovacao, 2),
        "taxa_reprovacao": round(taxa_reprovacao, 2)
    }

    return resultado

def calcular_e_plotar_impacto(matriz_xgb, matriz_seq, valor_medio_emprest, taxa_juros):
    """Retorna um gráfico de barras empilhadas para comparação 
    de ganhos e perdas com e sem o uso dos modelos"""           

    # 1. Cenário Baseline – extraído diretamente da matriz_xgb (todos os clientes)
    bons_baseline = matriz_xgb[1][0] + matriz_xgb[1][1]
    maus_baseline = matriz_xgb[0][0] + matriz_xgb[0][1]
    
    ganhos_baseline = bons_baseline * valor_medio_emprest # Retorno do Valor emprestado
    ganhos_baseline = ganhos_baseline + (ganhos_baseline * taxa_juros )   # Acrescenta os juros no ganho (se houver)
    perdas_baseline = maus_baseline * valor_medio_emprest              # Valor perdido com os maus pagadores
    impacto_baseline = ganhos_baseline - perdas_baseline               # Impacto líquido    
    
    # 2. Modelo XGB – considerando apenas os aprovados (coluna 1 da matriz_xgb)
    bons_xgb = matriz_xgb[1][1] # Bons clientes corretamente identificados
    maus_xgb = matriz_xgb[0][1] # Maus classificados como bons
    bons_xgb_negados = matriz_xgb[1][0] # Bons Classificados como maus
        
    ganhos_xgb = bons_xgb * valor_medio_emprest # Bons classificados como bons
    ganhos_xgb = ganhos_xgb + (ganhos_xgb * taxa_juros) # Acrescenta os juros (se houver)
    perdas_xgb = maus_xgb * valor_medio_emprest # Maus classificados como bons (perda de capital)    
    perda_clientes_xgb = (bons_xgb_negados * taxa_juros) * valor_medio_emprest # Bons classificados como maus (perda de dividendos)
    impacto_xgb = ganhos_xgb - (perdas_xgb + perda_clientes_xgb) # Impacto líquido

    # 3. Modelo Sequential – considerando apenas os aprovados (coluna 1 da matriz_seq)
    bons_seq = matriz_seq[1][1] # Bons clientes corretamente identificados
    maus_seq = matriz_seq[0][1] # Maus classificados como bons
    bons_seq_negados = matriz_seq[1][0] # Bons Classificados como maus

    
    ganhos_seq = bons_seq * valor_medio_emprest # Bons classificados como bons
    ganhos_seq = ganhos_seq + (ganhos_seq * taxa_juros) # Acrescenta os juros (se houver)
    perdas_seq = maus_seq * valor_medio_emprest # Maus classificados como bons (perda de capital)    
    perda_clientes_seq = (bons_seq_negados * taxa_juros) * valor_medio_emprest # Bons classificados como maus (perda de dividendos)
    impacto_seq = ganhos_seq - (perdas_seq + perda_clientes_seq) # Impacto líquido
    
    # Agrupando os resultados para o gráfico
    estrategias = ['Baseline\n(Sem modelos)', 'XGB', 'Redes Neurais']
    ganhos_list   = [ganhos_baseline, ganhos_xgb, ganhos_seq]
    perdas_list   = [perdas_baseline, perdas_xgb, perdas_seq]    
    
    x = np.arange(len(estrategias))
    width = 0.7
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plota a parte dos ganhos (segmento verde) – a partir do zero
    ax.bar(x, ganhos_list, width, label='Valores Ganhos', color='green')#, edgecolor="black", linewidth=1.5)
    ax.bar(x, perdas_list, width, label='Valores Perdas', color='red')#, edgecolor="black", linewidth=1.5)
          
    #sns.despine(right=True, top=True)
    ax.yaxis.set_major_formatter(mtick.StrMethodFormatter("R$ {x:,.0f}"))
    ax.set_xticks(x)
    ax.set_xticklabels(estrategias)
    ax.set_xlabel('Estratégia', fontsize=15)
    ax.set_ylabel('Valor (R$)', fontsize=15)
    ax.set_title('Impacto Financeiro: Perdas e Ganhos por Estratégia', fontsize=15)
    ax.axhline(0, color='black', linewidth=0.9)
    ax.legend(bbox_to_anchor=(0.08,-0.02))
    plt.tight_layout()

     # Consolidação dos resultados em um dicionário
    resultados = {
        "Baseline": {
            "ganhos": ganhos_baseline,
            "perdas": perdas_baseline,
            "retorno_liquido": impacto_baseline
        },
         "XGB": {            
            "ganhos": ganhos_xgb,
            "perdas": perdas_xgb,
            "retorno_liquido": impacto_xgb
         },
        "Sequential": {            
        "ganhos": ganhos_seq,
        "perdas": perdas_seq,
        "retorno_liquido": impacto_seq          
        }}
    df = pd.DataFrame(resultados)

    return df, plt.gcf()
   


def plot_inadimplencia(inadimplencia_sem_modelo, inadimplencia_xgb, inadimplencia_seq):
    """Cria um gráfico de barras comparando a inadimplência nos diferentes cenários."""
    modelos = ["Baseline\n(Sem Modelo)", "XGBoost", "Redes Neurais"]
    inadimplencia = [inadimplencia_sem_modelo, inadimplencia_xgb, inadimplencia_seq]
    
    # Criando o gráfico de barras
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(x=modelos, y=inadimplencia, palette=["red", "blue", "green"], hue=modelos, 
                edgecolor="black", linewidth=1.5)
    #sns.barplot(x=modelos, y=inadimplencia, ax=ax, edgecolor="black", linewidth=1.5)
    
    # Adicionando detalhes estéticos
    for i, valor in enumerate(inadimplencia):
        ax.text(i, valor + 0.5, f"{valor:.2f}%", ha="center", fontsize=12, fontweight="bold", color="black")

    ax.set_title("Comparação da Taxa de Inadimplência", fontsize=15)
    ax.set_xlabel("Modelo", fontsize=15)
    ax.set_ylabel("Taxa de Inadimplência (%)", fontsize=15)
    ax.set_ylim(0, max(inadimplencia) + 5)  # Ajustando o limite do eixo Y

    return plt.gcf()  # Retorna a figura atual para exibição no Streamlit


def plot_aprovacao_reprovacao(taxa_aprovacao_xgb, taxa_reprovacao_xgb, 
                              taxa_aprovacao_seq, taxa_reprovacao_seq):
    """Cria um gráfico de barras empilhadas para taxa de aprovação e reprovação por modelo."""

    modelos = ["XGBoost", "Redes Neurais"]
    
    valores_aprovacao = [taxa_aprovacao_xgb, taxa_aprovacao_seq]
    valores_reprovacao = [taxa_reprovacao_xgb, taxa_reprovacao_seq]
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Posicionando as barras
    x = np.arange(len(modelos))
    
    ax.bar(x, valores_aprovacao, label="Aprovados", color="blue")
    ax.bar(x, valores_reprovacao, bottom=valores_aprovacao, label="Reprovados", color="red")

    ax.set_xticks(x)
    ax.set_xticklabels(modelos)
    
    sns.despine(right=True, top=True)
    ax.set_title("Taxa de Aprovação de Clientes", fontsize=15)
    ax.set_xlabel("Modelo", fontsize=15)
    ax.set_ylabel("Percentual (%)", fontsize=15)
    ax.legend(bbox_to_anchor=(0.17,-0.01))

    return plt.gcf()


def plot_captacao_bons_clientes(captacao_xgb, captacao_seq):
    """Cria um gráfico comparando a captação de bons clientes entre modelos."""
    modelos = ["XGBoost", "Sequential"]
    captacao = [captacao_xgb, captacao_seq]

    fig, ax = plt.subplots(figsize=(8,6))
    sns.barplot(x=modelos, y=captacao, palette=["blue", "green"], hue=modelos, orient="v", edgecolor="black", linewidth=1.2)

    for i, valor in enumerate(captacao):
        ax.text(i, valor + 0.5, f"{valor:.2f}%", ha="center", fontsize=16, fontweight="bold", color='black')

    ax.set_title("Captação de Bons Clientes por Modelo", fontsize=15)
    ax.set_xlabel("Modelo", fontsize=15)
    ax.set_ylabel("Percentual de Bons Clientes Aprovados (%)", fontsize=15)
    ax.set_ylim(0, max(captacao) + 10)  # Ajustando o eixo Y
    return plt.gcf()
