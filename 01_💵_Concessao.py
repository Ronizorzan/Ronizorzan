#Importação das bibliotecas
import streamlit as st
import pandas as pd
from funcoes import *
from tensorflow.keras.models import load_model
from numpy import load


#Configuração da página
st.set_page_config(page_title="Avaliação de Crédito", layout="centered", initial_sidebar_state="expanded")

    


#Variáveis para preencher as caixas de seleção

profissoes = ['Advogado', 'Arquiteto', 'Cientista de Dados', 'Contador', 'Dentista', 'Empresário', 'Engenheiro', 'Médico', 'Programador']
residencia = ['Alugada', 'Outros', 'Própria']
escolaridades = ['Ens.Fundamental', 'Ens.Médio', 'PósouMais', 'Superior']
scores = ['Baixo', 'Bom', 'Justo', 'MuitoBom']
estados_civis = ['Casado', 'Divorciado', 'Solteiro', 'Víuvo']
produtos = ['AgileXplorer', 'DoubleDuty', 'EcoPrestige', 'ElegantCruise', 'SpeedFury', 'TrailConqueror', 'VoyageRoamer', 'WorkMaster']

with st.sidebar:
        acuracia = load("objects/acuracia.npy") 
                        
        # Título com ícone
        st.markdown("""
        <div style='display: flex; align-items: center; gap: 10px;'>
            <span style='font-size: 24px; font-weight: bold; color: #eeeeee;'>📊 Acurácia do Modelo</span>
        </div>
        """, unsafe_allow_html=True)

        # Valor da acurácia com destaque visual na barra lateral     
        st.markdown(f"""
        <div style='margin-top: 10px; padding: 15px; background-color: #f9f9f9; border-left: 7px solid #972328; border-radius: 8px;'>
            <span style='font-size: 36px; font-weight: bold; color: #972328;'>{acuracia*100:.2f}%</span>
            <br>
            <span style='font-size: 16px; color: #555;'>Desempenho baseado nos dados de validação</span>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("---")
           
        st.markdown("""
        <div style='display: flex; align-items: center; gap: 10px;'>
            <span style='font-size: 24px; font-weight: bold; color: #eeeeee;'>🔗Links do desenvolvedor:</span>
        </div>
        """, unsafe_allow_html=True)
        
        # Rodapé na barra lateral com as informações do desenvolvedor
        st.markdown("""
<style>
.footer {
    background-color: #f8f9fa;
    padding: 20px 25px;
    border-radius: 10px;
    border-left: 9px solid #972328;
    text-align: center;
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    margin-top: 20px;
    color: #343a40;
    box-shadow: 0 2px 6px rgba(0,0,0,0.1);
}
.footer p {
    font-size: 16px;
    margin-bottom: 15px;
}
.footer a {
    margin: 0 12px;
    display: inline-block;
}
.footer img {
    height: 36px;
    width: 36px;
    border-radius: 6px;
    transition: transform 0.3s ease, box-shadow 0.3s ease;
}
.footer img:hover {
    transform: scale(1.1);
    box-shadow: 0 0 8px rgba(151, 35, 40, 0.4);
}
</style>

<div class="footer">
    <p><strong>Desenvolvido por Ronivan Zorzan Barbosa</strong></p>
    <a href="https://github.com/Ronizorzan/Ronizorzan/blob/master/" target="_blank">
        <img src="https://cdn-icons-png.flaticon.com/512/25/25231.png" alt="GitHub">
    </a>
    <a href="https://www.linkedin.com/in/ronivan-zorzan-barbosa" target="_blank">
        <img src="https://cdn-icons-png.flaticon.com/512/174/174857.png" alt="LinkedIn">
    </a>
    <a href="mailto:ronizorzan1992@gmail.com" target="_blank">
        <img src="https://cdn-icons-png.flaticon.com/512/732/732200.png" alt="Email">
    </a>
    <a href="https://share.streamlit.io/user/ronizorzan" target="_blank">
        <img src="https://images.seeklogo.com/logo-png/44/1/streamlit-logo-png_seeklogo-441815.png" alt="Streamlit">
    </a>
</div>
""", unsafe_allow_html=True)
        
        

#Formação das caixas de seleção e submit_button
with st.form(key = 'prediction_form'):
    with st.expander("Insira as características do novo cliente", expanded=True):
        profissao = st.selectbox('profissão', profissoes)
        tempoprofissao = st.number_input('Tempo na profissão (em anos)', min_value=0, value=25, step=1, max_value=70)
        renda = st.number_input("Renda", min_value=0.0, value=50000.0, step = 1000.0)
        tiporesidencia = st.selectbox("Tipo de residência", residencia)
        escolaridade = st.selectbox("Escolaridade", escolaridades)
        idade = st.number_input("Idade", min_value=18, value=28, step=1, max_value=110)
        dependentes = st.number_input('Dependentes', min_value = 0, value = 0, step = 1)
        estadocivil = st.selectbox("Estado civil", estados_civis)
        score = st.selectbox("Score", scores)
        produto = st.selectbox("Produto", produtos)
        valorsolicitado = st.number_input("Valor solicitado", min_value=1000.0, value=25000.0, step=1000.0)        
        valortotalbem = st.number_input("Valor total do bem", min_value=1000.0, value=100000.0, step=1000.0)

        submit_button = st.form_submit_button(label= "Consultar")


if submit_button:
    progresso = st.progress(50, "Aguarde um momento.. Processando os dados inseridos")
        
    dados_novos = {
    'profissao': [profissao],
    'tempoprofissao' : [tempoprofissao],
    'renda' : [renda],
    'tiporesidencia': [tiporesidencia],
    'escolaridade': [escolaridade],
    'score': [score],
    'idade': [idade],
    'dependentes': [dependentes],
    'estadocivil': [estadocivil],
    'produto': [produto],
    'valorsolicitado': [valorsolicitado],
    'valortotalbem': [valortotalbem]
    }

    caminho_modelo = 'meu_modelo.h5'
    modelo = load_model(caminho_modelo) # Carregamento do modelo treinado
    seletor = joblib.load("objects/seletor.joblib") # Carregamento do seletor de atributos
    
    df = pd.DataFrame(dados_novos)
    df = load_scalers(df,['tempoprofissao','renda','idade','dependentes','valorsolicitado','valortotalbem'] )
    df = load_encoders(df, ['profissao', 'tiporesidencia',  'escolaridade','score','estadocivil','produto'])
    df = seletor.transform(df)
    
    predictions = modelo.predict(df)
    probabilidade = predictions[0][0] * 100
    classe = "Bom" if probabilidade > 50 else "Ruim"
    if classe=="Bom":
        st.success(f"Resultado: {classe}")
        st.success(f'Probabilidade de inadimplência:{probabilidade:.2f}%')
    else:
        st.error(f"Resultado: {classe}")
        probabilidade = 100 - probabilidade
        st.error(f"Probabilidade de inadimplência: {probabilidade:.2f}%")

    progresso.progress(100, "Dados processados com sucesso...")








