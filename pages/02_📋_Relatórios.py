import streamlit as st
import numpy as np
from utilidades import *


#Configuração da página
st.set_page_config(page_title="Avaliação de Crédito", layout="wide", initial_sidebar_state="expanded")


matrix_xgb = np.load("objects/confusion_xgb.npy", allow_pickle=False)
matrix_seq = np.load("objects/confusion_seq.npy", allow_pickle=False)
valor_medio_emprest = np.load("objects/valor_medio_emprest.npy")

resultado_xgb = calcular_metricas(matrix_xgb)
resultado_seq = calcular_metricas(matrix_seq)


# Controles da Barra Lateral
with st.sidebar:    

    st.title("Visualização de Resultados")
    with st.expander("Configurações das visualizações", expanded=True):    
        visualizacao = st.radio("Selecione o tipo de visualização", ("Impacto Financeiro", "Redução da Inadimplência",
                                                                      "Captação de Bons Clientes", "Taxa de Aprovação"))
        if visualizacao == "Impacto Financeiro":
            taxa_juros = st.slider("Taxa média de juros", min_value=0, max_value=100, value=29,
                                   help="Selecione a taxa de juros cobrada por empréstimo\
                                    \ne veja como os valores se atualizam no gráfico") / 100
    visualizar = st.button("Visualizar")   

    st.markdown( # Rodapé na barra lateral com as informações do desenvolvedor
        """
        <style>
        .footer {
        background-color: #f8f9fa;
        padding: 15px 20px;
        border-radius: 8px;
        text-align: center;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        margin-top: 40px;
        color: #343a40;
        }
        .footer a {
        margin: 0 15px;
        display: inline-block;
        }
        .footer img {
        height: 40px;
        width: auto;
        transition: transform 0.3s ease;
        }
        .footer img:hover {
        transform: scale(1.1);
        }
        </style>
        <div class="footer">
        <p><strong>Desenvolvido por: Ronivan</strong></p>
        <a href="https://github.com/Ronizorzan" target="_blank">
            <img src="https://img.icons8.com/ios-filled/50/000000/github.png" alt="GitHub">
        </a>
        <a href="https://www.linkedin.com/in/ronivan-zorzan-barbosa" target="_blank">
            <img src="https://img.icons8.com/color/48/000000/linkedin.png" alt="LinkedIn">
        </a>
        <a href="https://share.streamlit.io/user/ronizorzan" target="_blank">
            <img src="https://images.seeklogo.com/logo-png/44/1/streamlit-logo-png_seeklogo-441815.png" alt="Streamlit Community">
        </a>
        </div>
        """,
        unsafe_allow_html=True)


            

if visualizar:
    if visualizacao == "Impacto Financeiro": # Gráfico e relatório de impacto financeiro
        st.header("Impacto Financeiro")
        st.markdown("<hr style='border: 2px solid #008000'>", unsafe_allow_html=True)
        col1, col2 = st.columns([0.6,0.4], gap="large")
        with col1:
            resultados, figura_impacto = calcular_e_plotar_impacto(matrix_xgb, matrix_seq, valor_medio_emprest, taxa_juros)
            st.pyplot(figura_impacto, use_container_width=True)            

        with col2:
            st.markdown("<div style='font-size: 28px; font-weight: bold; color: #008000'>Relatório de Impacto Detalhado", unsafe_allow_html=True)
            st.write(resultados)
            diferenca_baseline = (resultados["Sequential"][2]) - (resultados["Baseline"][2])
            diferenca_xgb = (resultados["Sequential"][2]) - (resultados["XGB"][2])
            st.text(" ")
            st.success(f"Retorno do uso de Redes Neurais em relação ao atual cenário:\
                        R$ {diferenca_baseline:,.2f} ")
            st.success(f"Retorno do uso de Redes Neurais em relação ao Modelo XGB:\
                        R$ {diferenca_xgb:,.2f} ")
            st.markdown("<hr style='border: 2px solid #008000'>", unsafe_allow_html=True)
            st.markdown("<div style='font-size: 26px; font-weight: bold; color: #008000'>Descrição da visualização ", unsafe_allow_html=True)
            st.markdown("<div style='font-size: 16px; font-weight: sans serif'>O gráfico ao labo traz uma análise detalhada\
                        dos ganhos com bons pagadores, subtraindo-se as perdas com inadimplência e possível perda de clientes.\
                        Através dele é possível ter uma estimativa real dos possíveis\
                        retornos financeiros possíveis com o uso de Redes Neurais comparado a um modelo menos preciso e \
                        também ao cenário atual da empresa que não utiliza Inteligência Artificial na aprovação dos seus clientes (baseline).", unsafe_allow_html=True)                                            
     
    if visualizacao == "Redução da Inadimplência": # Gráfico e relatório de Inadimplência
        st.header("Redução da Inadimplência", anchor="red_inadimplencia")
        st.markdown("<hr style='border: 2px solid #2020df'>", unsafe_allow_html=True)
        col1, col2 = st.columns([0.6, 0.4], gap="medium")
        with col1:            
            figura_inad = plot_inadimplencia(resultado_xgb['inadimplencia_sem_modelos'], 
                                         resultado_xgb['inadimplencia_prevista'], resultado_seq['inadimplencia_prevista'])
            st.pyplot(figura_inad, use_container_width=True)
        
        with col2:
            st.markdown("<div style='font-size: 30px; font-weight: bold; color: #2020df'>Descrição da visualização", unsafe_allow_html=True)
            st.markdown("<div style='font-size: 20px; font-weight: bold'>Este gráfico compara três abordagens distintas e revela o poder dos modelos\
                         de inteligência artificial para transformar a gestão de risco e aumentar a rentabilidade nos negócios.", unsafe_allow_html=True)            
            
            st.markdown("<hr style='border: 2px solid #2020df'>", unsafe_allow_html=True) # Linha de separação

            st.markdown("<div style='font-size: 30px; font-weight: bold; color: #2020df'>Comparação de Cenários", unsafe_allow_html=True)
            st.markdown(f"<span style='font-size: 20px; font-weight: bold'>Taxa de Inadimplência Atual atinge alarmantes: \t </span>\
                      <span style='color: red; font-size: 25px; font-weight: bold'> {resultado_xgb['inadimplencia_sem_modelos']}%</span>", unsafe_allow_html=True)
            st.markdown(f"<span style='font-size: 20px; font-weight: bold'>Taxa de Inadimplência estimada com o uso do modelo mais\
                        eficaz (Redes Neurais) é de apenas: </span> <span style='color: green; font-size: 25px; font-weight: bold'>\
                        \t {resultado_seq['inadimplencia_prevista']}0%</span>", unsafe_allow_html=True)
            
            st.markdown("<hr style='border: 2px solid #2020df'>", unsafe_allow_html=True) # Linha de separação

            st.markdown("<div style='font-size: 30px; font-weight: bold; color: #2020df'>Impacto final na Inadimplência", unsafe_allow_html=True)
            st.markdown(f"<span style='font-size: 25px; font-weight: bold'>Redução total estimada na taxa de inadimplência com o uso de Redes Neurais:</span>\
                     <span style='font-size: 30px; font-weight: bold; color: green'>\
                         -{round(resultado_seq['inadimplencia_sem_modelos'] - resultado_seq['inadimplencia_prevista'], 2)}%</span>", unsafe_allow_html=True)


    elif visualizacao == "Captação de Bons Clientes": # Gráfico e relatório de captação de bons clientes
        st.header("Captação de Bons Clientes")
        st.markdown("<hr style='border: 2px solid #2020df'>", unsafe_allow_html=True)
        col1, col2 = st.columns([0.6, 0.4], gap="large")
        with col1:
            figura2 = plot_captacao_bons_clientes(resultado_xgb["captacao_bons_clientes"], resultado_seq["captacao_bons_clientes"])        
            st.pyplot(figura2, use_container_width=True)
        with col2:
            st.markdown("<div style='font-size: 30px; font-weight: bold; color: #2020df'>Descrição da visualização", unsafe_allow_html=True)
            st.markdown("<div style=' font-size: 20px; font-weight:bold'>🏆 O gráfico ao lado mostra a capacidade do modelo de identificar\
                        clientes de qualidade. Uma alta captação como a mostrada ao lado é imprescindível \
                        pois pode significar maior qualidade na aprovação e menos risco financeiro. ", unsafe_allow_html=True)            
            
            st.markdown("<hr style='border: 2px solid #2020df'>", unsafe_allow_html=True)
            st.markdown("<div style='font-size: 30px; font-weight: bold; color: #2020df'>Possível abordagem alternativa", unsafe_allow_html=True)
            st.markdown("<div style=' font-size: 20px; font-weight:bold'>🤖 XGBoost ainda é competitivo! Embora tenha ficado atrás," \
            " 88% de bons clientes captados ainda é um excelente desempenho, mostrando que árvores de decisão otimizadas\
                  continuam sendo uma alternativa sólida.", unsafe_allow_html=True)
            st.markdown("<hr style='border: 2px solid #2020df'>", unsafe_allow_html=True)

    
    elif visualizacao == "Taxa de Aprovação": # Gráfico e relatório de taxa de aprovação
        st.header("Taxa de Aprovação")
        st.markdown("<hr style='border: 2px solid #2020df'>", unsafe_allow_html=True)
        col1, col2 = st.columns([0.6,0.4], gap="large")
        with col1:            
            figura3 = plot_aprovacao_reprovacao(resultado_xgb["taxa_aprovacao"], resultado_xgb['taxa_reprovacao'],
                                                 resultado_seq['taxa_aprovacao'], resultado_seq['taxa_reprovacao'])
            st.pyplot(figura3, use_container_width=True)
        with col2:            
            st.markdown("<div style='font-size: 30px; font-weight: bold; color: #2020df'>Descrição da visualização", unsafe_allow_html=True)                      
            st.markdown("<div style=' font-size: 20px; font-weight:bold'>O gráfico ao lado mostra a taxa de aprovação \
                        de clientes. Note que Redes Neurais aprova menos clientes que XGBoost, mas \
                        ainda assim conseguiu captar uma maior quantidade de bons clientes e consequentemente reprovou\
                        clientes com alto risco de se tornar inadimplentes, o que traz grandes benefícios, não\
                        só na maior captação de recursos mas também na prevenção de perdas</div>", unsafe_allow_html=True )
            st.markdown("<hr style='border: 2px solid #2020df'>", unsafe_allow_html=True)
            st.markdown("<div style='font-size: 30px; font-weight: bold; color: #2020df'>Taxa de Aprovação dos Modelos", unsafe_allow_html=True)
            st.markdown(f"<div style=' font-size: 20px; font-weight:bold'>Redes Neurais aprovou {resultado_seq['taxa_aprovacao']}%\
                        dos clientes que solicitaram empréstimo enquanto XGB aprovou {resultado_xgb['taxa_aprovacao']}0% ", unsafe_allow_html=True)
                     
                       


