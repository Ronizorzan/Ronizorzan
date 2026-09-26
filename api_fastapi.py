# === API Produção FastAPI ===
# === Importações Principais ===
import logging
from fastapi import FastAPI
from fastapi import status
from pydantic import BaseModel
import pandas as pd

from funcoes import load_scalers, load_encoders
from tensorflow.keras.models import load_model
from joblib import load
# === Fim das Importações ===


# === Configuração do Logging ===
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%d/%m/%Y %H:%M:%S",
    handlers=[logging.StreamHandler()]
)

# === Definição do Modelo de Dados ===
class CreditData(BaseModel):
    profissao: str
    tiporesidencia: str
    escolaridade: str
    score: str
    estadocivil: str
    produto: str
    tempoprofissao: float
    renda: float
    idade: int
    dependentes: int
    valorsolicitado: float
    valortotalbem: float


# === Funções Auxiliares ===
def load_unique_values():
    """
    Carrega os valores únicos a partir do dataframe final gerado.
    """
    df = pd.read_csv("dados_tratados.csv")
    return {column: df[column].unique().tolist() for column in df.columns}


def get_feature_selector():
    """
    Carrega o seletor de features treinado a partir do arquivo .joblib.
    Retorna o seletor carregado.
    """
    selector = load("objects/seletor.joblib")
    return selector


def get_model():
    """
    Carrega o modelo de classificação binária treinado a partir do arquivo .h5.
    Retorna o modelo carregado.
    """
    model = load_model("objects/meu_modelo.h5")
    return model


# Instancia o modelo e os valores únicos ao iniciar a aplicação -> para evitar recarregar o modelo a cada requisição
logging.info("Carregando o modelo e os valores únicos...")
model = get_model()
unique_values = load_unique_values()
selector = get_feature_selector()
logging.info("Modelo e valores únicos carregados com sucesso.")


# === Criação da instância do FastAPI ===
app = FastAPI(
    title="Credit Risk Prediction API",
    description="API para previsão de risco de crédito - modelo de classificação binária",
    version="1.0.0",
)


@app.post(
    "/predict",
    summary="Faz a previsão de risco de crédito com base nos dados fornecidos pelo usuário",
    status_code=status.HTTP_200_OK,
)
async def predict_credit_risk(data: CreditData):
    """
    Endpoint para fazer a previsão de risco de crédito com base nos dados fornecidos pelo usuário.
    Recebe um dicionário contendo os dados do cliente e retorna a previsão de risco de crédito.
    """
    # Ordem idêntica à usada no fit do seletor (RFE)
    feature_order = [
        "profissao",
        "tempoprofissao",
        "renda",
        "tiporesidencia",
        "escolaridade",
        "score",
        "idade",
        "dependentes",
        "estadocivil",
        "produto",
        "valorsolicitado",
        "valortotalbem",
    ]
    df = pd.DataFrame([data.model_dump()])[feature_order]
    df = load_scalers(
        df,
        [
            "tempoprofissao",
            "renda",
            "idade",
            "dependentes",
            "valorsolicitado",
            "valortotalbem",
        ],
    )
    df = load_encoders(
        df,
        ["profissao", "tiporesidencia", "escolaridade", "score", "estadocivil", "produto"],
    )
    df = selector.transform(df)
    proba = float(model.predict(df, verbose=0)[0][0])
    return {"prediction": int(proba >= 0.5), "probability": proba}


@app.get(
    "/unique-values",
    summary="Retorna os valores únicos de cada coluna do dataframe final",
    status_code=status.HTTP_200_OK,
)
async def get_unique_values_endpoint():
    """
    Endpoint para retornar os valores únicos de cada coluna do dataframe final.
    Retorna um dicionário contendo os valores únicos de cada coluna.
    """
    return unique_values


@app.get("/health", summary="Verifica a saúde e funcionamento da API", status_code=status.HTTP_200_OK)
def health_check() -> dict[str, str]:
    """
    Endpoint para verificar a saúde e funcionamento da API.
    Retorna uma mensagem indicando se a API está funcionando corretamente.
    """
    if not model:
        return {"status": "API is not available at the moment.", "reason": "Model is not loaded."}

    if not unique_values:
        return {"status": "API is not available at the moment.", "reason": "Unique values are not loaded."}

    if not selector:
        return {"status": "API is not available at the moment.", "reason": "Feature selector is not loaded."}

    return {
        "status": "API is healthy and functioning properly.",
        "reason": "All components are loaded and operational.",
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api_fastapi:app", host="0.0.0.0", port=8000, workers=2, reload=True)
