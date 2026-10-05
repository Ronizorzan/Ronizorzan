#ARQUIVO DE CRIAÇÃO DO MODELO E VERIFICAÇÃO DE MÉTRICAS

#importação das bibliotecas
from keras.models import Sequential
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from utilidades import *
from funcoes import *
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import numpy as np
from keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping
from keras import metrics


N_SPLITS = 3
RANDOM_STATE = 1235


def build_model(input_dim: int) -> Sequential:
    """Constrói e compila a rede neural binária usada no projeto."""
    model = Sequential()
    model.add(Dense(50, activation="relu", input_dim=input_dim))
    model.add(Dropout(0.25))
    model.add(Dense(50, activation="relu"))
    model.add(Dropout(0.25))
    model.add(Dense(50, activation="relu"))
    model.add(Dropout(0.25))
    model.add(Dense(1, activation="sigmoid"))
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=[metrics.AUC(), metrics.Recall()])
    return model


#Consulta no banco de dados
try:
    df = fetch_data_from_db(const.consulta_sql)
except Exception as e:
    print(f"Erro ao consultar o banco de dados: {e}. Carregando últimos dados disponíveis...")
    df = pd.read_csv("dados_tratados.csv")    

#Transformação de Tipos e Criação de novo atributo
df["idade"] = df["idade"].astype(int)
df["valortotalbem"] = df["valortotalbem"].astype(float)
df["valorsolicitado"] = df["valorsolicitado"].astype(float)
df['proporcaosolicitadototal'] = df['valorsolicitado'] / df['valortotalbem']


lista = ["Advogado", "Arquiteto", "Cientista de Dados", "Contador", "Dentista", "Empresário", "Engenheiro", "Médico", "Programador"]  # Profissões Válidas
colunas_categoricas = ["profissao", "tiporesidencia", "escolaridade", "score", "estadocivil", "produto"]
colunas_numericas = ["tempoprofissao", "renda", "idade", "dependentes", "valorsolicitado", "valortotalbem", "proporcaosolicitadototal"]


#Chamada das Funções para Tratamento dos dados
substitui_nulos(df)
tratar_outliers(df, "idade", 0, 110)
tratar_outliers(df, "tempoprofissao", 0, 70)
corrigir_erros_digitacao(df, "profissao", lista)

df.to_csv("dados_tratados.csv", index=False)  # Salva os dados tratados em arquivo CSV para análises


#Separação da classe
X = df.drop("classe", axis=1)
y = df["classe"]


#Divisão em treino e teste (estratificada para preservar o balanceamento das classes)
X_treino, X_teste, y_treino, y_teste = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)


#Carregamento dos padronizadores
X_treino = save_scalers(X_treino, ["tempoprofissao", "renda", "idade", "dependentes", "valorsolicitado", "valortotalbem"])
X_teste = load_scalers(X_teste, ["tempoprofissao", "renda", "idade", "dependentes", "valorsolicitado", "valortotalbem"])

#Carregamento dos Codificadores
X_treino = save_encoders(X_treino, ["profissao", "tiporesidencia", "escolaridade", "score", "estadocivil", "produto"])
X_teste = load_encoders(X_teste, ["profissao", "tiporesidencia", "escolaridade", "score", "estadocivil", "produto"])

#Carregamento do Seletor de Atributos
seletor = RFE(RandomForestClassifier(n_estimators=500), n_features_to_select=6, step=1)
X_treino = seletor.fit(X_treino, y_treino).transform(X_treino)
X_teste = seletor.transform(X_teste)
joblib.dump(seletor, "objects/seletor.joblib")


#Codificação Manual (Ruim receberá 0 e Bom receberá 1)
mapeamento = {"ruim": 0, "bom": 1}
y_treino = np.array([mapeamento[item] for item in y_treino])
y_teste = np.array([mapeamento[item] for item in y_teste])


# === Cross-validation estratificada para métricas de generalização ===
cv_accuracies = []
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
print(f"\nIniciando cross-validation estratificada ({N_SPLITS} folds)...")
for fold, (idx_treino, idx_val) in enumerate(skf.split(X_treino, y_treino), start=1):
    X_fold_treino, X_fold_val = X_treino[idx_treino], X_treino[idx_val]
    y_fold_treino, y_fold_val = y_treino[idx_treino], y_treino[idx_val]

    early_stopping_cv = EarlyStopping(
        monitor="val_loss", patience=50, restore_best_weights=True, mode="min"
    )
    model_cv = build_model(input_dim=X_fold_treino.shape[1])
    model_cv.fit(
        X_fold_treino,
        y_fold_treino,
        epochs=500,
        batch_size=20,
        validation_data=(X_fold_val, y_fold_val),
        callbacks=[early_stopping_cv],
        verbose=0,
    )

    previsoes_fold = (model_cv.predict(X_fold_val, verbose=0) > 0.5).astype(int).ravel()
    acc_fold = accuracy_score(y_fold_val, previsoes_fold)
    cv_accuracies.append(acc_fold)
    print(f"  Fold {fold}/{N_SPLITS} — acurácia: {acc_fold:.4f}")

cv_accuracies = np.array(cv_accuracies)
acuracia = float(cv_accuracies.mean())
acuracia_std = float(cv_accuracies.std())

print(f"\nAcurácia média (CV estratificada): {acuracia:.4f} ± {acuracia_std:.4f}")
print(f"Acurácias por fold: {np.round(cv_accuracies, 4)}")
np.save("objects/acuracia.npy", acuracia)


#Empilhamento e treinamento do modelo final (todos os dados de treino)
model_seq = build_model(input_dim=X_treino.shape[1])
model_seq.summary()

early_stopping = EarlyStopping(
    monitor="val_loss", patience=50, restore_best_weights=True, mode="min"
)
model_seq.fit(
    X_treino,
    y_treino,
    epochs=500,
    batch_size=20,
    validation_data=(X_teste, y_teste),
    callbacks=[early_stopping],
)

model_seq.save("meu_modelo.keras")


#Métricas no holdout (complementares à CV)
previsoes = (model_seq.predict(X_teste, verbose=0) > 0.5).astype(int)
acuracia_holdout = accuracy_score(y_teste, previsoes)
print(f"\nAcurácia do modelo final nos dados de teste (holdout): {acuracia_holdout:.4f}")

report = classification_report(y_teste, previsoes)
print(f"Outras métricas do modelo com os dados de teste:\n{report}")

confusion = confusion_matrix(y_teste, previsoes)
print(f"Matriz de confusão:\n{confusion}\n")
