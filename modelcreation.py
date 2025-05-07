#ARQUIVO DE CRIAÇÃO DO MODELO E VERIFICAÇÃO DE MÉTRICAS

#importação das bibliotecas
from keras.models import Sequential
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from utilidades import *
from funcoes import *
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import numpy as np
from keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping
from keras import metrics


#Consulta no banco de dados
df = fetch_data_from_db(const.consulta_sql)

#Transformação de Tipos e Criação de novo atributo
df['idade'] = df['idade'].astype(int)
df['valortotalbem'] = df['valortotalbem'].astype(float)
df['valorsolicitado'] = df['valorsolicitado'].astype(float)
#df['proporcaosolicitadototal'] = df['valorsolicitado'] / df['valortotalbem']


lista = ['Advogado', 'Arquiteto', 'Cientista de Dados', 'Contador','Dentista','Empresário', 'Engenheiro','Médico','Programador'] #Profissões Válidas
colunas_categoricas = ['profissao', 'tiporesidencia',  'escolaridade','score','estadocivil','produto']  
colunas_numericas = ['tempoprofissao','renda','idade','dependentes','valorsolicitado','valortotalbem']


#Chamada das Funções para Tratamento dos dados
substitui_nulos(df)
tratar_outliers(df, 'idade', 0, 110)
tratar_outliers(df, 'tempoprofissao', 0, 70)
corrigir_erros_digitacao(df, 'profissao', lista)


#Separação da classe
X = df.drop('classe', axis = 1)
y = df['classe']



#Divisão em treino e teste
X_treino, X_teste, y_treino, y_teste = train_test_split(X, y, test_size = 0.3, random_state = 1235)


#Carregamento dos padronizadores
x_treino = save_scalers(X_treino,['tempoprofissao','renda','idade','dependentes','valorsolicitado','valortotalbem'] )
x_teste = save_scalers(X_teste,['tempoprofissao','renda','idade','dependentes','valorsolicitado','valortotalbem'] )

#Carregamento dos Codificadores
X_treino = save_encoders(X_treino, ['profissao', 'tiporesidencia',  'escolaridade','score','estadocivil','produto'])
X_teste = save_encoders(X_teste, ['profissao', 'tiporesidencia',  'escolaridade','score','estadocivil','produto'])

#Carregamento do Seletor de Atributos
seletor = RFE(RandomForestClassifier(n_estimators=500), n_features_to_select=6, step=1)
X_treino = seletor.fit(X_treino, y_treino).transform(X_treino)
X_teste = seletor.transform(X_teste)
joblib.dump(seletor, "objects\seletor.joblib")


#Codificação Manual (Ruim receberá 0 e Bom receberá 1)
mapeamento = {'ruim' : 0, 'bom' : 1}
y_treino = np.array([mapeamento[item] for item in y_treino])
y_teste = np.array([mapeamento[item] for item in y_teste])


#Empilhamento das camadas das redes neurais
model_seq = Sequential()
model_seq.add(Dense(50, activation = 'relu', input_dim = X_treino.shape[1]))
model_seq.add(Dropout(0.2))
model_seq.add(Dense(50, activation = 'relu'))
model_seq.add(Dropout(0.2))
model_seq.add(Dense(50, activation = 'relu'))
model_seq.add(Dropout(0.2))
model_seq.add(Dense(1, activation = 'sigmoid'))

model_seq.summary()

#Compilação e treinamento
model_seq.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = [metrics.AUC()])
early_stopping = EarlyStopping(monitor="val_loss", patience=50, restore_best_weights=True, mode="min")
model_seq.fit(X_treino, y_treino, epochs = 500, batch_size = 20, validation_data=(X_teste, y_teste), callbacks=[early_stopping])

model_seq.save("meu_modelo.keras")

previsoes = model_seq.predict(X_teste)

previsoes = (previsoes > 0.5).astype(int)


#Algumas métricas adicionais para conferir a acurácia do modelo nos dados de teste
acuracia = accuracy_score(y_teste, previsoes)
print(f'Acurácia do modelo nos dados de teste: {acuracia:.2f}')
np.save("objects/acuracia.npy", acuracia)


report = classification_report(y_teste, previsoes)
print(f'Outras métricas do modelo com os dados de teste: \n  {report}')

confusion = confusion_matrix(y_teste, previsoes)
print(f'Matriz de confusão: \n  {confusion} \n ')






