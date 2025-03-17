#importação das bibliotecas
from keras.models import Sequential
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from utilidades import *
from funcoes import *
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import numpy as np
from keras.layers import Dense, Dropout
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix


#Consulta no banco de dados

df = fetch_data_from_db(const.consulta_sql)

df['idade'] = df['idade'].astype(int)
df['valortotalbem'] = df['valortotalbem'].astype(float)
df['valorsolicitado'] = df['valorsolicitado'].astype(float)
df['proporcaosolicitadototal'] = df['valorsolicitado'] / df['valortotalbem']
df['proporcaosolicitadototal'] = df['proporcaosolicitadototal'].astype(float)

lista = ['Advogado', 'Arquiteto', 'Cientista de Dados', 'Contador','Dentista','Empresário', 'Engenheiro','Médico','Programador']
colunas_categoricas = ['profissao', 'tiporesidencia',  'escolaridade','score','estadocivil','produto']
colunas_numericas = ['tempoprofissao','renda','idade','dependentes','valorsolicitado','valortotalbem','proporcaosolicitadototal']

#Chamada das funções para tratamento dos dados e divisão em treino e teste


substitui_nulos(df)

tratar_outliers(df, 'idade', 0, 110)
tratar_outliers(df, 'tempoprofissao', 0, 70)

corrigir_erros_digitacao(df, 'profissao', lista)


x = df.drop('classe', axis = 1)
y = df['classe']




x_treino, x_teste, y_treino, y_teste = train_test_split(x, y, test_size = 0.3, random_state = 1)

x_treino = save_scalers(x_treino,['tempoprofissao','renda','idade','dependentes','valorsolicitado','valortotalbem','proporcaosolicitadototal'] )
x_teste = save_scalers(x_teste,['tempoprofissao','renda','idade','dependentes','valorsolicitado','valortotalbem','proporcaosolicitadototal'] )

x_treino = save_encoders(x_treino, ['profissao', 'tiporesidencia',  'escolaridade','score','estadocivil','produto'])
x_teste = save_encoders(x_teste, ['profissao', 'tiporesidencia',  'escolaridade','score','estadocivil','produto'])



mapeamento = {'ruim' : 0, 'bom' : 1}
y_treino = np.array([mapeamento[item] for item in y_treino])
y_teste = np.array([mapeamento[item] for item in y_teste])

#Empilhamento das camadas das redes neurais

modelo = Sequential()

modelo.add(Dense(256, activation = 'relu', input_dim = x_treino.shape[1]))
modelo.add(Dropout(0.2))
modelo.add(Dense(128, activation = 'relu'))
modelo.add(Dropout(0.2))
modelo.add(Dense(10, activation = 'relu'))
modelo.add(Dropout(0.2))
modelo.add(Dense(1, activation = 'sigmoid'))


modelo.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])


modelo.fit(x_treino, y_treino, epochs = 500, batch_size = 20)

modelo.save('meu_modelo.keras')


previsoes = modelo.predict(x_teste)

previsoes = (previsoes > 0.5).astype(int)


#Algumas métricas para conferir a acurácia do modelo

accuracy = accuracy_score(y_teste, previsoes)
print(f'Acurácia do modelo nos dados de teste: {accuracy}')

report = classification_report(y_teste, previsoes)
print(f'Outras métricas do modelo com os dados de teste: \n  {report}')

confusion = confusion_matrix(y_teste, previsoes)
print(f'Matriz de confusão: \n  {confusion} \n ')






