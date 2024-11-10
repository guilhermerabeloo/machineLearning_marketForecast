from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesClassifier
import pandas as pd

df = pd.read_csv('GSPC.csv')
df.head()

df = df.drop('Date', axis = 1) # Excluindo coluna Date. Desnecessaria para a analise

marketTomorrow = df[-1::] # Guardando ultima linha como valor da cotacao amanha, para testar a previsao

base = df.drop(df[-1::].index, axis = 0) # Preparando a base para treinamento; excluindo a linha de "amanha"

base['Target'] = base['Close'][1:len(base)].reset_index(drop=True) # Adicionando coluna Target, com o valor de fechamento do dia seguinte

marketToday = base[-1::].drop('Target', axis=1) # Guardando informacoes do dia atual para usar como base para teste e previsao do fechamento no dia amanha

training = base.drop(base[-1::].index, axis=0) # Removendo o dia de hoje e criando a base de treinamento

training.loc[training['Target'] > training['Close'], 'Target'] = 1 # Classificando dados que fazem o fechamento de amanha ser maior do que o fechamento de hoje
training.loc[training['Target'] != 1, 'Target'] = 0 # Classificando dados que fazem o fechamento de amanha ser menor do que o fechamento de hoje

y = training['Target'] # Criando variavel com o target para treino e teste
x = training.drop('Target', axis=1) # Criando variavel com os parametros para treino e teste    

x_training, x_test, y_training, y_test = train_test_split(x, y, test_size=0.3) # Separando dados de treino e teste
model = ExtraTreesClassifier()

model.fit(x_training, y_training) # Passando variaveis de treino para o modelo treinar

result = model.score(x_test, y_test) # Obtendo a acuracidade do modelo treinado
print(f'Accuracy: {result}')

marketToday['Target'] = model.predict(marketToday) # Gerando predicao de dados para prever o fechamento de amanha com base nos dados de hoje
print(marketToday)
print(marketTomorrow)
