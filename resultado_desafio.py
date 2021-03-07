# -*- coding: utf-8 -*-
"""


@author: andre ribeiro de brito
"""

# Importação das bibliotecas
import pandas as pd
import cv2

from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, accuracy_score,classification_report
import numpy as np
import seaborn as sns
import statistics  as sts
from sklearn import svm
from sklearn import metrics
from sklearn.metrics import roc_curve, roc_auc_score,auc
from sklearn.linear_model import LogisticRegression

# Leitura do arquivo treino
# C:/Users/andre/OneDrive/Documentos/Beegol/treino.csv -> local do arquivo 
arq_treino = pd.read_csv("treino.csv",sep =",")
print(arq_treino.head())

# Verifica quais tabelas possui dados NaN
print(arq_treino.isnull().sum())

# Verifica a descrição de cada coluna
#print(arq_treino['loja'].describe())
#print(arq_treino['idade'].describe())
#print(arq_treino['sexo'].describe())
#print(arq_treino['flag_aposentado'].describe())
#print(arq_treino['flag_bolsa_familia'].describe())
#print(arq_treino['flag_emprego_formal'].describe())
#print(arq_treino['flag_socio'].describe())
#print(arq_treino['qtd_consultas'].describe())
#print(arq_treino['rating_1'].describe())
#print(arq_treino['rating_2'].describe())


# Substitui NaN da tabela 'sexo' pela letra 'f', baseado na moda
arq_treino['sexo'].fillna('f', inplace=True) # moda
arq_treino['sexo'].isnull().sum()

# Substitui NaN da tabela 'rating_1' pela letra 'E', baseado na moda
arq_treino['rating_1'].fillna('E', inplace=True) # moda
arq_treino['rating_1'].isnull().sum()

# Substitui NaN da tabela das tabelas pela mediana
mediana = sts.median(arq_treino['loja'])
arq_treino['loja'].fillna(mediana, inplace=True)
arq_treino['loja'].isnull().sum()

mediana = sts.median(arq_treino['idade'])
arq_treino['idade'].fillna(mediana, inplace=True)
arq_treino['idade'].isnull().sum()

mediana = sts.median(arq_treino['flag_aposentado'])
arq_treino['flag_aposentado'].fillna(mediana, inplace=True)
arq_treino['flag_aposentado'].isnull().sum()

mediana = sts.median(arq_treino['flag_bolsa_familia'])
arq_treino['flag_bolsa_familia'].fillna(mediana, inplace=True)
arq_treino['flag_bolsa_familia'].isnull().sum()

mediana = sts.median(arq_treino['flag_emprego_formal'])
arq_treino['flag_emprego_formal'].fillna(mediana, inplace=True)
arq_treino['flag_emprego_formal'].isnull().sum()

mediana = sts.median(arq_treino['flag_socio'])
arq_treino['flag_socio'].fillna(mediana, inplace=True)
arq_treino['flag_socio'].isnull().sum()

mediana = sts.median(arq_treino['qtd_consultas'])
arq_treino['qtd_consultas'].fillna(mediana, inplace=True)
arq_treino['qtd_consultas'].isnull().sum()

mediana = sts.median(arq_treino['rating_2'])
arq_treino['rating_2'].fillna(mediana, inplace=True)
arq_treino['rating_2'].isnull().sum()


# Indexador usado para resgatar os valores da coluna
previsores = arq_treino.iloc[:,0:13].values

# Transformar os dados de uma coluna especifica em codificação numerica
rotulocoluna2 = LabelEncoder()
previsores[:,2] = rotulocoluna2.fit_transform(previsores[:,2])

rotulocoluna5 = LabelEncoder()
previsores[:, 5] = rotulocoluna5.fit_transform(previsores[:, 5])

rotulocoluna11 = LabelEncoder()
previsores[:, 11] = rotulocoluna11.fit_transform(previsores[:, 11])

# Indexador usado para resgatar os valores da classe
classe = arq_treino.iloc[:,2].values
# Transformar os dados de uma coluna especifica em codificação numerica
rotulocolunaClasse = LabelEncoder()
classe = rotulocolunaClasse.fit_transform(classe)



# Leitura do arquivo teste
# C:/Users/andre/OneDrive/Documentos/Beegol/teste.csv -> local do arquivo 
arq_teste = pd.read_csv("teste.csv",sep =",")
print(arq_teste.head())

# Verifica quais tabelas possui dados NaN
print(arq_teste.isnull().sum())

# Verifica a descrição de cada coluna
#print(arq_teste['resposta'].describe())
#print(arq_teste['sexo'].describe())
#print(arq_teste['flag_aposentado'].describe())
#print(arq_teste['flag_bolsa_familia'].describe())
#print(arq_teste['flag_emprego_formal'].describe())
#print(arq_teste['flag_socio'].describe())
#print(arq_teste['qtd_consultas'].describe())
#print(arq_teste['rating_1'].describe())
#print(arq_teste['rating_2'].describe())

# Substitui NaN da tabela 'resposta' pela letra '0', baseado na moda
arq_teste['resposta'].fillna(' ', inplace=True) # moda
arq_teste['resposta'].isnull().sum()

# Substitui NaN da tabela 'sexo' pela letra 'f', baseado na moda
arq_teste['sexo'].fillna('f', inplace=True) # moda
arq_teste['sexo'].isnull().sum()

# Substitui NaN da tabela 'rating_1' pela letra 'E', baseado na moda
arq_teste['rating_1'].fillna('E', inplace=True) # moda
arq_teste['rating_1'].isnull().sum()

# Substitui NaN da tabela das tabelas pela mediana
mediana = sts.median(arq_teste['flag_aposentado'])
arq_teste['flag_aposentado'].fillna(mediana, inplace=True)
arq_teste['flag_aposentado'].isnull().sum()

mediana = sts.median(arq_teste['flag_bolsa_familia'])
arq_teste['flag_bolsa_familia'].fillna(mediana, inplace=True)
arq_teste['flag_bolsa_familia'].isnull().sum()

mediana = sts.median(arq_teste['flag_emprego_formal'])
arq_teste['flag_emprego_formal'].fillna(mediana, inplace=True)
arq_teste['flag_emprego_formal'].isnull().sum()

mediana = sts.median(arq_teste['flag_socio'])
arq_teste['flag_socio'].fillna(mediana, inplace=True)
arq_teste['flag_socio'].isnull().sum()

mediana = sts.median(arq_teste['qtd_consultas'])
arq_teste['qtd_consultas'].fillna(mediana, inplace=True)
arq_teste['qtd_consultas'].isnull().sum()


mediana = sts.median(arq_teste['rating_2'])
arq_teste['rating_2'].fillna(mediana, inplace=True)
arq_teste['rating_2'].isnull().sum()

# Indexador usado para resgatar os valores da coluna
previsores1 = arq_teste.iloc[:,0:13].values

# Transformar os dados de uma coluna especifica em codificação numerica
rotulocoluna2 = LabelEncoder()
previsores1[:,2] = rotulocoluna2.fit_transform(previsores1[:,2])

rotulocoluna5 = LabelEncoder()
previsores1[:, 5] = rotulocoluna5.fit_transform(previsores1[:, 5])

rotulocoluna11 = LabelEncoder()
previsores1[:, 11] = rotulocoluna11.fit_transform(previsores1[:, 11])

# Indexador usado para resgatar os valores da classe
classe1 = arq_teste.iloc[:,2].values

# Transformar os dados de uma coluna especifica em codificação numerica
rotulocolunaClasse1 = LabelEncoder()
classe1 = rotulocolunaClasse1.fit_transform(classe1)

# Classificador supervisionado Naive Bayes
naive_bayes = GaussianNB()
y_pred = naive_bayes.fit(previsores, classe).predict(previsores1)
print("Naive Bayes '0' -> Bom Pagador / '1' -> Mau Pagador  ",y_pred)
accuracia_bayesiana = metrics.accuracy_score(classe1, y_pred)
print("Acurácia Naive Bayes: ", accuracia_bayesiana)

# Matriz de confusão
#cm = confusion_matrix(classe1, y_pred)
#sns.heatmap(cm, annot=True, fmt="d")


# Classificador supervisionado SVM
class_svm = svm.LinearSVC()
class_svm.fit(previsores, classe)
y_pred = class_svm.predict(previsores1)
print("SVM '0' -> Bom Pagador / '1' -> Mau Pagador  ",y_pred)
accuracia_svm = metrics.accuracy_score(classe1, y_pred)
print("Acurácia SVM: ",accuracia_svm)


# Classificador supervisionado Regressão Logistica
logistica = LogisticRegression()
logistica.fit(previsores, classe)
y_pred = logistica.predict(previsores1)
print("Regressão Logistica '0' -> Bom Pagador / '1' -> Mau Pagador  ",y_pred)
y_pred_prob = logistica.predict_proba(previsores1)[:,1]
accuracia_regressao = metrics.accuracy_score(classe1, y_pred)
print("Acurácia regressao: ", accuracia_regressao)
print("Taxa de erro regressao: ", 1- accuracia_regressao )


# Erro ao realizar a curva ROC para o classificador Naive Bayes
#y_pred_prob =  naive_bayes.fit(previsores, classe).predict_proba(previsores1)[:, 1]
#y_test_scores_lr = [x for x in y_pred_prob]
#fpr, tpr, thresholds = roc_curve(classe1, y_test_scores_lr)


cv2.waitKey(0)
cv2.destroyAllWindows()


