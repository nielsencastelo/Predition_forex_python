# -*- coding: utf-8 -*-
""" recorrentes com dropout
Created on Wed Sep 25 11:00:19 2019

@author: Nielsen
"""

import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from sklearn.metrics import mean_absolute_error

base = pd.read_excel('XAUUSD_M5_201902080100_201902081430.xlsx')

base = base.dropna() # Apaga registros nulos

base = base.iloc[:,1].values # pega somente o open

plt.plot(base)

periodos = 30
previsao_futura = 1 # horizonte

X = base[0:(len(base) - (len(base) % periodos))]
X_batches = X.reshape(-1, periodos, 1)

y = base[1:(len(base) - (len(base) % periodos)) + previsao_futura]
y_batches = y.reshape(-1, periodos, 1)

X_teste = base[-(periodos + previsao_futura):]
X_teste = X_teste[:periodos]
X_teste = X_teste.reshape(-1, periodos, 1)
y_teste = base[-(periodos):]
y_teste = y_teste.reshape(-1, periodos, 1)

tf.reset_default_graph() # reset pra não ficar nada em memoria

entradas = 1
neuronios_oculta = 50
neuronios_saida = 1

xph = tf.placeholder(tf.float32, [None, periodos, entradas])
yph = tf.placeholder(tf.float32, [None, periodos, neuronios_saida])

def cria_uma_celula():
    return tf.contrib.rnn.LSTMCell(num_units = neuronios_oculta, activation = tf.nn.relu)

def cria_varias_celulas():
    celulas =  tf.nn.rnn_cell.MultiRNNCell([cria_uma_celula() for i in range(1)])
    return tf.contrib.rnn.DropoutWrapper(celulas, output_keep_prob = 0.1)

celula = cria_varias_celulas()
# camada saída
celula = tf.contrib.rnn.OutputProjectionWrapper(celula, output_size = 1)

saida_rnn, _ = tf.nn.dynamic_rnn(celula, xph, dtype = tf.float32)
erro = tf.losses.mean_squared_error(labels = yph, predictions = saida_rnn)
otimizador = tf.train.AdamOptimizer(learning_rate = 0.005)
treinamento = otimizador.minimize(erro)

max_epoca = 2000
custo = [0] * max_epoca 

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    for epoca in range(max_epoca):
        _, custo[epoca] = sess.run([treinamento, erro], feed_dict = {xph: X_batches, yph: y_batches})
        if epoca % 100 == 0:
            print(epoca + 1, ' erro: ', custo[epoca])
    
    previsoes = sess.run(saida_rnn, feed_dict = {xph: X_teste})
    
y_teste.shape
y_teste2 = np.ravel(y_teste) # diminuir a dimensão

previsoes2 = np.ravel(previsoes)

mae = mean_absolute_error(y_teste2, previsoes2)

plt.plot((y_teste2 - y_teste2.mean())/ y_teste2.std(), '', markersize = 10, label = 'Valor real')
plt.plot((previsoes2 - previsoes2.mean())/previsoes2.std(), '', label = 'Previsões')
plt.legend()




#plt.plot(y_teste2, label = 'Valor real')
#plt.plot(y_teste2, 'w*', markersize = 10, color = 'red')
#plt.plot(previsoes2, label = 'Previsões')
#plt.legend()