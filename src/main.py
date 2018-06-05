#! /usr/bin/python
# -*- coding:utf-8 -*-

# Importações
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.layers import Conv2D, MaxPooling2D
from keras.callbacks import EarlyStopping
from keras import backend as K
from keras import utils

import numpy as np
import matplotlib.pyplot as plt

TRAINING_FILE = "../data/mnist/train.csv"
TEST_FILE = "../data/mnist/test.csv"


def carrega_dados_treinamento():
    fid = open(TRAINING_FILE,'r')
    lines = fid.readlines()
    fid.close()

    dataset = []

    for i in range(1024):
    	line = lines[i]
        dataset.append(line.rstrip('\n').split(','))

    dataset = np.array(dataset)

    # Divide o dataset em entradas (X) e saídas (Y)
    X = list()
    Y = list()

    for d in dataset:
    	
    	d_in = list(map(int, d[1:]))# coverte o vertor de strings para um vetor de inteiros
        X.append(d_in)
        y = int(d[0])
        labels = np.zeros(10)
        labels[y] = 1
#        Y.append(utils.to_categorical(y))
        Y.append(labels)		
	
    X = np.array(X)
    Y = np.array(Y)	

    img_rows, img_cols = 28, 28
        
    if K.image_data_format() == 'channels_first':
        X = X.reshape(X.shape[0], 1, img_rows, img_cols)
        input_shape = (1, img_rows, img_cols)
    else:
        X = X.reshape(X.shape[0], img_rows, img_cols,1)
        input_shape = (img_rows, img_cols,1)

    # Normalização
    X = X.astype('float32')
    X /= 255

    #Y = Y.astype('float32')

    return (X, Y, input_shape)


def carrega_dados_validacao():
    fid = open(TEST_FILE, 'r')
    lines = fid.readlines()
    fid.close()

    dataset = []
    
    for i in range(256):
    	line = lines[i]
        dataset.append(line.rstrip('\n').split(','))

    dataset = np.array(dataset)

    # Divide o dataset em entradas (X) e saídas (Y)
    X = list()
    Y = list()

    for d in dataset:
    
    	d_in = list(map(int, d[1:]))# coverte o vertor de strings para um vetor de inteiros
             
        X.append(d_in)
        y = int(d[0])
        labels = np.zeros(10)
        labels[y] = 1
#        Y.append(utils.to_categorical(y))
        Y.append(labels)		

#        y = int(d[0])
#        Y.append(utils.to_categorical(y))
		
    X = np.array(X)
    Y = np.array(Y)				

    img_rows, img_cols = 28, 28
        
    if K.image_data_format() == 'channels_first':
        X = X.reshape(X.shape[0], 1, img_rows, img_cols)
    else:
        X = X.reshape(X.shape[0], img_rows, img_cols,1)

    # Normalização
    X = X.astype('float32')
    X /= 255

    #Y = Y.astype('float32')

    return (X, Y)


#
# Train net
#  
def train_net():
    
    (X, Y, input_shape) = carrega_dados_treinamento()
    (X_val, Y_val) = carrega_dados_validacao()
   
    batch_size = 16
    epochs = 5 #mostra para a rede a quantidade de vezes a amostra sera apresentada para o treinamento

    # Cria o modelo
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),activation='relu',input_shape=input_shape))# a função de ativação balanceia os valores para o resultado de saida
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(65, kernel_initializer="uniform", activation='relu'))# camada oculta com 64 neuronios
    model.add(Dropout(0.2))
    model.add(Dense(65, kernel_initializer="uniform", activation='relu'))# camada oculta com 64 neuronios
    
    
    model.add(Dropout(0.2))
    
    model.add(Dense(units=10))
    model.add(Activation('softmax'))
	
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    # Treina o modelo
    history = model.fit(X, Y, nb_epoch=epochs, batch_size=batch_size,
        validation_data=(X_val, Y_val),
        callbacks=[EarlyStopping(monitor='val_loss', patience=10, verbose=1)])

    # Exporta o modelo
    #model.save('model.h5')

    print(history.history.keys())
#    print(history.history['acc'])

 #   plt.plot(history.history['acc'])
    plt.plot(history.history['acc'])

    plt.show()

    # Avalia o modelo
  #  scores = model.evaluate(X, Y)
  #  print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))



#
# Teste da rede com padrões desconhecidos
#  
def test_net(): 

    (X, Y) = carrega_dados_validacao()    

    model = load_model('model.h5')

    pred = model.predict(x=X, batch_size=1, verbose=0)

    n_correct = 0
    n_wrong = 0
    erro = 0 
    y_pred = pred
    print("Vetor y",Y[0])
    print("Vetor y_pred",pred[0])
    y_aux = np.array(Y) 
    pred_aux = np.array(pred)
    v_erro = y_aux - pred_aux

    erro = np.mean(np.square(v_erro))
    #v_erro = (v_erro ** 2) ** 0.5
    #erro = sum (sum (v_erro))

    
    print (" Valor do erro erro: ", erro)


if __name__ == "__main__":
    # Fixa o gerador de números aleatórios
    np.random.seed(10)

    #split_datasets()
    train_net()
    #test_net()


