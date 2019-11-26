import pandas as pd
import numpy as np
import keras as ks


def regresion(x):
    input = pd.DataFrame()
    input = x.iloc[:,0:20]
    output = pd.DataFrame()
    output = x.iloc[:,20]
    
    x_traind, y_traind = input[:10000], output[:10000]
    x_testd , y_testd = input[10000:], output[10000:]
    x_train , y_train = x_traind.values, y_traind.values
    x_test, y_test = x_testd.values, y_testd.values

    model = ks.models.Sequential()
    model.add(ks.layers.Dense(10,input_shape=(20,),activation='tanh'))
    model.add(ks.layers.Dense(10,activation='tanh'))
    model.add(ks.layers.Dense(10,activation='tanh'))
    model.add(ks.layers.Dense(10,activation='tanh'))
    model.add(ks.layers.Dense(10,activation='tanh'))
    model.add(ks.layers.Dense(10,activation='tanh'))
    model.add(ks.layers.Dense(10,activation='tanh'))
    model.add(ks.layers.Dense(10,activation='tanh'))
    model.add(ks.layers.Dense(10,activation='tanh'))
    model.add(ks.layers.Dense(10,activation='tanh'))
    model.add(ks.layers.Dense(1,activation='tanh'))
    model.compile(loss='mae', optimizer='sgd')

    print('Training -----------')
    for step in range(10001):
        cost = model.train_on_batch(x_train, y_train)
        if step % 100 == 0:
            print('train cost: ', cost)

    print('\nTesting ------------')
    cost = model.evaluate(x_test, y_test, batch_size=40)
    print('test cost:', cost)