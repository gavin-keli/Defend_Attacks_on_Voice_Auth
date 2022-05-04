'''

Save Guardian model

Modify structure, num_sample, note below

'''

import random
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import time    
import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

import guardian.constants as c

def discriminator_model(optimizer, loss, metrics, num_sample):
    ## Guardian
    inputs = keras.Input(shape=(32,32,1))
    # 2*2
    #x = layers.Conv2D(32, kernel_size=(2, 2), activation='relu')(inputs)
    # 4*4
    x = layers.Conv2D(32, kernel_size=(4, 4), activation='relu')(inputs)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Conv2D(64, kernel_size=(3, 3), activation='relu')(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    #x = layers.Dropout(0.4)(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation="relu")(x)
    #x = layers.Dropout(0.4)(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(32, activation="relu")(x)
    #x = layers.Dropout(0.4)(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)
    model = keras.Model(inputs=inputs, outputs=outputs, name="Guardian_model")
    
    '''
    ## FC
    inputs = keras.Input(shape=(1024,))
    x = layers.Dense(512, activation="relu")(inputs)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(32, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)
    model = keras.Model(inputs=inputs, outputs=outputs, name="FC_discriminator_model")
    '''

    # compile
    model.compile(
        optimizer = optimizer,
        loss = loss,
        metrics = metrics
        )
    num_layer = len(model.layers)
    return model, num_layer

if __name__ == '__main__':

    ###################################################### change detail here ################################################
    optimizer = tf.keras.optimizers.Adam()
    
    loss = tf.keras.losses.BinaryCrossentropy()
    metrics = ['accuracy']
    num_sample = 2
    #note = 'multi_classes_CNN_discriminator_model'
    note = 'CNN_discriminator_model'
    #note = 'FC'

    ###################################################### change detail here ################################################

    model, num_layer = discriminator_model(optimizer, loss, metrics, num_sample)
    name_model = random.randrange(1000000000,9999999999)
    print(name_model)
    now = time.strftime('%Y-%m-%d %H:%M:%S')
    model.save(c.DISCRIMINATOR_MODEL+str(name_model)+'.h5')
   

