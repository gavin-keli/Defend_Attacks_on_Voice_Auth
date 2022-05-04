##
##
##   Training Guardian
##   
##   Modify ALL parameter below
##
##

import logging
import sys
import os
import tensorflow as tf

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

import guardian.constants as c
from guardian.utils import get_last_checkpoint_model_id, loading_embedding, FC_loading_embedding

def main(model_ID,epochs,batch_size,validation_split,validation_freq,embedding_folder):
    
    x, y, num_files = loading_embedding(embedding_folder)
    print('CNN model')

    #create model
    model = tf.keras.models.load_model(c.DISCRIMINATOR_MODEL+str(model_ID)+'.h5')
    grad_steps = 0
    last_checkpoint = get_last_checkpoint_model_id(c.DISCRIMINATOR_CHECKPOINT_FOLDER, model_ID)
    print(last_checkpoint)

    if last_checkpoint is not None:
        logging.info('Found checkpoint [{}]. Resume from here...'.format(last_checkpoint))
        model.load_weights(last_checkpoint)
        grad_steps = int(last_checkpoint.split('-')[-1].split('.')[0])
        logging.info('[DONE]')

    model.fit(x, y, batch_size=batch_size, epochs=epochs, validation_split=validation_split, validation_freq=validation_freq)
    grad_steps += epochs
    #save model
    model.save_weights('{0}/{1}-{2}.h5'.format(c.DISCRIMINATOR_CHECKPOINT_FOLDER, model_ID, grad_steps))
    name_training = str(model_ID)+"-"+str(grad_steps)
    #evaluate
    print("Evaluate on test data")
    results = model.evaluate(x, y, batch_size=batch_size)
    print("test loss, test acc:", results)

    return name_training, grad_steps, num_files

if __name__ == '__main__':
    validation_split = 0.2
    validation_freq = 10

    #model_ID = input('Please enter the model ID: ')
    model_ID = '5651345512'

    epochs = 50
    batch_size = 10

    deep_speaker_ID = "[1,2,3,4,5,6,7,8,9,10]"
    #deep_speaker_ID = "[1]"

    embedding_folder = '../data/audio/embedding/'

    print('model ID is', model_ID)
    print('The number of iteration is', epochs)
    print('The batch size is', batch_size)
    print('deep speaker ID is', deep_speaker_ID)
    print('embedding folder is', embedding_folder)

    main(model_ID,epochs,batch_size,validation_split,validation_freq,embedding_folder)
    
