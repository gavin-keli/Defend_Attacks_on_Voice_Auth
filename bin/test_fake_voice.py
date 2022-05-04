##
##
##    Statistics Tesing Model Results 
##
##    True/False Positives/Negatives
##   
##    Modify ALL parameter below
##
##    Test outcome positive(Attacked) Actually condition positive(Attacked) ==> TP
##    Test outcome positive(Attacked) Actually condition negative(Normal) ==> FP
##    Test outcome negative(Normal) Actually condition positive(Attacked) ==> FN
##    Test outcome negative(Normal) Actually condition negative(Normal) ==> TN
##
##

import glob
import os
import sys
import time
import tensorflow as tf

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from collections import Counter
import guardian.constants as c
from guardian.utils import auto_stat_test_model, get_checkpoint_name_training, get_last_checkpoint_if_any

#### loading deep speaker model
from guardian.models import convolutional_model


def main(name_training,file_list,num_of_prediction):
    if num_of_prediction == 1:
        deep_speaker_ID = [1]
        times = 1
    elif num_of_prediction == 10:
        deep_speaker_ID = [1,2,3,4,5,6,7,8,9,10]
        times = 10
    elif num_of_prediction == 20:
        deep_speaker_ID = [1,2,3,4,5,6,7,8,9,10,1,2,3,4,5,6,7,8,9,10]
        times = 20
    else:
        print('NUM ERROR')

    folder = file_list[0:-1]
    file_list = (glob.glob(file_list))

    Test_T_P = 0
    Test_F_N = 0
    Test_T_N = 0
    Test_F_P = 0
    
    model_ID = name_training.split('-')[0]

    model1 = []
    for i in range(times):
        model = convolutional_model()
        last_checkpoint = get_last_checkpoint_if_any(c.CHECKPOINT_FOLDER_ARRAY[i])
        if last_checkpoint is not None:
            model.load_weights(last_checkpoint)
        model1.append(model)
    
    model2 = tf.keras.models.load_model(c.DISCRIMINATOR_MODEL+str(model_ID)+'.h5')
    model2_checkpoint = get_checkpoint_name_training(c.DISCRIMINATOR_CHECKPOINT_FOLDER, name_training)
    if model2_checkpoint is not None:
        model2.load_weights(model2_checkpoint)

    TF_list = []
    FT_list = []

    index = 0

    for i in file_list:
        if (index % 500) == 0:
            print(index)
        #after conbining two embeddings
        i = i.split("/")[-1]
        filename = i.split('/')[-1].split('-')[0]

        total_raw_result = 0
        try:

            if "fake_voice_" in i:
                for checkpoint_index in range(times):
                    #checkpoint -> a array length 10
                    raw_result, test_result = auto_stat_test_model(model1[checkpoint_index], model2, name_training, folder, i, checkpoint_index)

                    # 0.5/0.25
                    total_raw_result += raw_result

                    if checkpoint_index == times-1:

                        if total_raw_result < times * 0.5:
                            Test_F_N += 1
                            TF_list.append(filename)
                        else:
                            Test_T_P += 1
                        #print('attack', total_raw_result,raw_result_var,raw_result_list)
            else:
                for checkpoint_index in range(times):
                    raw_result, test_result = auto_stat_test_model(model1[checkpoint_index], model2, name_training, folder, i, checkpoint_index)

                    # 0.5/0.25
                    total_raw_result += raw_result

                    if checkpoint_index == times-1:
        
                        if total_raw_result < times * 0.5:
                            Test_T_N += 1
                        else:
                            Test_F_P += 1
                            FT_list.append(filename)
                        #print('normal', total_raw_result,raw_result_var,raw_result_list)
        except ValueError:
            continue

        index += 1

    return deep_speaker_ID, Test_T_P, Test_F_N, Test_T_N, Test_F_P, TF_list, FT_list



if __name__ == '__main__':
    #name_training = input('Please enter the name_training: ')
    name_training = '5651345512-100'
    file_list = "../data/audio/npy/*"
    num_of_prediction = 10                              # 1, 10, 20

    print('Training Model name is', name_training)
    print('The testing folder is', file_list)
    print('The number of prediction is', num_of_prediction)
    print('note', " ".join(c.CHECKPOINT_FOLDER_ARRAY))

    _, Test_T_P, Test_F_N, Test_T_N, Test_F_P, TF_list, FT_list = main(name_training,file_list,num_of_prediction)

    print("\nTest_T_P %s",Test_T_P)
    print("Test_F_N %s",Test_F_N)
    print("Test_T_N %s",Test_T_N)
    print("Test_F_P %s",Test_F_P)
    print("\n",Counter(TF_list))
    print(len(TF_list))
    print("\n",Counter(FT_list))
    print(len(FT_list))