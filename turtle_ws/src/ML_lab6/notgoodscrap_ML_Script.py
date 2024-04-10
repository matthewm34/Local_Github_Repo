# ME 7785 Lab 5
# Authors: Jeongwoo Cho, Matthew McKenna

# Author: Matthew McKenna
"""
Summary:
Create 2 generators that convert data from either OSL->PK or PK -> OSL
One generator uses speed labeled data and the other generator uses speed unlabeled data
After generators are built, the generated gait cycle is compared to the real gait cycle data for the 2 generators and MAE is computed
"""
import tensorflow as tf

from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
from PIL import Image
image = Image.open('opera_house.jpg')
print(image.format)
print(image.mode)
print(image.size)
# from tensorflow.keras.layers import Input, Conv1D, Conv1DTranspose, Dense, LeakyReLU, LSTM, Reshape, Flatten, BatchNormalization, Dropout
# from keras.layers import Input, Conv1D, Conv1DTranspose, Dense, LeakyReLU, LSTM, Reshape, Flatten, BatchNormalization, Dropout

# # from tensorflow.keras.models import Model
# from keras.models import Model
# import pickle
# import numpy as np
# from data_organize import plot_stacks, min_max_normalization, get_stack
# from sklearn.model_selection import train_test_split
# # from tensorflow.keras.models import load_model
# from keras.models import load_model

import sys
import csv
import time
import math
import numpy as np
import random
import matplotlib.pyplot as plt


### Load training images and labels

imageDirectory = '2023Fimgs/'

with open(imageDirectory + 'labels.txt', 'r') as f:
    reader = csv.reader(f)
    lines = list(reader)

#Randomly choose train and test data (50/50 split).
random.shuffle(lines)
train_lines = lines[:math.floor(len(lines)/2)][:]
test_lines = lines[math.floor(len(lines)/2):][:]

# this line reads in all images listed in the file in color, and resizes them to 25x33 pixels

train = np.array([np.array(cv2.resize(cv2.imread(imageDirectory+train_lines[i][0]+".jpg"),(25,33))) for i in range(len(train_lines))])








import matplotlib.pyplot as plt

def show_images(train_images,
            	class_names,
            	train_labels,
            	nb_samples = 12, nb_row = 4):
    
	plt.figure(figsize=(12, 12))
	for i in range(nb_samples):
    	plt.subplot(nb_row, nb_row, i + 1)
    	plt.xticks([])
    	plt.yticks([])
    	plt.grid(False)
    	plt.imshow(train_images[i], cmap=plt.cm.binary)
    	plt.xlabel(class_names[train_labels[i][0]])
	plt.show()



























# --------------------------------------------------Create the Paired and Unpaired data stacks For Analysis--------------------------------------------------


def build_generator():
    alpha = 0
    model = tf.keras.Sequential()
    model.add(Input(shape=(200, 4)))

    # Convolutional layers with BatchNormalization
    # Output 100,16
    model.add(Conv1D(16, kernel_size=2, strides=2, padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha))
    # model.add(BatchNormalization())

    # Output 50, 32
    model.add(Conv1D(32, kernel_size=2, strides=2, padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha))
    # model.add(BatchNormalization())

    # Output 25, 64
    model.add(Conv1D(64, kernel_size=2, strides=2, padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha))
    # model.add(BatchNormalization())

    # Output 50, 32
    # Convolutional Transpose layers with BatchNormalization
    model.add(Conv1DTranspose(32, kernel_size=2, strides=2, padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha))
    # model.add(BatchNormalization())
    
    # Output 100,16
    model.add(Conv1DTranspose(16, kernel_size=2, strides=2, padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha))
    # model.add(BatchNormalization())

    # Output 200, 4
    model.add(Conv1DTranspose(4, kernel_size=2, strides=2, padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha))
    # model.add(BatchNormalization())

    model.add(Flatten())
    model.add(Dense(200*4, activation='sigmoid'))
    model.add(Reshape((200, 4)))

    return model

generator = build_generator()
generator.compile(loss='mean_absolute_error', optimizer=tf.keras.optimizers.Adam(0.0001))


# with open('norm_stacks_v3.pkl', 'rb') as f:
#     norm_stacks = pickle.load(f)
    


# --------------------------------------------------TRAIN TEST SPLITTING--------------------------------------------------


    
real_train = []
noise_train = []
real_test = {}
real_test_array = []# matthew
noise_test = {}
noise_test_array = []# matthew
# ESSENTIALLY TRAIN TEST SPLITTING EACH OF THE 3 WALKING SPEED DATASETS, OUTPUT IS A GROUP OF TRAIN(for 7 speeds) AND TEST DATA(for 7 speeds) 
# Real Data is the OSL and noise data is the able bodied
# # If PK -> OSL
# if OSL_to_PK == True:

for speed in ['.3', '.4', '.5', '.6', '.7', '.8', '.9']:
    # Calculate the split index for 80%
    split_index = int(grouped_norm_stacks['PK'][speed].shape[0] * 0.8)

    # Extract 80% of real data
    real_data_part = grouped_norm_stacks['PK'][speed][:split_index]
    real_train.append(real_data_part)
    # real_train_labels += [label_encodings[speed]]*len(real_data_part) #[label_encodings[speed]] is list like [1] for medium, multiply by len (n) duplicates the element n times


    # Keep the remaining 20% in the dictionary
    real_test[speed] = grouped_norm_stacks['PK'][speed][split_index:]
    real_test_array.append(grouped_norm_stacks['PK'][speed][split_index:])

    # Similarly for noise data
    split_index = int(grouped_norm_stacks['OSL'][speed].shape[0] * 0.8)
    noise_data_part = grouped_norm_stacks['OSL'][speed][:split_index]
    noise_train.append(noise_data_part)
    noise_test[speed] = grouped_norm_stacks['OSL'][speed][split_index:]
    noise_test_array.append(grouped_norm_stacks['OSL'][speed][split_index:])


# If PK -> OSL
if OSL_to_PK == False:
    # swapping the noise and real data before training
    temp_real_train = real_train
    real_train = noise_train
    noise_train = temp_real_train

    temp_real_test = real_test
    real_test = noise_test
    noise_test = temp_real_test

    temp_real_test_array = real_test_array
    real_test_array = noise_test_array
    noise_test_array = temp_real_test_array


# Concatenate the 80% parts
real_train = np.concatenate(real_train, axis=0)
noise_train = np.concatenate(noise_train, axis=0)

real_test_array = np.concatenate(real_test_array, axis=0)
noise_test_array = np.concatenate(noise_test_array, axis=0)

# Shuffle
indices = np.arange(real_train.shape[0]) #array of indices from [0:end of data]
np.random.shuffle(indices) 
real_train = real_train[indices] #Shuffle the data so that the training data mixes the [.3-.9 speeds] but the real and noise are still paired
noise_train = noise_train[indices] 


# ----------------------------------------------Train Paired Generator (OSL -> PK) or (PK -> OSL) ----------------------------------------------
# Training The generator to create fake PK signals from an input of OSL "noise" 
g_loss_vec = []
'''
epochs = 250  # Number of epochs for training
batch_size = 64
num_batches = int(real_train.shape[0] / batch_size)  # Calculate the number of batches per epoch
for epoch in range(epochs):
    for batch in range(num_batches):
        start_idx = batch * batch_size
        end_idx = start_idx + batch_size

        real_samples = real_train[start_idx:end_idx,:,:] #shape 64,200,4
        noise_samples = noise_train[start_idx:end_idx]
        generated_samples = generator.predict(noise_samples)
    
        # Train the generator on noise samples
        g_loss = generator.train_on_batch(noise_samples, real_samples) # training the generator -> Input data: Noise & Output data: real_samples 
        g_loss_vec.append(g_loss)
    # Optionally, log losses or save models
    print(f"Epoch {epoch+1}/{epochs}, Generator Loss: {g_loss}")

    
    # Optionally, log losses or save models
    if epoch % 25 == 0 and OSL_to_PK == True:
        plot_stacks({'Generated':generated_samples, 'PK':real_samples,'OSL':noise_samples}, ['knee_theta','knee_thetadot','forceZ','shank_accelY'])
    elif epoch % 25 == 0 and OSL_to_PK == False :
        plot_stacks({'Generated':generated_samples, 'OSL':real_samples, 'PK':noise_samples}, ['knee_theta','knee_thetadot','forceZ','shank_accelY'])


#random plotting for report
epochs_plot = np.array([0,25,75,250]) 
count = 0
for i in list:
    plt.figure(i)   
    plt.xlabel('Gait Cycle %')
    plt.ylabel('Normalized Knee Theta')
    plt.title('OSL to PK: Paired Generated Knee Theta Signals vs OSL vs PK at ' + str(epochs_plot[count]) + ' Epochs')
    count = count + 1

# plotting epochs decay over time
plt.figure()
g_loss_vec = np.array(g_loss_vec)
plt.plot(np.arange(len(g_loss_vec)), g_loss_vec)
plt.xlabel('Epochs')
plt.ylabel('Model Loss: Mean Absolute Error (MAE)')

'''

if OSL_to_PK == True:
    # OSL -> PK
    # generator.save('paired_osl_to_pk_4_channel_250_epoch.h5')
    generator = load_model('paired_osl_to_pk_4_channel_250_epoch.h5')
else:
    # PK -> OSL
    # generator.save('paired_pk_to_osl_4_channel_250_epoch_v2.h5')
    generator = load_model('paired_pk_to_osl_4_channel_250_epoch_v2.h5')



# # ---------------------------------------------- Testing Generator Paired ----------------------------------------------
results_paired = {}
for speed in real_test.keys():
    results_paired[speed] = []
    for i in range(len(real_test[speed])):
        real = real_test[speed][i,:,:].reshape(1,200,4)
        noise = noise_test[speed][i,:,:].reshape(1,200,4)
        gen = generator.predict(noise)
        results_paired[speed].append((real, noise, gen))
        
        # plot_stacks({'Generated':gen, 'OSL':real,'AB':noise}, ['knee_theta','knee_thetadot','forceZ','shank_accelY'])



# ----------------------------------------------Train Unpaired Generator (OSL -> PK) or (PK -> OSL)----------------------------------------------
# Training The generator to create fake PK signals from an input of OSL "noise" 


generator = build_generator()
generator.compile(loss='mean_absolute_error', optimizer=tf.keras.optimizers.Adam(0.0001))

# Shuffle 
indices = np.arange(real_train.shape[0]) #array of indices from [0:end of data]
np.random.shuffle(indices) 
real_train = real_train[indices] #Shuffle the data so that the training data mixes the [.3-.9 speeds] but the real and noise are still paired

np.random.shuffle(indices)  # Shuffling the real_train differently than the noise_train to unpair the speeds and make random OSL to random PK stride 
noise_train = noise_train[indices] 

# UNPAIRED
# shuffle again
real_data = real_train
noise_data = noise_train
'''
epochs = 250  # Number of epochs for training
batch_size = 64
num_batches = int(real_train.shape[0] / batch_size)  # Calculate the number of batches per epoch
for epoch in range(epochs):
    np.random.shuffle(real_data)
    np.random.shuffle(noise_data)
    
    for batch in range(num_batches):
        start_idx = batch * batch_size
        end_idx = start_idx + batch_size

        real_samples = real_data[start_idx:end_idx,:,:]
        noise_samples = noise_data[start_idx:end_idx]
        generated_samples = generator.predict(noise_samples)
    
        # Train the generator on noise samples
        g_loss = generator.train_on_batch(noise_samples, real_samples)



    # Optionally, log losses or save models
    print(f"Epoch {epoch+1}/{epochs}, Generator Loss: {g_loss}")

    
    # Optionally, log losses or save models
    if epoch % 124 == 0 and OSL_to_PK == True:
        plot_stacks({'Generated':generated_samples, 'PK':real_samples,'OSL':noise_samples}, ['knee_theta','knee_thetadot','forceZ','shank_accelY'])
    elif epoch % 124 == 0 and OSL_to_PK == False :
        plot_stacks({'Generated':generated_samples, 'OSL':real_samples, 'PK':noise_samples}, ['knee_theta','knee_thetadot','forceZ','shank_accelY'])
    # print(f"Epoch {epoch}/{epochs} Discriminator Loss: {discriminator_loss} Generator Loss: {generator_loss}")

'''

if OSL_to_PK == True:
    # OSL -> PK
    # generator.save('unpaired_osl_to_pk_4_channel_250_epoch.h5')
    generator = load_model('unpaired_osl_to_pk_4_channel_250_epoch.h5')

else:
    # PK -> OSL
    # generator.save('unpaired_pk_to_osl_4_channel_250_epoch_v2.h5')
    generator = load_model('unpaired_pk_to_osl_4_channel_250_epoch_v2.h5')


# # ---------------------------------------------- Testing Generator Unpaired ----------------------------------------------
indices_noise = np.arange(noise_test_array.shape[0]) #shuffle the testing noise data in order to test on unpaired data 
np.random.shuffle(indices_noise) # noise_test_array is (num_strides, 200,4) np.array

cur_index = 0
count = 0
results_unpaired = {}
for speed in real_test.keys():
    results_unpaired[speed] = []
    for i in range(len(real_test[speed])):
        real = real_test[speed][i,:,:].reshape(1,200,4)
        noise = noise_test_array[indices_noise[count],:,:].reshape(1,200,4) # to test unpaired take a random noise signal
        gen = generator.predict(noise)
        results_unpaired[speed].append((real, noise, gen))
        count= count + 1
        # plot_stacks({'Generated':gen, 'OSL':real,'AB':noise}, ['knee_theta','knee_thetadot','forceZ','shank_accelY'])



# # ---------------------------------------------- Evaluate Generator Results for Both Paired and Unpaired (3/23/2024)----------------------------------------------
mae_arr_paired= []
mae_arr_unpaired= []
stack_avg_gen_signals_paired = []
stack_avg_gen_signals_unpaired = []
stack_avg_real_signals = []

stack_avg_signals = {}

dict_avg_gen_signals_paired = {}
dict_avg_gen_signals_unpaired = {} 
dict_avg_signals_real = {}


# plt.show(block=False) 
for speed in results_paired.keys():
    real_signals, gen_signals_paired, noise_signals_paired = [], [], []
    real_signals, gen_signals_unpaired, noise_signals_unpaired = [], [], []
    true_predictions = []

    stack_avg_signals[speed] = []

    for i in range(len(real_test[speed])):
        real_paired, noise_paired, gen_paired = results_paired[speed][i]
        real_unpaired, noise_unpaired, gen_unpaired = results_unpaired[speed][i]

        # real signals for paired and unpaired are same
        real_signal = real_unpaired #each of the real signal like ['knee_theta','knee_thetadot','forceZ','shank_accelY']
        real_signals.append(real_signal)
          
        # unpaired and paired generated signals
        gen_signal_unpaired = gen_unpaired #each of the gen signal like ['knee_theta','knee_thetadot','forceZ','shank_accelY']
        gen_signals_unpaired.append(gen_signal_unpaired)
        gen_signal_paired = gen_paired #each of the gen signal like ['knee_theta','knee_thetadot','forceZ','shank_accelY']
        gen_signals_paired.append(gen_signal_paired)


    # stack of 3 signals like 3,200,4 
    gen_signals_paired = np.concatenate(gen_signals_paired, axis=0) 
    gen_signals_unpaired = np.concatenate(gen_signals_unpaired, axis=0)
    real_signals = np.concatenate(real_signals, axis=0)
   
    avg_gen_signals_paired = np.average(gen_signals_paired,0) # average gen paired signal 200,4
    avg_gen_signals_unpaired = np.average(gen_signals_unpaired,0) # average unpaired gen signal 200,4
    avg_real_signals = np.average(real_signals,0) # average real signal 200,4

    dict_avg_gen_signals_paired[speed] = avg_gen_signals_paired
    dict_avg_gen_signals_unpaired[speed] = avg_gen_signals_unpaired
    dict_avg_signals_real[speed] = avg_real_signals

    stack_avg_gen_signals_paired.append(avg_gen_signals_paired.T) 
    stack_avg_gen_signals_unpaired.append(avg_gen_signals_unpaired.T) 
    stack_avg_real_signals.append(avg_real_signals.T)

    # MAE TABLE Calculate and Print
    mae_per_channel_paired = np.sum(np.absolute((avg_gen_signals_paired - avg_real_signals)),0) / 200 # mae of each of the average 4 channels in the form [1,4] size sum columns
    mae_per_channel_unpaired = np.sum(np.absolute((avg_gen_signals_unpaired - avg_real_signals)),0) / 200 # mae of each of the average 4 channels in the form [1,4] size sum columns
    mae_arr_paired.append(mae_per_channel_paired)
    mae_arr_unpaired.append(mae_per_channel_unpaired)
    # mae_per_speed = np.average(mae_per_channel)
    # mae_per_speed = np.average(mae_per_channel)
    # print(f'MAE: speed of ' + speed + ' = ' + str(np.round((mae_per_channel),4)) + '      Avg MAE: ' + str(np.round((mae_per_speed),4)))

stack_avg_signals

#Plots for all speeds average signal for each 4x sensor channel and each Domain(Real, Generated Paired, Generated Unpaired) 
stack_avg_gen_signals_paired = np.concatenate(stack_avg_gen_signals_paired, axis=1).T
stack_avg_gen_signals_unpaired = np.concatenate(stack_avg_gen_signals_unpaired, axis=1).T
stack_avg_real_signals = np.concatenate(stack_avg_real_signals, axis=1).T

all_signal_stack = np.array([stack_avg_gen_signals_paired,stack_avg_gen_signals_unpaired,stack_avg_real_signals])

dict_signal_stack = np.array([dict_avg_gen_signals_paired,dict_avg_gen_signals_unpaired,dict_avg_signals_real])

mae_arr_paired = np.concatenate(mae_arr_paired, axis=0).reshape(-1,4)
mae_arr_unpaired = np.concatenate(mae_arr_unpaired, axis=0).reshape(-1,4)

# Print in Table AVG MAE For Each Speed For Both Paired and Unpaired
# NOTE: rows are speed [.3 to .9] m/s and cols are ['knee_theta','knee_thetadot','forceZ','shank_accelY']
print('============== PRINT MAE TABLES ==============')
print('rows are speed [.3 to .9] m/s and cols are [knee_theta,knee_thetadot,forceZ,shank_accelY]')
print('')
print('AVG MAE PAIRED TABLE: \n'+ str(np.round((mae_arr_paired),5)))
print('')
print('AVG MAE UNPAIRED TABLE: \n' + str(np.round((mae_arr_unpaired),5)))
