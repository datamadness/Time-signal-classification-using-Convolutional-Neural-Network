# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 21:09:59 2019

@author:    DATAmadness
Github:     https://github.com/datamadness
Blog:       ttps://datamadness.github.io
Description: Extracts Kaggle power transmission time domain discharge data for Train/Eval, 
calculates SFFT, pools the 2D matrix and saves the reduced data into TFRs
Includes upscaling for highly imbalanced classes
"""

import pandas as pd
import tensorflow as tf
import numpy as np
import os
import pyarrow.parquet as pq
import random
from scipy import signal
import matplotlib.pyplot as plt
from skimage.measure import block_reduce
import math
#%% PARAMETER SPECIFICATION
eval_fraction = 0.25 #fraction of data for evaluation purposes

#Specify location of the source and output data
data_path = 'G:\\powerLineData\\all\\'
output_path_train = 'G:\\powerLineData\\TFR_train_sfft\\'
output_path_eval = 'G:\\powerLineData\\TFR_eval_sfft\\'
os.listdir(data_path)

#Get the metadata for the entire training dataset
meta_train = pd.read_csv(data_path + 'metadata_train.csv')
measurement_id = np.array(meta_train['id_measurement'].unique())
#Randomly shuffle the measurement ids
np.random.shuffle(measurement_id)

#Split the id array into train and eval arrays
trainIDs, evalIDs = np.split(measurement_id,[int(len(measurement_id)*(1-eval_fraction))])

#Find how many disharges are in each measurement (int between 0 and 3) and create weight vector
trainWeights = np.zeros(len(trainIDs)).astype(int)

for i, ID in enumerate(trainIDs):
    trainWeights[i] = sum(meta_train['target'].loc[meta_train['id_measurement'] == ID])

#%% Functions
    
# Function that selects data of random measurement with a discharge(weighted selection with replacement based on trainWeights)
def select_random_discharge():
    ID = random.choices(trainIDs, weights = trainWeights, k = 1)[0]
    columns=[str(i) for i in range(ID*3,ID*3+3,1)]
    measurement = pq.read_pandas(data_path + 'train.parquet', columns).to_pandas()
    #Tag signal ids with 999 so we can recognize it is artificcialy made dataset
    measurement.columns = list(map(lambda x: '999' + x, measurement.columns.values))
    #Roll the original singnal data by random integer to alter them
    rolled_measurement = measurement.apply(lambda col: np.roll(col, shift = random.randint(1e3,7e5)), axis = 0)
    labels = meta_train['target'].iloc[columns]

    return ID, rolled_measurement, labels
    
# Function gets data of a specific measurement based on measurement_id
def get_measurement(ID):
    columns=[str(i) for i in range(ID*3,ID*3+3,1)]
    measurement = pq.read_pandas(data_path + 'train.parquet', columns).to_pandas()
    labels = meta_train['target'].iloc[columns]
    
    return ID, measurement, labels

# Function to parse a single record from the orig MNIST data into dict
def parser(signal, signal_ID, measurement_ID, label):
    parsed_data = {
            'signal': signal.flatten(order='C'),  #Makes it 50000 float32 1D vector
            'signal_ID': signal_ID,
            'measurement_ID': measurement_ID,
            'label': label
            }
    return parsed_data

# Create the example object with features
def get_tensor_object(single_record):
    
    tensor = tf.train.Example(features=tf.train.Features(feature={
        'signal': tf.train.Feature(
            float_list=tf.train.FloatList(value=single_record['signal'])),
        'signal_ID': tf.train.Feature(
            int64_list=tf.train.Int64List(value=[single_record['signal_ID']])),
        'measurement_ID': tf.train.Feature(
            int64_list=tf.train.Int64List(value=[single_record['measurement_ID']])),                
        'label': tf.train.Feature(
            int64_list=tf.train.Int64List(value=[single_record['label']]))
    }))
    return tensor

# Execute SFFT on phase signal data and reduce the resulting 2D matrix
def signal_sfft(phase_data,plot = False):
    
    fs = 40e6
    f, t, Zxx = signal.stft(phase_data, fs, nperseg=1999,boundary = None)
    
    reducedZ = block_reduce(np.abs(Zxx), block_size=(4, 4), func=np.max)

    reducedf = f[0::4]
    reducedt = t[0::4]
    
    if plot:
        plt.figure(figsize = (16, 10))
        plt.pcolormesh(reducedt, reducedf, reducedZ, rasterized=True, linewidth=0, vmin=0, vmax=0.5)
        plt.title('STFT Magnitude Reduced')
        plt.ylabel('Frequency [Hz]')
        plt.xlabel('Time [ms]')
        plt.show()
    return reducedZ

#%% Produce TFR files for Training (with upsampling and permutations)

def generate_TFR(IDs, measurements_per_file, upscale, output_path):

    #Artificaially generate every x measurement
    generate_every = 4
    
    numFiles = math.ceil(len(IDs) / measurements_per_file)
    
    for file_id in range(numFiles):
        print('\n Creating file # %2d' %file_id)
        with tf.python_io.TFRecordWriter(output_path + 'train_data_' + str(file_id) + '.tfrecord') as tfwriter:
            
            measurements_left = len(IDs) - file_id * measurements_per_file
            if measurements_left < measurements_per_file:
                iterations = measurements_left
            else:
                iterations = measurements_per_file
      
            # Iterate through all measurements
            for i in range(iterations):
                
                measurement = get_measurement(IDs[file_id * measurements_per_file + i])
                #measurement = select_random_discharge()
    
                def commit_record(measurement):
                    for j in range(3):
                        measurement_ID = measurement[0]
                        signal_ID = measurement[1].columns.values[j]
                        #float32 of 250*200
                        signal_data = signal_sfft(measurement[1][signal_ID].values)
                        label = measurement[2].iloc[j]
                        parsed_data = parser(signal_data, int(signal_ID), measurement_ID, label)
                        record_tensor = get_tensor_object(parsed_data)
                        # Append tensor data into tfrecord file
                        tfwriter.write(record_tensor.SerializeToString())
                        
                commit_record(measurement)
                
                if upscale and (((i + 1) % generate_every) == 0):
                    measurement = select_random_discharge()
                    commit_record(measurement)
#%% Execute functions
# Train IDs: List of mesurement IDs to export into TFR files
# Measurements_per_file: How many real measurements to save in each TFR file
# Upscale: True / False add artificially computed reords with discharges (improving class balance)
# Output path: folder path where to save the TFR files

#Generate TFR records for training data with upscaling
generate_TFR(trainIDs, measurements_per_file = 8, upscale = True, output_path = output_path_train)

#Generate TFR records for evaluation data - NO upscaling
generate_TFR(evalIDs, measurements_per_file = 8, upscale = False, output_path = output_path_eval)






