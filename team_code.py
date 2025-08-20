#!/usr/bin/env python

# Edit this script to add your team's code. Some functions are *required*, but you can edit most parts of the required functions,
# change or remove non-required functions, and add your own functions.

################################################################################
#
# Optional libraries, functions, and variables. You can change or remove them.
#
################################################################################

import joblib
import numpy as np
import os
from sklearn import base
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import sys
import glob

from helper_code import *

################################################################################
#
# Required functions. Edit these functions to add your code, but do not change the arguments for the functions.
#
################################################################################

# Train your models. This function is *required*. You should edit this function to add your code, but do *not* change the arguments
# of this function. If you do not train one of the models, then you can return None for the model.

# Train your model.
def train_model(data_folder, model_folder, verbose):
    # Find the data files.
    if verbose:
        print('Finding the Challenge data...')

    #records = find_records(data_folder)
    records = glob.glob(os.path.join(data_folder, '**', 'one_beat.csv'), recursive=True)
    #print("Found records:", records)  # added this line

    num_records = len(records)

    if num_records == 0:
        raise FileNotFoundError('No data were provided.')

    # Extract the features and labels from the data.
    if verbose:
        print('Extracting features and labels from the data...')

   # Extract features and labels
    features = []
    labels = []
    for i in range(num_records):
        if verbose:
            width = len(str(num_records))
           #clear
           #  print(f'- {i+1:>{width}}/{num_records}: {records[i]}...')
        record = records[i]
        age, sex, source, signal_mean, signal_std = extract_features(record)
        label = load_label(record)
        #features.append(np.concatenate((age, sex, signal_mean, signal_std)))
        #features.append(np.concatenate((signal_mean, signal_std)))
        features.append(np.concatenate((signal_mean[:7], signal_std[:7])))
        labels.append(label)

    features = np.asarray(features, dtype=np.float32)
    labels = np.asarray(labels, dtype=bool)
    print("Training features shape:", features.shape)
    # Train the models on the features.
    if verbose:
        print('Training the model on the data...')

    # This very simple model trains a random forest model with very simple features.

    # Define the parameters for the random forest classifier and regressor.
    n_estimators = 12  # Number of trees in the forest.
    max_leaf_nodes = 34  # Maximum number of leaf nodes in each tree.
    random_state = 56  # Random state; set for reproducibility.

    # Fit the model.
    model = RandomForestClassifier(
        n_estimators=n_estimators, max_leaf_nodes=max_leaf_nodes, random_state=random_state).fit(features, labels)

    # Create a folder for the model if it does not already exist.
    os.makedirs(model_folder, exist_ok=True)

    # Save the model.
    save_model(model_folder, model)

    if verbose:
        print('Done.')
        print()

# Load your trained models. This function is *required*. You should edit this function to add your code, but do *not* change the
# arguments of this function. If you do not train one of the models, then you can return None for the model.
def load_model(model_folder, verbose):
    model_filename = os.path.join(model_folder, 'model_gpr.joblib')
    model = joblib.load(model_filename)
    return model

# Run your trained model. This function is *required*. You should edit this function to add your code, but do *not* change the
# arguments of this function.

def run_model(record, model, verbose):
    age, sex, source, signal_mean, signal_std = extract_features(record)
    features = np.concatenate((signal_mean[:7], signal_std[:7])).reshape(1, -1)
    features = features.astype(np.float64)
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

    # For GPR, predict returns a continuous value
    prediction = model.predict(features)[0]

    # You need to define a threshold to convert regression output to binary
    threshold = 0.5  # You may want to tune this value
    binary_output = int(prediction >= threshold)
    probability_output = float(prediction)  # Use regression output as "probability"

    return binary_output, probability_output

################################################################################
#
# Optional functions. You can change or remove these functions and/or add new functions.
#
################################################################################

# Extract your features.
def extract_features(record):
    header = load_header(record)

    # Extract the age from the record.
    age = get_age(header)
    age = np.array([age])

    # Extract the sex from the record and represent it as a one-hot encoded vector.
    sex = get_sex(header)
    sex_one_hot_encoding = np.zeros(3, dtype=bool)
    if sex is not None:
        if sex.casefold().startswith('f'):
            sex_one_hot_encoding[0] = 1
        elif sex.casefold().startswith('m'):
            sex_one_hot_encoding[1] = 1
        else:
            sex_one_hot_encoding[2] = 1
    else:
        sex_one_hot_encoding[2] = 1

    # Extract the source from the record (but do not use it as a feature).
    source = get_source(header)

    # Load the signal data directly from one_beat.csv in the same folder as the .hea file
    import pandas as pd
    base = os.path.basename(record)
    if not base.endswith('.csv'):
        base += '.csv'
    csv_path = os.path.join("data", base)
    #print("csv_path:", csv_path)
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"Signal file not found: {csv_path}")
    df = pd.read_csv(csv_path)
    signal = df.values.astype(np.float64)
    # If your signal is 1D, reshape to (N, 1)
    if signal.ndim == 1:
        signal = signal.reshape(-1, 1)

    # For compatibility, create dummy channel names
    reference_channels = ['I', 'II', 'III', 'AVR', 'AVL', 'AVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
    num_channels = len(reference_channels)
    # If your signal has fewer channels, pad with zeros
    if signal.shape[1] < num_channels:
        pad_width = num_channels - signal.shape[1]
        signal = np.pad(signal, ((0,0),(0,pad_width)), mode='constant', constant_values=0)

    # Compute two per-channel features as examples.
    signal_mean = np.zeros(num_channels)
    signal_std = np.zeros(num_channels)
    for i in range(num_channels):
        num_finite_samples = np.sum(np.isfinite(signal[:, i]))
        if num_finite_samples > 0:
            signal_mean[i] = np.nanmean(signal[:, i])
        else:
            signal_mean[i] = 0.0
        if num_finite_samples > 1:
            signal_std[i] = np.nanstd(signal[:, i])
        else:
            signal_std[i] = 0.0

    return age, sex_one_hot_encoding, source, signal_mean, signal_std
# Save your trained model.
def save_model(model_folder, model):
    #d = {'model': model}
    filename = os.path.join(model_folder, 'model_gpr.joblib')
    joblib.dump(model, filename, protocol=0)

   

