import os
import json
import pickle
import numpy as np
from datetime import datetime


def mkdir_path(path):
    if not os.path.exists(path):
        os.makedirs(path)


def check_ext(filepath, ext):
    if filepath[-len(ext)] != ext:
        return filepath + ext
    else:
        return filepath


def save_obj(filepath, obj):
    # check filepath
    filepath = check_ext(filepath, '.pkl')
    # create path if necessary
    mkdir_path(os.path.dirname(filepath))
    # save object
    with open(filepath, 'wb') as fs:
        pickle.dump(obj, fs)


def load_obj(filepath):
    # check filepath
    filepath = check_ext(filepath, '.pkl')
    # load object
    with open(filepath, 'rb') as fs:
        obj = pickle.load(fs)

    return obj


def save_json(filepath, obj):
    # create path if not existing
    mkdir_path(os.path.dirname(filepath))
    # add MODIFIED tag value to keep track of modification time
    obj['MODIFIED'] = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    # save json
    with open(filepath, 'w') as fs:
        json.dump(obj, fs)


def load_json(filepath):
    with open(filepath, 'r') as fs:
        obj = json.load(fs)

    return obj


def save_arr_csv(filepath, arr):
    # create path if not existing
    mkdir_path(os.path.dirname(filepath))
    # save array
    np.savetxt(filepath, arr, delimiter=',')


def load_arr_csv(filepath, arr):
    # load array
    return np.loadtxt(filepath, delimiter=',')


def save_arr(filepath, arr):
    # create path if not existing
    mkdir_path(os.path.dirname(filepath))
    # save array
    np.save(filepath, arr, allow_pickle=True)


def load_arr(filepath):
    # load array
    return np.load(filepath, allow_pickle=True)
