import os
import importlib


def get_survae_path():
    init_path = importlib.util.find_spec("survae").origin
    path = os.path.dirname(os.path.dirname(init_path))
    return path


def get_data_path_file():
    path = get_survae_path()
    file = os.path.join(path, 'data_path')
    return file


def set_data_path(path):
    file = get_data_path_file()
    with open(file, 'w') as f:
        f.write(path)


def get_data_path():
    file = get_data_path_file()
    if not os.path.isfile(file):
        path = get_survae_path()
        default_path = os.path.join(path, 'data')
        set_data_path(default_path)
        os.mkdir(default_path)
    with open(file, 'r') as f:
        data_path = f.readline()
    return data_path


DATA_PATH = get_data_path()
