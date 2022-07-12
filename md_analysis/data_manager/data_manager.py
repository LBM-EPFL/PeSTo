import os
import re
from glob import glob

from .iomanip import save_json, load_json, save_arr, load_arr


class DataManager:
    def __init__(self, db_root, safe=True):
        # define root path for database
        self.db_root = db_root
        self.safe = safe

        # read gstr
        with open(os.path.join(db_root, 'meta'), 'r') as fs:
            self.gstr = fs.read().rstrip('\n')

        # precompile regex
        rstr = self.gstr.replace('<', '(?P<').replace('>', '>.*)')
        self.p = re.compile(rstr)

        # extract key values
        self.keys = [g.lstrip('<').rstrip('>') for g in self.p.match(self.gstr).groups()]

    def define_path(self, path_info):
        # copy generator string
        path = self.gstr
        # iteratively construct path from key values
        for key in self.keys:
            path = path.replace('<{}>'.format(key), str(path_info[key]))

        return os.path.join(self.db_root, path)

    def define_filepath(self, path_info, filename):
        # define path
        path = self.define_path(path_info)
        # define full filepath
        return os.path.join(path, filename)

    def find_files(self, path, filename):
        return glob(os.path.join(path, '**', filename), recursive=True)

    def parse_path(self, path):
        m = self.p.match(path)
        vals = list(m.groups())

        vals[0] = vals[0].split('/')[-1]
        vals[-1] = vals[-1].split('/')[0]

        return {k:v for k, v in zip(self.keys, vals)}

    def insert_info(self, path_info, mod, **kwargs):
        # define path
        path = self.define_path(path_info)
        # define filepath
        info_filepath = os.path.join(path, mod+'_info.json')
        # if file already exists insert / modify info
        if os.path.exists(info_filepath):
            info_dict = load_json(info_filepath)
            for key in kwargs:
                if key in info_dict:
                    if self.safe:
                        print("Error: overwriting {}".format(key))
                        assert False
                info_dict[key] = kwargs[key]
            # save to json file
            save_json(info_filepath, info_dict)
        else:
            # save to json file
            save_json(info_filepath, kwargs)

    def find_info(self, path, mod):
        # locate signals data filepaths
        return glob(os.path.join(path, '**', mod+'_info.json'), recursive=True)

    def load_info(self, path, mod):
        # locate signals info filepaths
        info_filepaths = self.find_info(path, mod)

        # load signals info
        info_l = []
        for info_filepath in info_filepaths:
            key_dict = self.parse_path(info_filepath)
            info_dict = load_json(info_filepath)

            info_l.append({**key_dict, **info_dict})

        return info_l

    def update_info(self, path_info, mod, **kwargs):
        # define path
        path = self.define_path(path_info)
        # locate signals info filepaths
        info_filepaths = self.find_info(path, mod)

        # update each info files
        for info_filepath in info_filepaths:
            info_dict = load_json(info_filepath)
            for key in kwargs:
                info_dict[key] = kwargs[key]

            save_json(info_filepath, info_dict)

    def find_data(self, path, mod):
        # locate signals data filepaths
        return glob(os.path.join(path, '**', mod+'_data.npy'), recursive=True)

    def insert_data(self, path_info, mod, data):
        # define path
        path = self.define_path(path_info)
        # define filepath
        data_filepath = os.path.join(path, mod+'_data.npy')
        if os.path.exists(data_filepath):
            if self.safe:
                print("Error: {} already exists".format(data_filepath))
                assert False
        # save arr
        save_arr(data_filepath, data)

    def load_data(self, path, mod):
        # locate signals data filepaths
        data_filepaths = self.find_data(path, mod)

        # load all selected data
        data = []
        for data_filepath in data_filepaths:
            data.append(load_arr(data_filepath))

        return data
