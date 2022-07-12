import h5py
import numpy as np
import torch as pt
from tqdm import tqdm

from src.structure_io import read_pdb
from src.structure import clean_structure, tag_hetatm_chains, split_by_chain, filter_non_atomic_subunits, remove_duplicate_tagged_subunits


def h5_store_structure(hf, structure):
    for key in structure:
        data = structure[key]
        if '<U' in data.dtype.str:
            hf[key] = data.astype(np.string_)
        else:
            hf[key] = data


def h5_load_structure(hf):
    structure = {}
    for key in hf.keys():
        data = np.array(hf[key])
        if '|S' in data.dtype.str:
            structure[key] = data.astype('U')
        else:
            structure[key] = data

    return structure


class PDBStore:
    def __init__(self, h5_filepath):
        #self.hf = h5py.File(h5_filepath, 'r')
        self.h5_filepath = h5_filepath
        with h5py.File(h5_filepath, 'r') as hf:
            print(np.array(hf['metadata/keys']))
            self.keys = np.array(hf['metadata/keys']).astype(dtype=np.dtype('U'))

    def __iter__(self):
        return self.keys.__iter__()

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, key):
        with h5py.File(self.h5_filepath, 'r') as hf:
            return h5_load_structure(hf[key])


def store_pdbs(h5_filepath, pdb_key_filepaths):
    with h5py.File(h5_filepath, 'w') as hf:
        keys = []
        for key, pdb_filepath in tqdm(pdb_key_filepaths):
            # parse pdb file
            structure = read_pdb(pdb_filepath)

            # store data
            hgrp = hf.create_group(key)
            h5_store_structure(hgrp, structure)
            keys.append(key)

        # write all keys
        hf['metadata/keys'] = np.array(keys).astype(np.string_)


class StructuresStoreDataset(pt.utils.data.Dataset):
    def __init__(self, h5_filepath, with_preprocessing=True):
        super().__init__()
        # open store file and load keys
        self.hf = h5py.File(h5_filepath, 'r')
        self.keys = np.array(self.hf['metadata/keys']).astype(dtype=np.dtype('U'))

        # store flag
        self.with_preprocessing = with_preprocessing

    def __del__(self):
        self.hf.close()

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, i):
        # find key
        key = self.keys[i]

        # load structure
        structure = h5_load_structure(self.hf[key])

        if self.with_preprocessing:
            # process structure
            structure = clean_structure(structure)

            # update molecules chains
            structure = tag_hetatm_chains(structure)

            # split structure
            subunits = split_by_chain(structure)

            # remove non atomic structures
            subunits = filter_non_atomic_subunits(subunits)

            # remove duplicated molecules and ions
            subunits = remove_duplicate_tagged_subunits(subunits)

            return subunits, key
        else:
            return structure, key
