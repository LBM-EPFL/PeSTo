import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.structure import atom_select
from structures_store import StructuresStoreDataset, h5_store_structure


def load_data_from_h5(h5_filepath, key):
    with h5py.File(h5_filepath, 'r') as hf:
        if key in hf.keys():
            data = np.array(hf[key])
        else:
            return None
    return data


def main():
    # parameters
    store_filepath = "datasets/alphafold_models.h5"
    predictions_filepath = "datasets/predicted_interfaces.h5"
    secondary_structure_filepath = "datasets/secondary_structures.h5"
    uniprot_info_filepath = "datasets/uniprot_localized_features.csv"

    # structures dataset
    structures_dataset = StructuresStoreDataset(store_filepath, with_preprocessing=False)

    # uniprot information
    df = pd.read_csv(uniprot_info_filepath)

    # parameters
    org_sel = "HUMAN"

    # filter out by selected organism
    m = np.array([key.split('/')[0] == org_sel for key in structures_dataset.keys])
    structures_dataset.keys = structures_dataset.keys[m]

    # aggregate data
    data = {}
    for structure, key in tqdm(structures_dataset):
        org, uniprot, mid = key.split('/')
        # if org == org_sel:
        if True:
            # get structure subset
            m = (structure['name'] == 'CA')
            structure_residues = atom_select(structure, m)

            # load residue information
            p = load_data_from_h5(predictions_filepath, key)
            if p is None:
                continue

            s = load_data_from_h5(secondary_structure_filepath, key).astype(np.dtype('U')).ravel()
            if s is None:
                continue

            # store structure data
            data[uniprot] = {
                'resid': structure_residues['resid'],
                'resname': structure_residues['resname'],
                'ss': s,
                'afs': structure_residues['bfactor'],
            }

            # add interface predictions
            for i in range(5):
                data[uniprot]['p{}'.format(i)] = p[:,i]

            # add coordinates
            for i in range(3):
                data[uniprot]['xyz'[i]] = structure['xyz'][m][:,i]

            # extract annotated regions
            dfs = df[df['NAME'] == uniprot]
            regions = []

            # add regions annotations
            resids = structure['resid'][m]
            annotations = [[] for _ in range(len(resids))]
            for k, row in dfs.iterrows():
                ids_region = np.arange(row['START'], row['END']+1)-1
                regions.append((ids_region, row['TYPE']))
                for k in np.where(np.isin(resids, ids_region))[0]:
                    annotations[k].append(row['TYPE'])
            data[uniprot]['annotation'] = np.array([':'.join(note) for note in annotations])

    with h5py.File("datasets/aggregated_structures_data.h5", 'w') as hf:
        for uniprot in tqdm(data):
            hgrp = hf.create_group(uniprot)
            h5_store_structure(hgrp, data[uniprot])


if __name__ == '__main__':
    main()
