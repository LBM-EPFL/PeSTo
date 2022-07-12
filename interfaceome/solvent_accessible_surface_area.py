import h5py
import numpy as np
import pandas as pd
import mdtraj as md
from tqdm import tqdm
from multiprocessing import Pool

from src.structure import concatenate_chains
from structures_store import StructuresStoreDataset


def structure_to_mdtraj(structure):
    dfs = pd.DataFrame({
        'serial': np.arange(structure['xyz'].shape[0])+1,
        'name': structure['name'],
        'element': structure['element'],
        'resSeq': structure['resid'],
        'resName': structure['resname'],
        'chainID': structure['chain_name'],
        'segmentID': ['']*structure['xyz'].shape[0],
    })
    topo = md.Topology.from_dataframe(dfs)
    xyz = np.expand_dims(structure['xyz'], 0)*1e-1
    return md.Trajectory(xyz, topo)


def wrapper_solvent_accessible_surface_area(inp):
    subunits, key = inp
    structure = concatenate_chains(subunits)
    sasa = md.shrake_rupley(structure_to_mdtraj(structure))
    return sasa, key


def main():
    # parameters
    store_filepath = "datasets/alphafold_models.h5"
    output_filepath = "datasets/solvent_accessible_surface_area.h5"

    # create dataset
    dataset = StructuresStoreDataset(store_filepath)

    with h5py.File(output_filepath, 'w') as hf:
        keys = []
        # for subunits, key in tqdm(dataset):
        with Pool(processes=2) as pool:
            for sasa, key in tqdm(pool.imap(wrapper_solvent_accessible_surface_area, dataset), total=len(dataset)):
                # store data
                hf[key] = sasa.ravel().astype(np.string_)
                keys.append(key)

        # write all keys
        hf['metadata/keys'] = np.array(keys).astype(np.string_)


if __name__ == '__main__':
    main()
