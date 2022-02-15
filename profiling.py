import os
import sys
import numpy as np
import torch as pt
import pandas as pd
from tqdm import tqdm
from glob import glob
from time import time

from src.dataset import StructuresDataset, collate_batch_features
from src.data_encoding import encode_structure, encode_features, extract_topology
from src.structure import concatenate_chains


# global parameters
#save_path = "model/save/i_v3_1_2021-05-28_12-40"  # 90
save_path = "model/save/i_v4_1_2021-09-07_11-21"  # 91
#model_filepath = os.path.join(save_path, 'model.pt')
model_filepath = os.path.join(save_path, 'model_ckpt.pt')

# add module to path
if save_path not in sys.path:
    sys.path.insert(0, save_path)

# load functions
from config import config_model, config_data
from data_handler import Dataset
from model import Model


def profiling():
    # parameters
    configs = [
        ("cuda", 16*1024),
        ("cpu", 2*1024),
    ]
    max_num_atoms = 8*1024
    min_num_atoms = 64

    # run profiling
    for device_name, N in configs:
        # define device
        device = pt.device(device_name)

        # create model
        model = Model(config_model)

        # reload model
        model.load_state_dict(pt.load(model_filepath, map_location=pt.device("cpu")))

        # set model to inference
        model = model.eval().to(device)

        # initialize dataset: load directly pdb files
        pdb_filepaths = glob("data/all_biounits/*/*.pdb*.gz")
        dataset = StructuresDataset(pdb_filepaths)

        # debug print
        print(len(dataset))

        # select data randomly
        ids = np.arange(len(dataset))
        np.random.shuffle(ids)

        # evaluate model computational performances
        profile = []
        with pt.no_grad():
            for i in tqdm(ids[:N]):
                # start time
                t0 = time()

                # load and preprocess pdb
                subunits, filepath = dataset[i]

                if subunits is not None:
                    if len(subunits) > 0:
                        # load time
                        t1 = time()

                        # concatenate all chains together
                        structure = concatenate_chains(subunits)

                        # do not break memory
                        if (structure['xyz'].shape[0] > max_num_atoms) or (structure['xyz'].shape[0] < min_num_atoms):
                            continue

                        # encode structure and features
                        X, M = encode_structure(structure, device=device)
                        # q = pt.cat(encode_features(structure, device=device), dim=1)
                        q = encode_features(structure, device=device)[0]

                        # extract topology
                        ids_topk, D_topk, R_topk, D, R = extract_topology(X, 64)

                        # pack data and setup sink (IMPORTANT)
                        X, ids_topk, q, M = collate_batch_features([[X, ids_topk, q, M]])

                        # processing time
                        t2 = time()

                        # run model
                        z = model(X, ids_topk, q, M.float())

                        # prediction
                        p = pt.sigmoid(z)

                        # end time
                        t3 = time()

                        # store profiling results
                        profile.append({
                            'num_atoms': M.shape[0],
                            'num_res': M.shape[1],
                            'load': t1-t0,
                            'process': t2-t1,
                            'run': t3-t2,
                            'total': t3-t0,
                            'pdbid': filepath.split('/')[-1].split('.')[0],
                        })

        # save profiling to csv
        pd.DataFrame(profile).to_csv("results/interface_ppi_{}_profiling.csv".format(device_name), index=False)


if __name__ == '__main__':
    profiling()
