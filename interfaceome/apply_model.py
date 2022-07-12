import os
import sys
import h5py
import torch as pt
from tqdm import tqdm

from src.dataset import collate_batch_features
from src.data_encoding import encode_structure, encode_features, extract_topology
from src.structure import concatenate_chains
from structures_store import StructuresStoreDataset


def main():
    # data parameters
    h5_filepath = "datasets/alphafold_models.h5"
    output_filepath = "datasets/predicted_interfaces.h5"

    # model parameters
    # save_path = "model/save/i_v3_0_2021-05-27_14-27"  # 89
    # save_path = "model/save/i_v3_1_2021-05-28_12-40"  # 90
    # save_path = "model/save/i_v4_0_2021-09-07_11-20"  # 89
    save_path = "model/save/i_v4_1_2021-09-07_11-21"  # 91

    # select saved model
    model_filepath = os.path.join(save_path, 'model_ckpt.pt')
    #model_filepath = os.path.join(save_path, 'model.pt')

    # add module to path
    if save_path not in sys.path:
        sys.path.insert(0, save_path)

    # load functions
    from config import config_model
    from model import Model

    # define device
    device = pt.device("cuda")

    # create model
    model = Model(config_model)

    # reload model
    model.load_state_dict(pt.load(model_filepath, map_location=pt.device("cpu")))

    # set model to inference
    model = model.eval().to(device)

    # create dataset loader with preprocessing
    dataset = StructuresStoreDataset(h5_filepath, with_preprocessing=True)
    dataloader = pt.utils.data.DataLoader(dataset, batch_size=None, shuffle=True, num_workers=8, prefetch_factor=2)

    # run model on all subunits
    with h5py.File(output_filepath, 'w') as hf:
        with pt.no_grad():
            for subunits, key in tqdm(dataloader):
                # run model on structure
                try:
                    # concatenate all chains together
                    structure = concatenate_chains(subunits)

                    # encode structure and features
                    X, M = encode_structure(structure)
                    # q = pt.cat(encode_features(structure), dim=1)
                    q = encode_features(structure)[0]

                    # extract topology
                    ids_topk, D_topk, R_topk, D, R = extract_topology(X.to(device), 64)

                    # pack data and setup sink (IMPORTANT)
                    X, ids_topk, q, M = collate_batch_features([[X, ids_topk, q, M]])

                    # run model
                    z = model(X.to(device), ids_topk.to(device), q.to(device), M.float().to(device))

                    # prediction
                    p = pt.sigmoid(z)

                    # save results
                    hf[key] = p.cpu().numpy()

                except Exception as e:
                    print("error with {}: {}".format(key, e))


if __name__ == '__main__':
    main()
