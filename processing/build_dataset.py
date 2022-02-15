import re
# import sys
import h5py
import numpy as np
import torch as pt
from glob import glob
from tqdm import tqdm

from src.structure import clean_structure, tag_hetatm_chains, split_by_chain, filter_non_atomic_subunits, remove_duplicate_tagged_subunits
from src.data_encoding import config_encoding, encode_structure, encode_features, extract_topology, extract_all_contacts
from src.dataset import StructuresDataset, save_data

pt.multiprocessing.set_sharing_strategy('file_system')


config_dataset = {
    # parameters
    "r_thr": 5.0,  # Angstroms
    "max_num_atoms": 1024*8,
    "max_num_nn": 64,
    "molecule_ids": np.array([
        'GLU', 'LEU', 'ALA', 'ASP', 'SER', 'VAL', 'GLY', 'THR', 'ARG', 'PHE', 'TYR', 'ILE',
        'PRO', 'ASN', 'LYS', 'GLN', 'HIS', 'TRP', 'MET', 'CYS', 'A', 'U', 'G', 'C', 'DA',
        'DT', 'DG', 'DC', 'MG', 'ZN', 'CL', 'CA', 'NA', 'MN', 'K', 'IOD', 'CD', 'CU', 'FE',
        'NI', 'SR', 'BR', 'CO', 'HG', 'SO4', 'NAG', 'PO4', 'EDO', 'ACT', 'MAN', 'HEM', 'FMT',
        'BMA', 'ADP', 'FAD', 'NAD', 'NO3', 'GLC', 'ATP', 'NAP', 'BGC', 'GDP', 'FUC', 'FES',
        'FMN', 'GAL', 'GTP', 'PLP', 'MLI', 'ANP', 'H4B', 'AMP', 'NDP', 'SAH', 'OXY', 'PLM',
        'CLR', 'CDL', 'RET'
    ]),

    # input filepaths
    "pdb_filepaths": glob("data/all_biounits/*/*.pdb[0-9]*.gz"),
    # "pdb_filepaths": glob(f"/tmp/{sys.argv[-1]}/all_biounits/*/*.pdb[0-9]*.gz"),

    # output filepath
    "dataset_filepath": "data/datasets/contacts_rr5A_64nn_8192_wat.h5",
    # "dataset_filepath": f"/tmp/{sys.argv[-1]}/contacts_rr5A_64nn_8192.h5",
}


def contacts_types(s0, M0, s1, M1, ids, molecule_ids, device=pt.device("cpu")):
    # molecule types for s0 and s1
    c0 = pt.from_numpy(s0['resname'].reshape(-1,1) == molecule_ids.reshape(1,-1)).to(device)
    c1 = pt.from_numpy(s1['resname'].reshape(-1,1) == molecule_ids.reshape(1,-1)).to(device)

    # categorize contacts
    H = (c1[ids[:,1]].unsqueeze(1) & c0[ids[:,0]].unsqueeze(2))

    # residue indices of contacts
    rids0 = pt.where(M0[ids[:,0]])[1]
    rids1 = pt.where(M1[ids[:,1]])[1]

    # create detailed contact map: automatically remove duplicated atom-atom to residue-residue contacts
    Y = pt.zeros((M0.shape[1], M1.shape[1], H.shape[1], H.shape[2]), device=device, dtype=pt.bool)
    Y[rids0, rids1] = H

    # define assembly type fingerprint matrix
    T = pt.any(pt.any(Y, dim=1), dim=0)

    return Y, T


def pack_structure_data(X, qe, qr, qn, M, ids_topk):
    return {
        'X': X.cpu().numpy().astype(np.float32),
        'ids_topk': ids_topk.cpu().numpy().astype(np.uint16),
        'qe':pt.stack(pt.where(qe > 0.5), dim=1).cpu().numpy().astype(np.uint16),
        'qr':pt.stack(pt.where(qr > 0.5), dim=1).cpu().numpy().astype(np.uint16),
        'qn':pt.stack(pt.where(qn > 0.5), dim=1).cpu().numpy().astype(np.uint16),
        'M':pt.stack(pt.where(M), dim=1).cpu().numpy().astype(np.uint16),
    }, {
        'qe_shape': qe.shape, 'qr_shape': qr.shape, 'qn_shape': qn.shape,
        'M_shape': M.shape,
    }


def pack_contacts_data(Y, T):
    return {
        'Y':pt.stack(pt.where(Y), dim=1).cpu().numpy().astype(np.uint16),
    }, {
        'Y_shape': Y.shape, 'ctype': T.cpu().numpy(),
    }


def pack_dataset_items(subunits, contacts, molecule_ids, max_num_nn, device=pt.device("cpu")):
    # prepare storage
    structures_data = {}
    contacts_data = {}

    # extract features and contacts for all subunits with contacts
    for cid0 in contacts:
        # get subunit
        s0 = subunits[cid0]

        # extract features, encode structure and compute topology
        qe0, qr0, qn0 = encode_features(s0)
        X0, M0 = encode_structure(s0, device=device)
        ids0_topk = extract_topology(X0, max_num_nn)[0]

        # store structure data
        structures_data[cid0] = pack_structure_data(X0, qe0, qr0, qn0, M0, ids0_topk)

        # prepare storage
        if cid0 not in contacts_data:
            contacts_data[cid0] = {}

        # for all contacting subunits
        for cid1 in contacts[cid0]:
            # prepare storage for swapped interface
            if cid1 not in contacts_data:
                contacts_data[cid1] = {}

            # if contacts not already computed
            if cid1 not in contacts_data[cid0]:
                # get contacting subunit
                s1 = subunits[cid1]

                # encode structure
                X1, M1 = encode_structure(s1, device=device)

                # nonzero not supported for array with more than I_MAX elements
                if (M0.shape[1] * M1.shape[1] * (molecule_ids.shape[0]**2)) > 2e9:
                    # compute interface targets
                    ctc_ids = contacts[cid0][cid1]['ids'].cpu()
                    Y, T = contacts_types(s0, M0.cpu(), s1, M1.cpu(), ctc_ids, molecule_ids, device=pt.device("cpu"))
                else:
                    # compute interface targets
                    ctc_ids = contacts[cid0][cid1]['ids'].to(device)
                    Y, T = contacts_types(s0, M0.to(device), s1, M1.to(device), ctc_ids, molecule_ids, device=device)

                # if has contacts of compatible type
                if pt.any(Y):
                    # store contacts data
                    contacts_data[cid0][cid1] = pack_contacts_data(Y, T)
                    contacts_data[cid1][cid0] = pack_contacts_data(Y.permute(1,0,3,2), T.transpose(0,1))

                # clear cuda cache
                pt.cuda.empty_cache()

    return structures_data, contacts_data


def store_dataset_items(hf, pdbid, bid, structures_data, contacts_data):
    # metadata storage
    metadata_l = []

    # for all subunits with contacts
    for cid0 in contacts_data:
        # define store key
        key = f"{pdbid.upper()[1:3]}/{pdbid.upper()}/{bid}/{cid0}"

        # save structure data
        hgrp = hf.create_group(f"data/structures/{key}")
        save_data(hgrp, attrs=structures_data[cid0][1], **structures_data[cid0][0])

        # for all contacting subunits
        for cid1 in contacts_data[cid0]:
            # define contacts store key
            ckey = f"{key}/{cid1}"

            # save contacts data
            hgrp = hf.create_group(f"data/contacts/{ckey}")
            save_data(hgrp, attrs=contacts_data[cid0][cid1][1], **contacts_data[cid0][cid1][0])

            # store metadata
            metadata_l.append({
                'key': key,
                'size': (np.max(structures_data[cid0][0]["M"], axis=0)+1).astype(int),
                'ckey': ckey,
                'ctype': contacts_data[cid0][cid1][1]["ctype"],
            })

    return metadata_l


if __name__ == "__main__":
    # set up dataset
    dataset = StructuresDataset(config_dataset['pdb_filepaths'], with_preprocessing=False)
    dataloader = pt.utils.data.DataLoader(dataset, batch_size=None, shuffle=True, num_workers=16, pin_memory=False, prefetch_factor=4)

    # define device
    device = pt.device("cuda")

    # process structure, compute features and write dataset
    with h5py.File(config_dataset['dataset_filepath'], 'w', libver='latest') as hf:
        # store dataset encoding
        for key in config_encoding:
            hf[f"metadata/{key}"] = config_encoding[key].astype(np.string_)

        # save contact type encoding
        hf["metadata/mids"] = config_dataset['molecule_ids'].astype(np.string_)

        # prepare and store all structures
        metadata_l = []
        pbar = tqdm(dataloader)
        for structure, pdb_filepath in pbar:
            # check that structure was loaded
            if structure is None:
                continue

            # parse filepath
            m = re.match(r'.*/([a-z0-9]*)\.pdb([0-9]*)\.gz', pdb_filepath)
            pdbid = m[1]
            bid = m[2]

            # check size
            if structure['xyz'].shape[0] >= config_dataset['max_num_atoms']:
                continue

            # process structure
            structure = clean_structure(structure)

            # update molecules chains
            structure = tag_hetatm_chains(structure)

            # split structure
            subunits = split_by_chain(structure)

            # remove non atomic structures
            subunits = filter_non_atomic_subunits(subunits)

            # check not monomer
            if len(subunits) < 2:
                continue

            # remove duplicated molecules and ions
            subunits = remove_duplicate_tagged_subunits(subunits)

            # extract all contacts from assembly
            contacts = extract_all_contacts(subunits, config_dataset['r_thr'], device=device)

            # check there are contacts
            if len(contacts) == 0:
                continue

            # pack dataset items
            structures_data, contacts_data = pack_dataset_items(
                subunits, contacts,
                config_dataset['molecule_ids'],
                config_dataset['max_num_nn'], device=device
            )

            # store data
            metadata = store_dataset_items(hf, pdbid, bid, structures_data, contacts_data)
            metadata_l.extend(metadata)

            # debug print
            pbar.set_description(f"{metadata_l[-1]['key']}: {metadata_l[-1]['size']}")

        # store metadata
        hf['metadata/keys'] = np.array([m['key'] for m in metadata_l]).astype(np.string_)
        hf['metadata/sizes'] = np.array([m['size'] for m in metadata_l])
        hf['metadata/ckeys'] = np.array([m['ckey'] for m in metadata_l]).astype(np.string_)
        hf['metadata/ctypes'] = np.stack(np.where(np.array([m['ctype'] for m in metadata_l])), axis=1).astype(np.uint32)
