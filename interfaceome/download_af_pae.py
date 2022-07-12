import os
import json
import requests
import numpy as np
from glob import glob
from tqdm import tqdm
from multiprocessing import Pool


def wrapper_download_af_pae(pdb_filepath):
    _, org_dir, filename = pdb_filepath.split('/')
    key = filename.replace('-model_v2.pdb','')
    save_dir = os.path.join("alphafold_pae", org_dir)
    out_filepath = os.path.join(save_dir, "{}-predicted_aligned_error_v2.npy".format(key))
    if os.path.exists(out_filepath):
        return

    os.makedirs(save_dir, exist_ok=True)

    url = "https://alphafold.ebi.ac.uk/files/{}-predicted_aligned_error_v2.json".format(key)
    r = requests.get(url)

    if r.status_code == 200:
        data = json.loads(r.text)

        resids0 = np.array(data[0]['residue1'])-1
        resids1 = np.array(data[0]['residue2'])-1
        d = np.array(data[0]['distance'])

        D = np.zeros((np.max(resids0)+1, np.max(resids1)+1))
        D[resids0,resids1] = d

        np.save(out_filepath, D.astype(np.float16))
    else:
        print(key)


def main():
    pdb_filepaths = glob("alphafold_structures/**/*.pdb", recursive=True)

    # download data
    with Pool(processes=8) as pool:
        for _ in tqdm(pool.imap_unordered(wrapper_download_af_pae, pdb_filepaths), total=len(pdb_filepaths)):
            pass


if __name__ == '__main__':
    main()
