import os
import re
import requests
from glob import glob
from tqdm import tqdm
from multiprocessing import Pool


def wrapper_download_from_uniprot(inp):
    # global variables
    base_url = "https://www.uniprot.org/uniprot/"
    formats = ['txt', 'xml', 'rdf', 'fasta', 'gff']
    org, uniprot = inp
    save_dir = os.path.join("uniprot", org, uniprot)
    os.makedirs(save_dir, exist_ok=True)
    for fmt in formats:
        # skip already downloaded
        filepath = os.path.join(save_dir, "{}.{}".format(uniprot, fmt))
        if os.path.exists(filepath):
            continue

        # download content
        url = "{}{}.{}".format(base_url, uniprot, fmt)
        r = requests.get(url)
        if r.status_code == 200:
            with open(filepath, 'w') as fs:
                fs.write(r.text)


def main():
    # find all alphafold models
    pdb_filepaths = glob("alphafold_structures/**/*.pdb", recursive=True)

    # get all uniprot ids
    info_l = []
    for pdb_filepath in pdb_filepaths:
        # extract identification key
        m = re.match(r'alphafold_structures/[A-Z0-9]*_[0-9]*_([A-Z0-9]*)_v2/AF-([A-Z0-9]*)-(F[0-9]*)-model_v2.pdb', pdb_filepath)
        org, uniprot, frag = m[1], m[2], m[3]
        if frag == 'F1':
            info_l.append((org, uniprot))

    # download data
    with Pool(processes=8) as pool:
        for _ in tqdm(pool.imap_unordered(wrapper_download_from_uniprot, info_l), total=len(info_l)):
            pass


if __name__ == '__main__':
    main()
