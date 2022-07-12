import re
from glob import glob

from structures_store import store_pdbs


def main():
    pdb_filepaths = glob("alphafold_structures/**/*.pdb", recursive=True)

    pdb_key_filepaths = []
    for pdb_filepath in pdb_filepaths:
        # extract identification key
        m = re.match(r'alphafold_structures/[A-Z0-9]*_[0-9]*_([A-Z0-9]*)_v2/AF-([A-Z0-9]*)-(F[0-9]*)-model_v2.pdb', pdb_filepath)
        org, uniprot, frag = m[1], m[2], m[3]
        key = "{}/{}/{}".format(org, uniprot, frag)
        pdb_key_filepaths.append((key, pdb_filepath))

    store_pdbs("datasets/alphafold_models.h5", pdb_key_filepaths)


if __name__ == '__main__':
    main()
