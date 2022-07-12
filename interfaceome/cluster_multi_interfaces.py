import json
import h5py
import numpy as np
from tqdm import tqdm

from structures_store import h5_load_structure


def follow_rabbit(M, i):
    ids_checked = {i}
    ids_checking = set(np.where(M[i])[0])
    while ids_checking:
        for j in ids_checking.copy():
            ids_checking.remove(j)
            ids_checking.update(set([i for i in np.where(M[j])[0] if i not in ids_checked]))
            ids_checked.add(j)

    return list(ids_checked)


def follow_rabbits(M):
    i = 0
    ids_checked = []
    ids_clust = []
    while len(ids_checked) < M.shape[0]:
        ids_connect = follow_rabbit(M, i)
        ids_checked.extend(ids_connect)
        ids_clust.append(ids_connect)
        for j in range(i,M.shape[0]):
            if j not in ids_checked:
                i = j
                break

    return ids_clust


def cluster_interfaces(entry, afs_thr, p_thr, d_thr):
    labels = ["protein", "dna/rna", "ion", "ligand", "lipid"]
    ids_interfaces = {}
    for i in range(5):
        for j in range(i,5):
            # interface coordinates
            pi = entry['p{}'.format(i)]
            pj = entry['p{}'.format(j)]
            m = ((entry['afs'] > afs_thr) & (pi > p_thr) & (pj > p_thr))
            xyz_int = np.stack([entry['x'], entry['y'], entry['z']], axis=1)[m]

            # distance matrix
            D = np.sqrt(np.sum(np.square(np.expand_dims(xyz_int, 0) - np.expand_dims(xyz_int, 1)), axis=2))

            # follow the rabbits into the rabbit hole
            ids_ints = follow_rabbits(D < d_thr)

            # store results
            ids_p = np.where(m)[0]
            if i == j:
                key = labels[i]
            else:
                key = "{}+{}".format(labels[i],labels[j])
            ids_interfaces[key] = [[int(v) for v in ids_p[ids]] for ids in ids_ints]

    return ids_interfaces


def main():
    # parameters
    afs_thr = 70.0
    p_thr = 0.5
    d_thr = 10.0

    # structure data
    data = {}
    with h5py.File("datasets/aggregated_structures_data.h5", 'r') as hf:
        for uniprot in tqdm(hf.keys()):
            data[uniprot] = h5_load_structure(hf[uniprot])

    # cluster interfaces
    interfaces = {}
    for uniprot in tqdm(data):
        interfaces[uniprot] = cluster_interfaces(data[uniprot], afs_thr, p_thr, d_thr)

    # write output file
    json.dump(interfaces, open("datasets/clustered_multi_interfaces.json", 'w'))


if __name__ == '__main__':
    main()
