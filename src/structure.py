import numpy as np


# resname convergion (37)
res3to1 = {
    'CYS': 'C', 'ASP': 'D', 'SER': 'S', 'GLN': 'Q', 'LYS': 'K',
    'ILE': 'I', 'PRO': 'P', 'THR': 'T', 'PHE': 'F', 'ASN': 'N',
    'GLY': 'G', 'HIS': 'H', 'LEU': 'L', 'ARG': 'R', 'TRP': 'W',
    'ALA': 'A', 'VAL':'V', 'GLU': 'E', 'TYR': 'Y', 'MET': 'M'
}
res1to3 = {v:k for k,v in res3to1.items()}


def clean_structure(structure, rm_wat=True):
    # mask for water, hydrogens and deuterium
    m_wat = (structure["resname"] == "HOH")
    m_h = (structure["element"] == "H")
    m_d = (structure["element"] == "D")
    m_hwat = (structure["resname"] == "DOD")

    if rm_wat:
        # remove water
        mask = ((~m_wat) & (~m_h) & (~m_d) & (~m_hwat))
    else:
        # keep but tag water
        mask = ((~m_h) & (~m_d) & (~m_hwat))
        structure["resid"][m_wat] = -999

    # filter structure atoms
    structure = {key:structure[key][mask] for key in structure}

    # find changes due to chain
    chains = structure["chain_name"]
    ids_chains = np.where(np.array(chains).reshape(-1,1) == np.unique(chains).reshape(1,-1))[1]
    delta_chains = np.abs(np.sign(np.concatenate([[0], np.diff(ids_chains)])))

    # find changes due to inertion code
    icodes = structure["icode"]
    ids_icodes = np.where(np.array(icodes).reshape(-1,1) == np.unique(icodes).reshape(1,-1))[1]
    delta_icodes = np.abs(np.sign(np.concatenate([[0], np.diff(ids_icodes)])))

    # find changes due to resids
    resids = structure["resid"]
    delta_resids = np.abs(np.sign(np.concatenate([[0], np.diff(resids)])))

    # renumber resids
    resids = np.cumsum(np.sign(delta_chains + delta_resids + delta_icodes)) + 1

    # update resids
    structure['resid'] = resids

    # remove uncessary icodes
    structure.pop("icode")

    # return process structure
    return structure


def atom_select(structure, sel):
    return {key: structure[key][sel] for key in structure}


def split_by_chain(structure):
    # define storage
    chains = {}

    # define mask for chains
    cnames = structure["chain_name"]
    ucnames = np.unique(cnames)
    m_chains = (cnames.reshape(-1,1) == np.unique(cnames).reshape(1,-1))

    # find all interfaces in biounit
    for i in range(len(ucnames)):
        # get chain
        chain = atom_select(structure, m_chains[:,i])
        chain.pop("chain_name")
        # store chain data
        chains[ucnames[i]] = chain

    return chains


def concatenate_chains(chains):
    # get intersection of keys between chains
    keys = set.intersection(*[set(chains[cid]) for cid in chains])

    # concatenate chains
    structure = {key: np.concatenate([chains[cid][key] for cid in chains]) for key in keys}

    # add chain information
    structure['chain_name'] = np.concatenate([np.array([cid]*chains[cid]['xyz'].shape[0]) for cid in chains])

    return structure


def tag_hetatm_chains(structure):
    # get hetatm
    m_hetatm = (structure['het_flag'] == "H")
    resids_hetatm = structure['resid'][m_hetatm]

    # split if multiple hetatm
    delta_hetatm = np.cumsum(np.abs(np.sign(np.concatenate([[0], np.diff(resids_hetatm)]))))

    # update chain name
    cids_hetatm = np.array([f"{cid}:{hid}" for cid, hid in zip(structure['chain_name'][m_hetatm], delta_hetatm)])
    cids = structure['chain_name'].copy().astype(np.object)
    cids[m_hetatm] = cids_hetatm
    structure['chain_name'] = cids.astype(str)

    return structure


def remove_duplicate_tagged_subunits(subunits):
    # located tagged subunits
    tagged_cids = [cid for cid in subunits if (len(cid.split(':')) == 3)]
    # remove if overlapping
    for i in range(len(tagged_cids)):
        cid_i = tagged_cids[i]
        for j in range(i+1, len(tagged_cids)):
            cid_j = tagged_cids[j]

            # check if still existing
            if (cid_i in subunits) and (cid_j in subunits):
                # extract distances
                xyz0 = subunits[cid_i]['xyz']
                xyz1 = subunits[cid_j]['xyz']

                # if same size
                if xyz0.shape[0] == xyz1.shape[0]:
                    # minimum self distances
                    d_min = np.min(np.linalg.norm(xyz0 - xyz1, axis=1))
                    if d_min < 0.2:
                        subunits.pop(cid_j)

    return subunits


def filter_non_atomic_subunits(subunits):
    for sname in list(subunits):
        n_res = np.unique(subunits[sname]['resid']).shape[0]
        n_atm = subunits[sname]['xyz'].shape[0]

        if (n_atm == n_res) & (n_atm > 1):
            subunits.pop(sname)

    return subunits


def data_to_structure(X, q, M, std_elements, std_resnames, std_names):
    # resnames
    resnames_enum = np.concatenate([std_resnames, [b'UNX']])
    q_resnames = q[:,len(std_elements)+1:len(std_elements)+len(std_resnames)+2]
    resnames = resnames_enum[np.where(q_resnames)[1]]

    # resids
    ids0, ids1 = np.where(M > 0.5)
    resids = np.zeros(M.shape[0], dtype=np.int64)
    resids[ids0] = ids1+1

    # names
    q_names = q[:,len(std_elements)+len(std_resnames)+2:]
    names_enum = np.concatenate([std_names, [b'UNK']])
    names = names_enum[np.where(q_names)[1]]

    # elements
    q_elements = q[:,:len(std_elements)+1]
    elements_enum = np.concatenate([std_elements, [b'X']])
    elements = elements_enum[np.where(q_elements)[1]]

    # infer hetatm flag
    het_flags = np.array(['A']*len(resnames))
    het_flags[resnames == 'ZZZ'] = 'H'

    # pack subunit struct
    return {
        'xyz': X,
        'name': names,
        'element': elements,
        'resname': resnames,
        'resid': resids,
        'het_flag': het_flags,
    }


def encode_bfactor(structure, p):
    # C_alpha mask
    names = structure["name"]
    elements = structure["element"]
    het_flags = structure["het_flag"]
    m_ca = (names == "CA") & (elements == "C") & (het_flags == "A")
    resids = structure["resid"]

    if p.shape[0] == m_ca.shape[0]:
        structure['bfactor'] = p

    elif p.shape[0] == np.sum(m_ca):
        # expand c_alpha bfactor to all
        bf = np.zeros(len(resids), dtype=np.float32)
        for i in np.unique(resids):
            m_ri = (resids == i)
            i_rca = np.where(m_ri[m_ca])[0]
            if len(i_rca) > 0:
                bf[m_ri] = float(np.max(p[i_rca]))

        # store result
        structure['bfactor'] = bf

    elif p.shape[0] == np.unique(resids).shape[0]:
        # expand c_alpha bfactor to all
        uresids = np.unique(resids)
        bf = np.zeros(len(resids), dtype=np.float32)
        for i in uresids:
            m_ri = (resids == i)
            m_uri = (uresids == i)
            bf[m_ri] = float(np.max(p[m_uri]))

        # store result
        structure['bfactor'] = bf

    else:
        print("WARNING: bfactor not saved")

    return structure
