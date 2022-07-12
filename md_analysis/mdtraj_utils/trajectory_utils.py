import numpy as np
import scipy as sp
import torch as pt
import mdtraj as md


def join_trajectories(traj_list, selection="all"):
    # align trajectories
    ids_sim = align(traj_list[0][0], *traj_list[1:], selection=selection)

    # topology checks
    df_topo_ref = traj_list[0].topology.to_dataframe()[0].iloc[ids_sim[:,0]][['name', 'resName']]
    for k in range(1,len(traj_list)):
        assert np.all(df_topo_ref.values == traj_list[k].topology.to_dataframe()[0].iloc[ids_sim[:,k]][['name', 'resName']].values)

    # create new trajectory
    xyz = np.concatenate([traj_list[k].xyz[:,ids_sim[:,k],:] for k in range(len(traj_list))], axis=0)
    topology = traj_list[0].atom_slice(ids_sim[:,0]).topology

    return md.Trajectory(xyz, topology=topology)


def get_atoms_per_chain(traj, selection='all'):
    # define filter for atom type
    return [np.array([a.index for a in chain.atoms]) for chain in traj.topology.chains]


def unwrap_pbc(traj):
    # setup meshgrid for PBC repetitions
    dgrid = np.array([0.0, 1.0, -1.0])
    dX, dY, dZ = np.meshgrid(dgrid, dgrid, dgrid)
    dV = np.stack([dX.ravel(), dY.ravel(), dZ.ravel()], -1)

    # get indices of atoms for each molecules
    ids_mol_l = get_atoms_per_chain(traj)

    # compute center of mass of each molecule and its images
    pcm_rep_mol = np.zeros((len(ids_mol_l), 27, traj.xyz.shape[0], 3))
    for i in range(len(ids_mol_l)):
        # compute center of mass
        pcm = md.geometry.distance.compute_center_of_mass(traj.atom_slice(ids_mol_l[i]))

        # compute CM for all nearest periodic images
        for k in range(dV.shape[0]):
            pcm_rep_mol[i][k] = (pcm + traj.unitcell_lengths * dV[k].reshape(1,-1))

    # choose reference molecule with CM in reference cell
    pcm_ref = pcm_rep_mol[0][0]

    # make copy of trajectory
    traj_fix = traj[:]

    # for each other molecule
    for i in range(1,pcm_rep_mol.shape[0]):
        # compute distance of all images with reference molecule
        dcm_rep = np.sqrt(np.sum(np.square(pcm_rep_mol[i] - np.expand_dims(pcm_ref,0)), axis=2))

        # find molecule image closest to reference molecule
        ids_img = np.argmin(dcm_rep, axis=0)

        # update position of molecule
        traj_fix.xyz[:,ids_mol_l[i],:] += np.expand_dims(traj.unitcell_lengths * dV[ids_img],1)

    return traj_fix


def identify(top_a, top_b):
    # identify similar and mutated atoms pairs
    ids_sim_l = []
    ids_mut_l = []
    chain_a_used = set()
    chain_b_used = set()
    for chain_a in top_a.chains:
        # get number of residue of chain from molecule a
        n_res_a = len(chain_a._residues)
        for chain_b in top_b.chains:
            # get number of residue of chain from molecule b
            n_res_b = len(chain_b._residues)

            # length check
            if (n_res_a == n_res_b) and (chain_a.index not in chain_a_used) and (chain_b.index not in chain_b_used):
                # single residue chains (molecules, ions)
                if n_res_a == 1:
                    if list(chain_a.residues)[0].name.lower() != list(chain_b.residues)[0].name.lower():
                        continue

                # sequence check
                for res_a, res_b in zip(chain_a.residues, chain_b.residues):
                    # mutation warning
                    if res_a.name.lower() != res_b.name.lower():
                        # print("WARNING: [{}]{} != [{}]{}".format(chain_a.index, res_a, chain_b.index, res_b))
                        pass

                # get indices of matching residues
                for ra, rb in zip(chain_a.residues, chain_b.residues):
                    if ra.name.lower() == rb.name.lower():
                        # get all atoms of corresponding residues
                        ra_atoms = [a for a in ra.atoms]
                        rb_atoms = [b for b in rb.atoms]

                        # check that the two residues have the same number of atoms
                        if (len(ra_atoms) != len(rb_atoms)):
                            # if not same number of atoms -> nothing to do
                            # print("ERROR: different number of atoms for {}({}) : {}({})".format(ra, len(ra_atoms), rb, len(rb_atoms)))
                            pass
                        else:
                            # try to find unique ordering of atoms
                            a_names = [a.name for a in ra.atoms]
                            b_names = [b.name for b in rb.atoms]

                            # if not unique -> nothing to do
                            if ((len(a_names) != len(np.unique(a_names))) or (len(b_names) != len(np.unique(b_names)))):
                                # print("ERROR: non-unique atoms mismatch for {} : {}".format(ra, rb))
                                pass

                            elif np.all([a_name==b_name for a_name,b_name in zip(a_names,b_names)]):
                                for a, b in zip(ra.atoms, rb.atoms):
                                    ids_sim_l.append([a.index, b.index])

                            else:
                                # print("INFO: reordering atoms mismatch for {} : {}".format(ra, rb))

                                # find unique ordering
                                ids_reo_a = np.argsort(a_names)
                                ids_reo_b = np.argsort(b_names)

                                # get corresponding reordered atom indices
                                a_ids = np.array([a.index for a in ra.atoms])[ids_reo_a]
                                b_ids = np.array([b.index for b in rb.atoms])[ids_reo_b]

                                for ia, ib in zip(a_ids, b_ids):
                                    ids_sim_l.append([ia, ib])

                    else:
                        ids_mut_l.append(([a.index for a in ra.atoms], [b.index for b in rb.atoms]))

                # history chain used
                chain_a_used.add(chain_a.index)
                chain_b_used.add(chain_b.index)

    return np.array(ids_sim_l), ids_mut_l


def align(traj_ref, *trajs, selection="all"):
    # reference trajectory with selection
    ids_sel_ref = traj_ref.topology.select(selection)
    traj_sel_ref = traj_ref[0].atom_slice(ids_sel_ref)

    # for each input trajectory
    ids_sim_l = []
    for traj in trajs:
        # selected trajectory
        ids_sel = traj.topology.select(selection)
        traj_sel = traj[0].atom_slice(ids_sel)

        # identify to reference trajectory
        ids_sim_sel, _ = identify(traj_sel_ref.topology, traj_sel.topology)
        # get indices for input and not selected subset
        ids_sim_l.append(np.stack([ids_sel_ref[ids_sim_sel[:,0]], ids_sel[ids_sim_sel[:,1]]], axis=-1))

    # find common atoms between all trajectories
    ids_sim = ids_sim_l[0].copy()
    for k in range(1, len(ids_sim_l)):
        # intersection masks
        m0 = np.in1d(ids_sim[:,0], ids_sim_l[k][:,0])
        m1 = np.in1d(ids_sim_l[k][:,0], ids_sim[:,0])

        # filter previous indices and insert new indices
        ids_sim = np.concatenate([ids_sim[m0], ids_sim_l[k][m1,1].reshape(-1,1)], axis=1)

    return ids_sim


def center(traj):
    traj_c = traj[:]
    traj_c.xyz = (traj_c.xyz - np.expand_dims(np.mean(traj_c.xyz,axis=1),1))
    return traj_c


def rm_h(traj):
    return traj.atom_slice(traj.topology.select("not element H"))


def residue_to_atom_index_mapping(traj):
    resids = traj.topology.to_dataframe()[0]["resSeq"].values
    uresids = np.unique(resids)
    return np.isclose(uresids.reshape(-1,1), resids.reshape(1,-1))


def superpose_transform(xyz_ref, xyz):
    # copy data
    p = xyz.copy()
    p_ref = xyz_ref.copy()

    # centering
    t = np.expand_dims(np.mean(p,axis=1),1)
    t_ref = np.expand_dims(np.mean(p_ref,axis=1),1)

    # SVD decomposition
    U, S, Vt = np.linalg.svd(np.matmul(np.swapaxes(p_ref-t_ref,1,2), p-t))

    # reflection matrix
    Z = np.zeros(U.shape) + np.expand_dims(np.eye(U.shape[1], U.shape[2]),0)
    Z[:,-1,-1] = np.linalg.det(U) * np.linalg.det(Vt)

    R = np.matmul(np.swapaxes(Vt,1,2), np.matmul(Z, np.swapaxes(U,1,2)))
    return t, R, t_ref  # np.matmul(xyz - t, R) + t_ref


def superpose(traj_ref, *trajs, selection='name CA'):
    # identify same chains
    ids_sim = align(traj_ref, *trajs, selection=selection)

    # get reference positions
    xyz_ref = traj_ref.xyz[:,ids_sim[:,0],:]

    # align each input trajectory to reference
    traj_sup_l = []
    for k in range(len(trajs)):
        # get positions
        xyz = trajs[k].xyz[:,ids_sim[:,k+1],:]

        # compute the alignment transformation
        t, R, t_ref = superpose_transform(xyz_ref, xyz)

        # superpose trajectory to reference
        traj_sup_l.append(trajs[k][:])
        traj_sup_l[-1].xyz = np.matmul(traj_sup_l[-1].xyz-t, R) + t_ref

    return tuple(traj_sup_l + [ids_sim])


def atoms_to_residue_contacts(topology, ic_l, dc_l):
    # get all resids
    resids = np.array([a.residue.index for a in topology.atoms])

    # setup mapping between resids and atom id
    mr = np.isclose(resids.reshape(-1,1), np.unique(resids).reshape(1,-1))

    # find residue-residue contacts
    resids_int_l = []
    dmin_rr_l = []
    for k in range(len(ic_l)):
        if len(ic_l[k]) > 0:
            # get residue to atom at interface A and B
            resids_ia = np.where(mr[ic_l[k][:,0]])[1]
            resids_ib = np.where(mr[ic_l[k][:,1]])[1]

            # get unique residue-residue contacts
            resids_int, ids_inv = np.unique(np.stack([resids_ia, resids_ib], axis=1), return_inverse=True, axis=0)

            # find minimum distances for each residue-residue contact
            dmin_rr = np.zeros(resids_int.shape[0], dtype=np.float32)
            for i in np.unique(ids_inv):
                dmin_rr[i] = np.min(dc_l[k][np.where(ids_inv == i)[0]])
        else:
            resids_int = np.array([])
            dmin_rr = np.array([])

        # store data
        resids_int_l.append(resids_int)
        dmin_rr_l.append(dmin_rr)

    return resids_int_l, dmin_rr_l


def interface_residues_within(sub_a, sub_b, r_thr, *trajs, selection="not type H"):
    # get indices for atoms for subunit A and B in the reference trajectory
    ids_sim_a = align(sub_a, *trajs, selection=selection)
    ids_sim_b = align(sub_b, *trajs, selection=selection)

    # compute reference distance matrix
    D = pairwise_distance_matrix(trajs[0][0], ids_sim_a[:,1], ids_sim_b[:,1])[0]

    # get indices for distances within r_thr
    ids_ia, ids_ib = np.where(D <= r_thr)

    # for each trajectory, find atoms from residues at interface
    ids_ira_l = []
    ids_irb_l = []
    for k in range(len(trajs)):
        # get indices of atoms for interface of trajectory k
        ids_ia_k = ids_sim_a[ids_ia,k+1]
        ids_ib_k = ids_sim_b[ids_ib,k+1]

        # get all residue indices
        resids = np.array([a.residue.index for a in trajs[k].topology.atoms])

        # get atoms from residues at interface by adding all atoms of a residue with at least one atom within r_thr
        ids_ira = np.where(np.isclose(resids.reshape(-1,1), np.unique(resids[ids_ia_k]).reshape(1,-1)))[0]
        ids_irb = np.where(np.isclose(resids.reshape(-1,1), np.unique(resids[ids_ib_k]).reshape(1,-1)))[0]

        # store indices
        ids_ira_l.append(ids_ira)
        ids_irb_l.append(ids_irb)

    return np.stack(ids_ira_l, axis=-1), np.stack(ids_irb_l, axis=-1)


def pairwise_distance_matrix(traj, ids_a, ids_b):
    # compute distance matrix
    xyz = traj.xyz
    D = np.sqrt(np.sum(np.square(np.expand_dims(xyz[:, ids_a], 2) - np.expand_dims(xyz[:, ids_b], 1)), -1))*1e1

    return D


def rmsd(traj_ref, *trajs, selection="name CA"):
    # superpose trajectory to reference
    superpose_output = superpose(traj_ref, *trajs, selection=selection)
    trajs_sup = superpose_output[:-1]
    ids_sim_sel = superpose_output[-1]

    # compute rmsd for each trajectory
    rmsd_l = []
    for k in range(len(trajs)):
        # get atom position
        xyz = trajs_sup[k].xyz[:,ids_sim_sel[:,k+1],:]
        xyz_ref = traj_ref.xyz[:,ids_sim_sel[:,0],:].copy()
        #xyz_ref = (xyz_ref - np.expand_dims(np.mean(xyz_ref,axis=1),1))

        # compute rmsd
        rmsd_l.append(np.sqrt(np.mean(np.sum(np.square(xyz - xyz_ref), axis=2), axis=1))*1e1)

    return tuple(rmsd_l)


def irmsd(traj_ref, traj_R, traj_L, *trajs, r_thr=10.0):
    # determine interface of reference and corresponding other units based on subunits
    ids_ira, ids_irb = interface_residues_within(traj_R, traj_L, r_thr, traj_ref, *trajs, selection="not type H")

    # get full interface
    ids_int = np.concatenate([ids_ira, ids_irb], axis=0)
    traj_ref_int = traj_ref.atom_slice(np.sort(ids_int[:,0]))
    trajs_int = [trajs[k].atom_slice(np.sort(ids_int[:,k+1])) for k in range(len(trajs))]

    # compute rmsd from reference interface for each trajectory
    return rmsd(traj_ref_int, *trajs_int, selection="name CA")


def fnat(traj_ref, traj_R, traj_L, *trajs, r_thr=5.0):
    # get residues within r_thr of the interface
    ids_ira, ids_irb = interface_residues_within(traj_R, traj_L, r_thr, traj_ref, *trajs, selection="not type H")

    # add reference trajectory at the top of the list
    all_trajs = [traj_ref]+list(trajs)

    # compute fnat for each trajectory
    Rc_map_ref = None
    fnat_l = []
    for k in range(len(trajs)+1):
        # extract current trajectory and interface atoms indices
        traj = all_trajs[k]
        ids_a = ids_ira[:,k]
        ids_b = ids_irb[:,k]

        # get all resids
        resids = np.array([a.residue.index for a in traj.topology.atoms])

        # setup mapping between resids and atom id
        mr_a = np.isclose(resids[ids_a].reshape(-1,1), np.unique(resids[ids_a]).reshape(1,-1))
        mr_b = np.isclose(resids[ids_b].reshape(-1,1), np.unique(resids[ids_b]).reshape(1,-1))

        # number of frames, residues on subunit A and residues on subunit B
        N = traj.xyz.shape[0]
        Nr_a = mr_a.shape[1]
        Nr_b = mr_b.shape[1]

        # find residue-residue contacts for each frames, one pair of residues at a time
        Rc_map = np.zeros((N, Nr_a, Nr_b), dtype=bool)
        for i in range(Nr_a):
            for j in range(Nr_b):
                # get atoms indices for residue i of A and residue j of B
                ids_ri_a = ids_a[np.where(mr_a[:,i])[0]]
                ids_rj_b = ids_b[np.where(mr_b[:,j])[0]]
                # compute distance matrix for all frames between residues i of A and residue j of B
                D_ij = pairwise_distance_matrix(traj, ids_ri_a, ids_rj_b)
                # update residue-residue contacts map
                Rc_map[:,i,j] = np.any(np.any((D_ij < r_thr), axis=2), axis=1)

        # set reference contacts map or compute fnat
        if Rc_map_ref is None:
            Rc_map_ref = Rc_map.copy()
        else:
            # compute fraction of native contacts based on
            fnat = np.sum(np.sum((Rc_map & Rc_map_ref), axis=2), axis=1) / (np.sum(Rc_map_ref))
            # store result
            fnat_l.append(fnat)

    return tuple(fnat_l)


def contacts(sub_a, sub_b, traj, r_thr=5.0, selection="not type H", device_name="cuda"):
    # get indices for atoms for subunit A and B in the reference trajectory
    ids_sim_a = align(sub_a, traj, selection=selection)
    ids_sim_b = align(sub_b, traj, selection=selection)

    # get atom positions from subunit A and B
    xyz_a = traj.xyz[:,ids_sim_a[:,1],:]
    xyz_b = traj.xyz[:,ids_sim_b[:,1],:]

    # define device
    device = pt.device(device_name)
    # send data to device
    xyz_a = pt.from_numpy(xyz_a.astype(np.float32)).to(device)
    xyz_b = pt.from_numpy(xyz_b.astype(np.float32)).to(device)

    # for each frame
    contacts_l = []
    for k in range(traj.xyz.shape[0]):
        # compute distance matrix
        D = pt.sqrt(pt.sum(pt.pow(xyz_a[k].unsqueeze(1) - xyz_b[k].unsqueeze(0), 2), axis=2))*1e1

        # get indices of atoms bellow threshold distance
        ids_ia, ids_ib = pt.where(D < r_thr)

        # send data back to cpu and to numpy array
        d = D[ids_ia, ids_ib].cpu().numpy().astype(np.float32)
        ica = ids_sim_a[ids_ia.cpu().numpy(),1].astype(np.int32)
        icb = ids_sim_b[ids_ib.cpu().numpy(),1].astype(np.int32)

        # store data
        contacts_l.append([d, np.stack([ica, icb], axis=1)])

    return contacts_l


def sasa(traj):
    # define number of frames and atoms
    N = traj.xyz.shape[0]
    M = traj.xyz.shape[1]

    # compute solvent accessible surface area for each frame for each atom
    sasa = np.zeros((N,M), dtype=np.float32)
    for k in range(traj.xyz.shape[0]):
        sasa[k] = md.shrake_rupley(traj[k]).ravel()

    return sasa


def hydrogen_bonds(sub_R, sub_L, traj):
    # get number of frames
    N = traj.xyz.shape[0]
    nhb = np.zeros(N)
    ids_ihb_l = []
    for k in range(N):
        # get all hydrogen bnds
        ids_hb = md.baker_hubbard(traj[k], periodic=False)

        # get indices of receptor and ligand
        ids_R = align(traj, sub_R)[:,0]
        ids_L = align(traj, sub_L)[:,0]

        # get hydrogen bonds donors of both subunits
        ids_hb_L = ids_hb[np.isin(ids_hb[:,0], ids_L)]
        ids_hb_R = ids_hb[np.isin(ids_hb[:,0], ids_R)]

        # get hydrogen bonds acceptors on corresponding subunit
        if len(ids_hb_L) > 0:
            ids_hb_LR = ids_hb_L[np.isin(ids_hb_L[:,2], ids_R)]
        if len(ids_hb_R) > 0:
            ids_hb_RL = ids_hb_R[np.isin(ids_hb_R[:,2], ids_L)]

        # pack all hydrogen bonds between subunits
        ids_ihb = np.concatenate([ids_hb_LR, ids_hb_RL])

        # store results
        nhb[k] = ids_ihb.shape[0]
        ids_ihb_l.append(ids_ihb)

    return nhb, ids_ihb_l


def interface_rigid_docking(sub_R, sub_L, traj_ref, traj, r_thr=10.0):
    # get interface residues within r_thr from the reference structure
    ids_irR, ids_irL = interface_residues_within(sub_R, sub_L, r_thr, traj_ref, traj)

    # center reference using alignment from R to C:R
    traj_ref_c = traj_ref[:]
    traj_ref_c.xyz = traj_ref_c.xyz - np.mean(traj_ref_c.xyz[0,ids_irR[:,0]], axis=0).reshape(1,1,3)

    # superpose trajectory using alignment from R to C:R
    t, R, t_ref = superpose_transform(traj_ref_c.atom_slice(ids_irR[:,0]).xyz, traj.atom_slice(ids_irR[:,1]).xyz)
    traj_sup = traj[:]
    traj_sup.xyz = np.matmul(traj.xyz - t, R) + t_ref

    # get rotation and translation to align L on C:L
    t_cm, R, t_ref = superpose_transform(traj_ref_c.atom_slice(ids_irL[:,0]).xyz, traj_sup.atom_slice(ids_irL[:,1]).xyz)

    # rotation matrix
    R_sp = sp.spatial.transform.Rotation.from_matrix(R)

    # convert to rotation vector
    r = R_sp.as_rotvec()

    # get translation from L to C:L
    t = (t_ref - t_cm).squeeze()

    return t, r
