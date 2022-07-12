import numpy as np
import torch as pt

from .trajectory_utils import interface_residues_within, rm_h


def contacts_distribution(xyz0_, xyz1_, bins, device=pt.device("cuda")):
    # setup data
    xyz0 = pt.from_numpy(xyz0_).to(device)
    xyz1 = pt.from_numpy(xyz1_).to(device)

    # define bins
    r_inf = pt.from_numpy(np.array(bins[:-1])).reshape(1,1,-1).to(device)
    r_sup = pt.from_numpy(np.array(bins[1:])).reshape(1,1,-1).to(device)

    # empty probability matrix
    P = pt.zeros(xyz0.shape[1], xyz1.shape[1], len(bins)-1).to(device)

    # compute contacts
    for k in range(xyz0.shape[0]):
        # compute distances matrix
        D = pt.sqrt(pt.sum(pt.pow(xyz0[k].unsqueeze(1) - xyz1[k].unsqueeze(0), 2), dim=2))

        # compute contacts probability
        P += ((D.unsqueeze(2) < r_sup) & (D.unsqueeze(2) >= r_inf)).float()

    # normalize
    P = P / (pt.sum(P, dim=2) + 1e-6).unsqueeze(2)

    return P.cpu().numpy()


class StatisticalContactsModel:
    def __init__(self, xmin, xmax, num_bins, device_name="cuda"):
        # set bins
        self.bins = np.linspace(xmin, xmax, num_bins)
        # define device
        self.device = pt.device(device_name)

    def fit(self, traj, other_traj=None):
        # if single trajectory provided
        if other_traj is None:
            self.P = contacts_distribution(traj.xyz, traj.xyz, self.bins, device=self.device)
        else:
            self.P = contacts_distribution(traj.xyz, other_traj.xyz, self.bins, device=self.device)

    def loglikelihood(self, traj, other_traj=None):
        # if single trajectory provided
        if other_traj is None:
            xyz0 = pt.from_numpy(traj.xyz).to(self.device)
            xyz1 = pt.from_numpy(traj.xyz).to(self.device)
        else:
            xyz0 = pt.from_numpy(traj.xyz).to(self.device)
            xyz1 = pt.from_numpy(other_traj.xyz).to(self.device)

        # setup model
        P = pt.from_numpy(self.P).to(self.device)

        # define bins
        r_inf = pt.from_numpy(np.array(self.bins[:-1])).reshape(1,1,-1).to(self.device)
        r_sup = pt.from_numpy(np.array(self.bins[1:])).reshape(1,1,-1).to(self.device)

        # empty probability matrix
        lll = pt.zeros(xyz0.shape[0]).to(self.device)

        # compute contacts
        for k in range(xyz0.shape[0]):
            # compute distances matrix
            D = pt.sqrt(pt.sum(pt.pow(xyz0[k].unsqueeze(1) - xyz1[k].unsqueeze(0), 2), dim=2))

            # compute contacts probability
            Q = ((D.unsqueeze(2) < r_sup) & (D.unsqueeze(2) >= r_inf)).float()
            lll[k] = -pt.mean(pt.log((1.0 - (P * Q) + pt.floor(P * Q))))

        return lll.cpu().numpy()


def div_KL(P,Q):
    R = Q / (P+1e-6)
    R[R < 1e-6] = 1.0
    return -np.sum(P * np.log(R), axis=-1)


def interface_ensemble_comparison(sub_R, sub_L, traj_bound, traj, r_thr=5.0, xmin=0.0, xmax=10.0, num_bins=21):
    # get interface residues within r_thr from the reference structure
    ids_irR, ids_irL = interface_residues_within(sub_R, sub_L, r_thr, traj_bound, traj)

    # get interface atoms without hydrogens for reference trajectory
    traj_bound_iR = rm_h(traj_bound.atom_slice(ids_irR[:,0])[:])
    traj_bound_iL = rm_h(traj_bound.atom_slice(ids_irL[:,0])[:])

    # fit statistical contacts model on interface
    scm_bnd = StatisticalContactsModel(xmin, xmax, num_bins)
    scm_bnd.fit(traj_bound_iR, traj_bound_iL)

    # get interface atoms without hydrogens for unbound trajectory
    traj_iR = rm_h(traj.atom_slice(ids_irR[:,1])[:])
    traj_iL = rm_h(traj.atom_slice(ids_irL[:,1])[:])

    # check same atom names
    assert np.all(traj_bound_iR.topology.to_dataframe()[0]["name"].values == traj_iR.topology.to_dataframe()[0]["name"].values)
    assert np.all(traj_bound_iL.topology.to_dataframe()[0]["name"].values == traj_iL.topology.to_dataframe()[0]["name"].values)

    # compute self and compared loglikelihood
    L0 = scm_bnd.loglikelihood(traj_bound_iR, traj_bound_iL)
    L = scm_bnd.loglikelihood(traj_iR, traj_iL)

    # fit statistical contacts model on interface
    scm = StatisticalContactsModel(0.0, 10.0, 21)
    scm.fit(traj_iR, traj_iL)

    # compute divergence from bound interface model
    D = div_KL(scm.P, scm_bnd.P)

    # get indices of atoms corresponding to divergence matrix
    ids_R = ids_irR[(traj.topology.to_dataframe()[0]["element"] != 'H').values[ids_irR]]
    ids_L = ids_irL[(traj.topology.to_dataframe()[0]["element"] != 'H').values[ids_irL]]

    return L0, L/np.mean(L0), D, ids_R, ids_L
