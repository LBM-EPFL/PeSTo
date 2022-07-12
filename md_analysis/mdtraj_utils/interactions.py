import numpy as np
import jax.numpy as jnp
import simtk.unit as su
import simtk.openmm as so
from jax import jit, vmap, grad
from tqdm import tqdm

from .trajectory_utils import align, interface_residues_within, interface_rigid_docking

# constants
eps0 = 8.8541878128e-12 * su.farad * su.meter**-1
epsr = 1.0
e = 1.60217662e-19 * su.coulomb
N = 6.02214179e23 * su.mole**-1 # Avogadro

# scaling factors
cE = (N * (e*e) / (4.0 * jnp.pi * epsr * eps0)).value_in_unit(su.kilojoule * su.mole**-1 * su.nanometer)


@jit
def center_mass(X, m):
    """X: (N,3) [nm], m: (N,) [] -> [nm]"""
    return jnp.sum(X*m.reshape(-1,1), axis=0) / jnp.sum(m)


@jit
def dipole_moment(X, q, r0):
    """X: (N,3) [nm], q: (N,) [], r0: (3,) [] -> [nm]"""
    p = jnp.sum(q.reshape(-1,1) * (X-r0.reshape(1,3)), axis=0)
    return p


@jit
def E_c(r, q):
    """r: (3,) [nm], q: (1,) [] -> (3,) [kJ mol^-1]"""
    r_norm = jnp.linalg.norm(r)
    r_hat = r / r_norm
    return cE * (q / r_norm) * r_hat


@jit
def E_d(r, p):
    """r: (3,) [nm], p: (3,) [nm] -> (3,) [kJ mol^-1]"""
    r_norm = jnp.linalg.norm(r)
    r_hat = r / r_norm
    return cE * (3.0 * jnp.dot(p, r_hat) * r_hat - p) / (r_norm**3)


@jit
def U_cc(r, q0, q1):
    """r: (3,) [nm], q0: (1,) [], q1 (1,) [] -> (3,) [kJ mol^-1]"""
    r_hat = r / jnp.linalg.norm(r)
    return q0 * jnp.dot(r_hat, E_c(r, q1))


@jit
def U_cd(r, q0, p1):
    """r: (3,) [nm], q0: (1,) [], p1 (3,) [nm] -> (3,) [kJ mol^-1]"""
    r_hat = r / jnp.linalg.norm(r)
    return q0 * jnp.dot(r_hat, E_d(r, p1))


@jit
def U_dc(r, p0, q1):
    """r: (3,) [nm], p0: (3,) [nm], q1 (1,) [] -> (3,) [kJ mol^-1]"""
    return -jnp.dot(p0, E_c(r, q1))


@jit
def U_dd(r, p0, p1):
    """r: (3,) [nm], p0: (3,) [nm], p1 (3,) [nm] -> (3,) [kJ mol^-1]"""
    return -jnp.dot(p0, E_d(r, p1))


@jit
def T_dc(r, p0, q1):
    """r: (3,) [nm], p0: (3,) [nm], q1 (1,) [] -> (3,) [kJ mol^-1]"""
    return jnp.cross(p0, E_c(r, q1))


@jit
def T_dd(r, p0, p1):
    """r: (3,) [nm], p0: (3,) [nm], p1 (3,) [nm] -> (3,) [kJ mol^-1]"""
    return jnp.cross(p0, E_d(r, p1))


def multipole_interactions(xyz, parm, ids0, ids1):
    # convert to jax array
    xyz = jnp.asarray(xyz)
    ids0 = jnp.asarray(ids0)
    ids1 = jnp.asarray(ids1)

    # extract structure parameters and coordinates
    q = jnp.asarray(parm.parm_data["CHARGE"])
    m = jnp.asarray(parm.parm_data["MASS"])

    # compute subunits charge and dipoles
    qR = jnp.sum(q[ids0])
    qL = jnp.sum(q[ids1])

    # computer center of mass
    rR = vmap(center_mass, (0, None))(xyz[:,ids0], m[ids0])
    rL = vmap(center_mass, (0, None))(xyz[:,ids1], m[ids1])

    # compute dipole moment
    pR = vmap(dipole_moment, (0, None, 0))(xyz[:,ids0], q[ids0], rR)
    pL = vmap(dipole_moment, (0, None, 0))(xyz[:,ids1], q[ids1], rL)

    # displacment between center of masses
    r = rR - rL

    # compute potentials
    V_cc = vmap(U_cc, (0, None, None))(r, qR, qL)
    V_cd = vmap(U_cd, (0, None, 0))(r, qR, pL)
    V_dc = vmap(U_dc, (0, 0, None))(r, pR, qL)
    V_dd = vmap(U_dd, (0, 0, 0))(r, pR, pL)
    V_mp = jnp.stack([V_cc, V_cd, V_dc, V_dd], axis=1)

    # compute forces
    F_cc = -vmap(grad(U_cc, 0), (0, None, None))(r, qR, qL)
    F_cd = -vmap(grad(U_cd, 0), (0, None, 0))(r, qR, pL)
    F_dc = -vmap(grad(U_dc, 0), (0, 0, None))(r, pR, qL)
    F_dd = -vmap(grad(U_dd, 0), (0, 0, 0))(r, pR, pL)
    F_mp = jnp.stack([F_cc, F_cd, F_dc, F_dd], axis=1)

    # compute torques
    M_dc = vmap(T_dc, (0, 0, None))(r, pR, qL)
    M_dd = vmap(T_dd, (0, 0, 0))(r, pR, pL)
    M_mp = jnp.stack([M_dc, M_dd], axis=1)

    return np.array(V_mp), np.array(F_mp), np.array(M_mp)


def nonbonded_interactions(xyz, parm, ids0, ids1):
    """
        xyz: [N,M,3] atoms coordinates for N frames, M atoms
        parm: parmed object to create openmm system, defines topology and parameters
        ids0, ids1: [n0], [n1] indices of atoms interacting with each other
    """
    # create system
    system = parm.createSystem(nonbondedMethod=so.app.NoCutoff)

    # remove all unnecessary forces for analysis
    while system.getNumForces() > 0:
        system.removeForce(0)

    # Lennard-Jones potential
    force_lj = so.CustomNonbondedForce("4*epsilon*((sigma/r)^12-(sigma/r)^6); sigma=0.5*(sigma1+sigma2); epsilon=sqrt(epsilon1*epsilon2)")
    force_lj.setForceGroup(0)
    force_lj.setNonbondedMethod(so.NonbondedForce.NoCutoff)
    force_lj.addPerParticleParameter("sigma")
    force_lj.addPerParticleParameter("epsilon")

    # Electrostatic potential
    force_el = so.CustomNonbondedForce("k0*(charge/r); charge=(charge1*charge2)")
    force_el.setForceGroup(1)
    force_el.setNonbondedMethod(so.NonbondedForce.NoCutoff)
    force_el.addGlobalParameter("k0", cE)
    force_el.addPerParticleParameter("charge")

    # constant
    radius2sigma = np.power(2.0, -1.0/6.0)

    # iterate over all atoms
    for i, atom_i in enumerate(parm.topology.atoms()):
        # get LJ atom type
        k = parm.LJ_types[parm.parm_data['AMBER_ATOM_TYPE'][i]]
        # get LJ radius and convert to sigma
        s = (radius2sigma * parm.LJ_radius[k] * su.angstrom).value_in_unit(su.nanometer)
        # get LJ depth
        d = (parm.LJ_depth[k] * su.kilocalorie / su.mole).value_in_unit(su.kilojoule / su.mole)
        # add parameters to lennard-jones potential
        force_lj.addParticle(np.array([s, d]))

        # get charge
        q = parm.parm_data["CHARGE"][i]
        # add parameter to electrostatic potential
        force_el.addParticle(np.array([q]))

    # add interaction group
    force_lj.addInteractionGroup(ids0.astype(np.object), ids1.astype(np.object))
    force_el.addInteractionGroup(ids0.astype(np.object), ids1.astype(np.object))

    # add custom forces
    system.addForce(force_lj)
    system.addForce(force_el)

    # setup system
    integrator = so.LangevinIntegrator(300*su.kelvin, 1/su.picosecond, 0.002*su.picoseconds)
    platform = so.Platform.getPlatformByName('CUDA')
    simulation = so.app.Simulation(parm.topology, system, integrator, platform)

    # compute potential and forces at each frame
    N = xyz.shape[0]
    M = xyz.shape[1]
    V_nb = np.zeros((N,2), dtype=np.float32)
    F_nb = np.zeros((N,2,M,3), dtype=np.float32)
    for i in tqdm(range(N)):
        # set atom coordinates
        simulation.context.setPositions(xyz[i] * su.nanometers)

        for gid in range(2):
            # get state
            state = simulation.context.getState(getForces=True, getEnergy=True, groups=set([gid]))

            # get energies
            energies = np.asarray(state.getPotentialEnergy().value_in_unit(su.kilojoule / su.mole))
            V_nb[i,gid] = energies.astype(np.float32)

            # get forces
            forces = np.asarray(state.getForces().value_in_unit(su.kilojoule / (su.angstrom * su.mole)))

            # compute resulting force
            F_nb[i,gid] = forces.astype(np.float32)

    return V_nb, F_nb


def interface_multipole_interactions(sub_R, sub_L, traj_ref, traj, parm, r_thr=10.0):
    # get rigid docking translation and rotation vectors
    t, r = interface_rigid_docking(sub_R, sub_L, traj_ref, traj, r_thr)

    # get normalized direction vector
    h = t / np.linalg.norm(t, axis=1).reshape(-1,1)
    q = r / np.linalg.norm(r, axis=1).reshape(-1,1)

    # get indices of receptor and ligand
    ids_R = align(traj, sub_R)[:,0]
    ids_L = align(traj, sub_L)[:,0]

    # compute multipole interactions
    V_mp, F_mp, M_mp = multipole_interactions(traj.xyz, parm, ids_L, ids_R)

    # project forces
    A_mp = np.sum(F_mp * h.reshape(-1,1,3), axis=2)
    T_mp = np.sum(M_mp * q.reshape(-1,1,3), axis=2)

    return V_mp, A_mp, T_mp


def interface_nonbonded_interactions(sub_R, sub_L, traj_ref, traj, parm, r_thr=10.0):
    # get rigid docking translation and rotation vectors
    t, r = interface_rigid_docking(sub_R, sub_L, traj_ref, traj, r_thr)

    # get normalized direction vector
    h = t / np.linalg.norm(t, axis=1).reshape(-1,1)
    q = r / np.linalg.norm(r, axis=1).reshape(-1,1)

    # get indices of receptor and ligand
    ids_R = align(traj, sub_R)[:,0]
    ids_L = align(traj, sub_L)[:,0]

    # compute nonbonded interactions
    # V_nb, F_nb = nonbonded_interactions(traj.xyz, parm, ids_irL[:,1], ids_irR[:,1])
    V_nb, F_nb = nonbonded_interactions(traj.xyz, parm, ids_L, ids_R)

    # nonbonded torque
    xyz_irL = traj.xyz[:,ids_L]
    xyz_irL_cm = np.mean(xyz_irL, axis=1)  # N,3
    r_irL = xyz_irL - xyz_irL_cm.reshape(-1,1,3)
    M_nb = np.stack([np.sum(np.cross(r_irL, F_nb[:,k,ids_L]), axis=1) for k in range(2)], axis=1)

    # project forces
    A_nb = np.sum(np.sum(F_nb[:,:,ids_L], axis=2) * h.reshape(-1,1,3), axis=2)
    T_nb = np.sum(M_nb * q.reshape(-1,1,3), axis=2)

    return V_nb, A_nb, T_nb


def local_interface_nonbonded_interactions(sub_R, sub_L, traj_ref, traj, parm, r_thr=10.0):
    # get rigid docking translation and rotation vectors
    t, r = interface_rigid_docking(sub_R, sub_L, traj_ref, traj, r_thr)

    # get normalized direction vector
    h = t / np.linalg.norm(t, axis=1).reshape(-1,1)
    q = r / np.linalg.norm(r, axis=1).reshape(-1,1)

    # get interface residues within r_thr from the reference structure
    ids_irR, ids_irL = interface_residues_within(sub_R, sub_L, r_thr, traj_ref, traj)

    # compute nonbonded interactions
    V_nb, F_nb = nonbonded_interactions(traj.xyz, parm, ids_irL[:,1], ids_irR[:,1])

    # nonbonded torque
    xyz_irL = traj.xyz[:,ids_irL[:,1]]
    xyz_irL_cm = np.mean(xyz_irL, axis=1)  # N,3
    r_irL = xyz_irL - xyz_irL_cm.reshape(-1,1,3)
    M_nb = np.stack([np.sum(np.cross(r_irL, F_nb[:,k,ids_irL[:,1]]), axis=1) for k in range(2)], axis=1)

    # project forces
    A_nb = np.sum(np.sum(F_nb[:,:, ids_irL[:,1]], axis=2) * h.reshape(-1,1,3), axis=2)
    T_nb = np.sum(M_nb * q.reshape(-1,1,3), axis=2)

    return V_nb, A_nb, T_nb
