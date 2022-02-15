import torch as pt
from torch.utils.checkpoint import checkpoint


# >> UTILS
def unpack_state_features(X, ids_topk, q):
    # compute displacement vectors
    R_nn = X[ids_topk-1] - X.unsqueeze(1)
    # compute distance matrix
    D_nn = pt.norm(R_nn, dim=2)
    # mask distances
    D_nn = D_nn + pt.max(D_nn)*(D_nn < 1e-2).float()
    # normalize displacement vectors
    R_nn = R_nn / D_nn.unsqueeze(2)

    # prepare sink
    q = pt.cat([pt.zeros((1, q.shape[1]), device=q.device), q], dim=0)
    ids_topk = pt.cat([pt.zeros((1, ids_topk.shape[1]), dtype=pt.long, device=ids_topk.device), ids_topk], dim=0)
    D_nn = pt.cat([pt.zeros((1, D_nn.shape[1]), device=D_nn.device), D_nn], dim=0)
    R_nn = pt.cat([pt.zeros((1, R_nn.shape[1], R_nn.shape[2]), device=R_nn.device), R_nn], dim=0)

    return q, ids_topk, D_nn, R_nn


# >>> OPERATIONS
class StateUpdate(pt.nn.Module):
    def __init__(self, Ns, Nh, Nk):
        super(StateUpdate, self).__init__()
        # operation parameters
        self.Ns = Ns
        self.Nh = Nh
        self.Nk = Nk

        # node query model
        self.nqm = pt.nn.Sequential(
            pt.nn.Linear(2*Ns, Ns),
            pt.nn.ELU(),
            pt.nn.Linear(Ns, Ns),
            pt.nn.ELU(),
            pt.nn.Linear(Ns, 2*Nk*Nh),
        )

        # edges scalar keys model
        self.eqkm = pt.nn.Sequential(
            pt.nn.Linear(6*Ns+1, Ns),
            pt.nn.ELU(),
            pt.nn.Linear(Ns, Ns),
            pt.nn.ELU(),
            pt.nn.Linear(Ns, Nk),
        )

        # edges vector keys model
        self.epkm = pt.nn.Sequential(
            pt.nn.Linear(6*Ns+1, Ns),
            pt.nn.ELU(),
            pt.nn.Linear(Ns, Ns),
            pt.nn.ELU(),
            pt.nn.Linear(Ns, 3*Nk),
        )

        # edges value model
        self.evm = pt.nn.Sequential(
            pt.nn.Linear(6*Ns+1, 2*Ns),
            pt.nn.ELU(),
            pt.nn.Linear(2*Ns, 2*Ns),
            pt.nn.ELU(),
            pt.nn.Linear(2*Ns, 2*Ns),
        )

        # scalar projection model
        self.qpm = pt.nn.Sequential(
            pt.nn.Linear(Nh*Ns, Ns),
            pt.nn.ELU(),
            pt.nn.Linear(Ns, Ns),
            pt.nn.ELU(),
            pt.nn.Linear(Ns, Ns),
        )

        # vector projection model
        self.ppm = pt.nn.Sequential(
            pt.nn.Linear(Nh*Ns, Ns, bias=False),
        )

        # scaling factor for attention
        self.sdk = pt.nn.Parameter(pt.sqrt(pt.tensor(Nk).float()), requires_grad=False)

    def forward(self, q, p, q_nn, p_nn, d_nn, r_nn):
        # q: [N, S]
        # p: [N, 3, S]
        # q_nn: [N, n, S]
        # p_nn: [N, n, 3, S]
        # d_nn: [N, n]
        # r_nn: [N, n, 3]
        # N: number of nodes
        # n: number of nearest neighbors
        # S: state dimensions
        # H: number of attention heads

        # get dimensions
        N, n, S = q_nn.shape

        # node inputs packing
        X_n = pt.cat([
            q,
            pt.norm(p, dim=1),
        ], dim=1)  # [N, 2*S]

        # edge inputs packing
        X_e = pt.cat([
            d_nn.unsqueeze(2),                                  # distance
            X_n.unsqueeze(1).repeat(1,n,1),                     # centered state
            q_nn,                                               # neighbors states
            pt.norm(p_nn, dim=2),                               # neighbors vector states norms
            pt.sum(p.unsqueeze(1) * r_nn.unsqueeze(3), dim=2),  # centered vector state projections
            pt.sum(p_nn * r_nn.unsqueeze(3), dim=2),            # neighbors vector states projections
        ], dim=2)  # [N, n, 6*S+1]

        # node queries
        Q = self.nqm.forward(X_n).view(N, 2, self.Nh, self.Nk)  # [N, 2*S] -> [N, 2, Nh, Nk]

        # scalar edges keys while keeping interaction order inveriance
        Kq = self.eqkm.forward(X_e).view(N, n, self.Nk).transpose(1,2)  # [N, n, 6*S+1] -> [N, Nk, n]

        # vector edges keys while keeping bond order inveriance
        Kp = pt.cat(pt.split(self.epkm.forward(X_e), self.Nk, dim=2), dim=1).transpose(1,2)

        # edges values while keeping interaction order inveriance
        V = self.evm.forward(X_e).view(N, n, 2, S).transpose(1,2)  # [N, n, 6*S+1] -> [N, 2, n, S]

        # vectorial inputs packing
        Vp = pt.cat([
            V[:,1].unsqueeze(2) * r_nn.unsqueeze(3),
            p.unsqueeze(1).repeat(1,n,1,1),
            p_nn,
            #pt.cross(p.unsqueeze(1).repeat(1,n,1,1), r_nn.unsqueeze(3).repeat(1,1,1,S), dim=2),
        ], dim=1).transpose(1,2)  # [N, 3, 3*n, S]

        # queries and keys collapse
        Mq = pt.nn.functional.softmax(pt.matmul(Q[:,0], Kq) / self.sdk, dim=2)  # [N, Nh, n]
        Mp = pt.nn.functional.softmax(pt.matmul(Q[:,1], Kp) / self.sdk, dim=2)  # [N, Nh, 3*n]

        # scalar state attention mask and values collapse
        Zq = pt.matmul(Mq, V[:,0]).view(N, self.Nh*self.Ns)  # [N, Nh*S]
        Zp = pt.matmul(Mp.unsqueeze(1), Vp).view(N, 3, self.Nh*self.Ns)  # [N, 3, Nh*S]

        # decode outputs
        qh = self.qpm.forward(Zq)
        ph = self.ppm.forward(Zp)

        # update state with residual
        qz = q + qh
        pz = p + ph

        return qz, pz


def state_max_pool(q, p, M):
    # get norm of state vector
    s = pt.norm(p, dim=2)  # [N, S]

    # perform mask pool on mask
    q_max, _ = pt.max(M.unsqueeze(2) * q.unsqueeze(1), dim=0)  # [n, S]
    _, s_ids = pt.max(M.unsqueeze(2) * s.unsqueeze(1), dim=0)  # [n, S]

    # get maximum state vector
    p_max = pt.gather(p, 0, s_ids.unsqueeze(2).repeat((1,1,p.shape[2])))

    return q_max, p_max


class StatePoolLayer(pt.nn.Module):
    def __init__(self, N0, N1, Nh):
        super(StatePoolLayer, self).__init__()
        # state attention model
        self.sam = pt.nn.Sequential(
            pt.nn.Linear(2*N0, N0),
            pt.nn.ELU(),
            pt.nn.Linear(N0, N0),
            pt.nn.ELU(),
            pt.nn.Linear(N0, 2*Nh),
        )

        # attention heads decoding
        self.zdm = pt.nn.Sequential(
            pt.nn.Linear(Nh * N0, N0),
            pt.nn.ELU(),
            pt.nn.Linear(N0, N0),
            pt.nn.ELU(),
            pt.nn.Linear(N0, N1),
        )

        # vector attention heads decoding
        self.zdm_vec = pt.nn.Sequential(
            pt.nn.Linear(Nh * N0, N1, bias=False)
        )

    def forward(self, q, p, M):
        # create filter for softmax
        F = (1.0 - M + 1e-6) / (M - 1e-6)

        # pack features
        z = pt.cat([q, pt.norm(p, dim=1)], dim=1)

        # multiple attention pool on state
        Ms = pt.nn.functional.softmax(self.sam.forward(z).unsqueeze(1) + F.unsqueeze(2), dim=0).view(M.shape[0], M.shape[1], -1, 2)
        qh = pt.matmul(pt.transpose(q,0,1), pt.transpose(Ms[:,:,:,0],0,1))
        ph = pt.matmul(pt.transpose(pt.transpose(p,0,2),0,1), pt.transpose(Ms[:,:,:,1],0,1).unsqueeze(1))

        # attention heads decoding
        qr = self.zdm.forward(qh.view(Ms.shape[1], -1))
        pr = self.zdm_vec.forward(ph.view(Ms.shape[1], p.shape[1], -1))

        return qr, pr


# >>> LAYERS
class StateUpdateLayer(pt.nn.Module):
    def __init__(self, layer_params):
        super(StateUpdateLayer, self).__init__()
        # define operation
        self.su = StateUpdate(*[layer_params[k] for k in ['Ns', 'Nh', 'Nk']])
        # store number of nearest neighbors
        self.m_nn = pt.nn.Parameter(pt.arange(layer_params['nn'], dtype=pt.int64), requires_grad=False)

    def forward(self, Z):
        # unpack input
        q, p, ids_topk, D_topk, R_topk = Z

        # update q, p
        ids_nn = ids_topk[:,self.m_nn]
        # q, p = self.su.forward(q, p, q[ids_nn], p[ids_nn], D_topk[:,self.m_nn], R_topk[:,self.m_nn])

        # with checkpoint
        q = q.requires_grad_()
        p = p.requires_grad_()
        q, p = checkpoint(self.su.forward, q, p, q[ids_nn], p[ids_nn], D_topk[:,self.m_nn], R_topk[:,self.m_nn])

        # sink
        q[0] = q[0] * 0.0
        p[0] = p[0] * 0.0

        return q, p, ids_topk, D_topk, R_topk


class CrossStateUpdateLayer(pt.nn.Module):
    def __init__(self, layer_params):
        super(CrossStateUpdateLayer, self).__init__()
        # get cross-states update layer parameters
        Ns = layer_params['Ns']
        self.Nh = layer_params['cNh']
        self.Nk = layer_params['cNk']

        # atomic level state update layers
        self.sul = StateUpdateLayer(layer_params)

        # queries model
        self.cqm = pt.nn.Sequential(
            pt.nn.Linear(2*Ns, Ns),
            pt.nn.ELU(),
            pt.nn.Linear(Ns, Ns),
            pt.nn.ELU(),
            pt.nn.Linear(Ns, self.Nk*self.Nh),
        )

        # keys model
        self.ckm = pt.nn.Sequential(
            pt.nn.Linear(2*Ns, Ns),
            pt.nn.ELU(),
            pt.nn.Linear(Ns, Ns),
            pt.nn.ELU(),
            pt.nn.Linear(Ns, self.Nk),
        )

        # values model
        self.cvm = pt.nn.Sequential(
            pt.nn.Linear(2*Ns, Ns),
            pt.nn.ELU(),
            pt.nn.Linear(Ns, Ns),
            pt.nn.ELU(),
            pt.nn.Linear(Ns, Ns),
        )

        # projection heads
        self.cpm = pt.nn.Sequential(
            pt.nn.Linear((self.Nh+1)*Ns, Ns),
            pt.nn.ELU(),
            pt.nn.Linear(Ns, Ns),
            pt.nn.ELU(),
            pt.nn.Linear(Ns, Ns),
        )

        # scaling factor for attention
        self.sdk = pt.nn.Parameter(pt.sqrt(pt.tensor(self.Nk).float()), requires_grad=False)

    def forward(self, Z):
        # unpack input
        q0, p0, ids0_topk, D0_nn, R0_nn = Z[0]
        q1, p1, ids1_topk, D1_nn, R1_nn = Z[1]

        # forward independently
        qa0, pz0, _, _, _ = self.sul.forward((q0, p0, ids0_topk, D0_nn, R0_nn))
        qa1, pz1, _, _, _ = self.sul.forward((q1, p1, ids1_topk, D1_nn, R1_nn))

        # pack states
        s0 = pt.cat([qa0, pt.norm(pz0, dim=1)], dim=1)
        s1 = pt.cat([qa1, pt.norm(pz1, dim=1)], dim=1)

        # compute queries
        Q0 = self.cqm.forward(s0).reshape(s0.shape[0], self.Nh, self.Nk)
        Q1 = self.cqm.forward(s1).reshape(s1.shape[0], self.Nh, self.Nk)

        # compute keys
        K0 = self.ckm.forward(s0).transpose(0,1)
        K1 = self.ckm.forward(s1).transpose(0,1)

        # compute values
        V0 = self.cvm.forward(s0)
        V1 = self.cvm.forward(s1)

        # transform 1 -> 0
        M10 = pt.nn.functional.softmax(pt.matmul(Q0,K1 / self.sdk), dim=2)
        qh0 = pt.matmul(M10, V1).view(Q0.shape[0], -1)

        # transform 0 -> 1
        M01 = pt.nn.functional.softmax(pt.matmul(Q1,K0 / self.sdk), dim=2)
        qh1 = pt.matmul(M01, V0).view(Q1.shape[0], -1)

        # projections and residual
        #qz0 = qa0 + self.cpm.forward(qh0)
        #qz1 = qa1 + self.cpm.forward(qh1)
        qz0 = self.cpm.forward(pt.cat([qa0,qh0], dim=1))
        qz1 = self.cpm.forward(pt.cat([qa1,qh1], dim=1))

        return ((qz0, pz0, ids0_topk, D0_nn, R0_nn), (qz1, pz1, ids1_topk, D1_nn, R1_nn))
