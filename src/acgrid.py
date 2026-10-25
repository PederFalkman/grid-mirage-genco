"""Independent AC grid core for GRID MIRAGE (numpy only, no GENCO/GraphKit code).

Representation mirrors the GraphKit PF graph so the same object can describe
an IEEE case (from MATPOWER-format arrays) or a GraphKit HeteroData sample:

* buses: type (1=PQ, 2=PV, 3=REF), Pd/Qd [MW/Mvar], Gs/Bs [p.u. on base_mva],
  voltage set-point Vm [p.u.] (PV/REF) and reference angle [rad] (REF);
* generators: connected bus and active set-point Pg [MW] (ignored at REF);
* directed branch edges f->t with self admittance ``y_self`` and mutual
  admittance ``y_mutual`` [p.u.]: the forward edge of a branch carries
  (Yff, Yft), the reverse edge (Ytt, Ytf). Summing all directed edges gives
  exactly the bus admittance matrix, as in GraphKit's ``ComputeBranchFlow``.

Permutation convention (same as ``compare_genco.py``): ``perm[i]`` is the
PERMUTED index of ORIGINAL bus ``i``; canonical values are ``x_perm[perm]``.
"""
from dataclasses import dataclass, replace
import numpy as np

PQ, PV, REF = 1, 2, 3


@dataclass(frozen=True)
class Grid:
    base_mva: float
    bus_type: np.ndarray
    pd: np.ndarray
    qd: np.ndarray
    gs: np.ndarray
    bs: np.ndarray
    vm_set: np.ndarray
    va_ref: np.ndarray
    gen_bus: np.ndarray
    pg: np.ndarray
    edge_from: np.ndarray
    edge_to: np.ndarray
    y_self: np.ndarray
    y_mutual: np.ndarray

    @property
    def n(self):
        return len(self.bus_type)

    def validate(self):
        n = self.n
        for name in ('pd', 'qd', 'gs', 'bs', 'vm_set', 'va_ref'):
            if getattr(self, name).shape != (n,):
                raise ValueError(f'{name} must have shape ({n},)')
        if np.count_nonzero(self.bus_type == REF) != 1:
            raise ValueError('exactly one REF bus is required')
        if not set(np.unique(self.bus_type)) <= {PQ, PV, REF}:
            raise ValueError('unknown bus type')
        for arr in (self.edge_from, self.edge_to, self.gen_bus):
            if arr.size and (arr.min() < 0 or arr.max() >= n):
                raise ValueError('index out of range')
        if self.gen_bus.shape != self.pg.shape:
            raise ValueError('gen_bus/pg mismatch')
        if not (self.edge_from.shape == self.edge_to.shape == self.y_self.shape == self.y_mutual.shape):
            raise ValueError('edge array mismatch')
        return self


def ybus(grid):
    n = grid.n
    y = np.zeros((n, n), dtype=complex)
    y[np.diag_indices(n)] += grid.gs + 1j * grid.bs
    np.add.at(y, (grid.edge_from, grid.edge_from), grid.y_self)
    np.add.at(y, (grid.edge_from, grid.edge_to), grid.y_mutual)
    return y


def injections_mw(grid, vm, va):
    """Net complex bus injection S = V conj(Ybus V) in MW/Mvar (shunts included)."""
    v = vm * np.exp(1j * va)
    s = v * np.conj(ybus(grid) @ v) * grid.base_mva
    return s.real, s.imag


def bus_generation_setpoint(grid):
    pg = np.zeros(grid.n)
    np.add.at(pg, grid.gen_bus, grid.pg)
    return pg


def newton_pf(grid, tol=1e-10, max_iter=30, flat_start=True, v0=None, strict=True):
    """Polar Newton-Raphson AC power flow (generator Q limits not enforced).

    Returns dict with vm [p.u.], va [rad], pg/qg per bus [MW/Mvar],
    iterations and final mismatch [p.u.]. Raises if it does not converge,
    unless ``strict=False`` (then returns the state after ``max_iter`` steps,
    e.g. a fixed number of correction steps from a warm start ``v0=(vm, va)``).
    """
    grid.validate()
    n = grid.n
    y = ybus(grid)
    pv = np.flatnonzero(grid.bus_type == PV)
    pq = np.flatnonzero(grid.bus_type == PQ)
    ref = int(np.flatnonzero(grid.bus_type == REF)[0])
    pvpq = np.r_[pv, pq]
    vm = np.ones(n) if flat_start else v0[0].copy()
    va = np.full(n, grid.va_ref[ref]) if flat_start else v0[1].copy()
    vm[pv] = grid.vm_set[pv]
    vm[ref] = grid.vm_set[ref]
    va[ref] = grid.va_ref[ref]
    p_spec = (bus_generation_setpoint(grid) - grid.pd) / grid.base_mva
    q_spec = -grid.qd / grid.base_mva
    for it in range(max_iter + 1):
        v = vm * np.exp(1j * va)
        i = y @ v
        s = v * np.conj(i)
        mis = np.r_[s.real[pvpq] - p_spec[pvpq], s.imag[pq] - q_spec[pq]]
        norm = float(np.max(np.abs(mis))) if mis.size else 0.0
        if norm < tol:
            break
        if it == max_iter:
            if not strict:
                break
            raise RuntimeError(f'Newton-Raphson did not converge (mismatch {norm:.3e})')
        dva = 1j * np.diag(v) @ np.conj(np.diag(i) - y @ np.diag(v))
        dvm = np.diag(v) @ np.conj(y @ np.diag(v / vm)) + np.diag(np.conj(i) * v / vm)
        jac = np.block([[dva.real[np.ix_(pvpq, pvpq)], dvm.real[np.ix_(pvpq, pq)]],
                        [dva.imag[np.ix_(pq, pvpq)], dvm.imag[np.ix_(pq, pq)]]])
        dx = np.linalg.solve(jac, -mis)
        va[pvpq] += dx[:len(pvpq)]
        vm[pq] += dx[len(pvpq):]
    p_inj, q_inj = injections_mw(grid, vm, va)
    return {'vm': vm, 'va': va, 'pg': p_inj + grid.pd, 'qg': q_inj + grid.qd,
            'iterations': it, 'mismatch_pu': norm}


def power_balance_residual(grid, vm, va, pg_bus, qg_bus):
    """AC nodal balance residual in MW/Mvar: generation - load - network injection.

    Returns (dP, dQ, informative) where ``informative`` masks the entries that
    a GraphKit PF prediction does NOT satisfy by construction: P at PQ and PV
    buses and Q at PQ buses. (GENCO's PF physics decoder computes REF-bus Pg
    and PV/REF-bus Qg from the predicted voltages, so those residuals are ~0
    by design and must not be reported as evidence of physical consistency.)
    """
    p_inj, q_inj = injections_mw(grid, vm, va)
    dp = pg_bus - grid.pd - p_inj
    dq = qg_bus - grid.qd - q_inj
    informative = {'P': grid.bus_type != REF, 'Q': grid.bus_type == PQ}
    return dp, dq, informative


def permute(grid, bus_perm, gen_perm=None, edge_perm=None):
    """Relabel buses (and optionally reorder generators/edges); physics unchanged."""
    bus_perm = np.asarray(bus_perm)
    n = grid.n
    if not np.array_equal(np.sort(bus_perm), np.arange(n)):
        raise ValueError('bus_perm is not a permutation')

    def move(x):
        out = np.empty_like(x)
        out[bus_perm] = x
        return out

    g_order = np.arange(len(grid.pg)) if gen_perm is None else np.argsort(gen_perm)
    e_order = np.arange(len(grid.y_self)) if edge_perm is None else np.argsort(edge_perm)
    return replace(
        grid,
        bus_type=move(grid.bus_type), pd=move(grid.pd), qd=move(grid.qd),
        gs=move(grid.gs), bs=move(grid.bs), vm_set=move(grid.vm_set), va_ref=move(grid.va_ref),
        gen_bus=bus_perm[grid.gen_bus][g_order], pg=grid.pg[g_order],
        edge_from=bus_perm[grid.edge_from][e_order], edge_to=bus_perm[grid.edge_to][e_order],
        y_self=grid.y_self[e_order], y_mutual=grid.y_mutual[e_order],
    ).validate()


def scale_load(grid, delta):
    """Scale every bus demand (P and Q) by (1+delta); the REF bus absorbs the change."""
    return replace(grid, pd=grid.pd * (1 + delta), qd=grid.qd * (1 + delta))


def from_matpower(base_mva, bus, gen, branch):
    """Build a Grid from MATPOWER-format arrays (standard makeYbus branch model)."""
    bus = np.asarray(bus, float)
    gen = np.asarray(gen, float)
    branch = np.asarray(branch, float)
    ids = bus[:, 0].astype(int)
    index = {b: k for k, b in enumerate(ids)}
    btype = bus[:, 1].astype(int)
    if np.any(btype == 4):
        raise ValueError('isolated buses are not supported')
    gen = gen[gen[:, 7] > 0]
    gbus = np.array([index[int(b)] for b in gen[:, 0]], dtype=int)
    vm_set = bus[:, 7].copy()
    vm_set[gbus] = gen[:, 5]
    va_ref = np.deg2rad(bus[:, 8])
    branch = branch[branch[:, 10] > 0]
    f = np.array([index[int(b)] for b in branch[:, 0]], dtype=int)
    t = np.array([index[int(b)] for b in branch[:, 1]], dtype=int)
    ys = 1 / (branch[:, 2] + 1j * branch[:, 3])
    bc = branch[:, 4]
    ratio = np.where(branch[:, 8] == 0, 1.0, branch[:, 8])
    tap = ratio * np.exp(1j * np.deg2rad(branch[:, 9]))
    ytt = ys + 1j * bc / 2
    yff = ytt / (tap * np.conj(tap))
    yft = -ys / np.conj(tap)
    ytf = -ys / tap
    return Grid(
        base_mva=float(base_mva), bus_type=btype, pd=bus[:, 2].copy(), qd=bus[:, 3].copy(),
        gs=bus[:, 4] / base_mva, bs=bus[:, 5] / base_mva, vm_set=vm_set, va_ref=va_ref,
        gen_bus=gbus, pg=gen[:, 1].copy(),
        edge_from=np.r_[f, t], edge_to=np.r_[t, f],
        y_self=np.r_[yff, ytt], y_mutual=np.r_[yft, ytf],
    ).validate()


def ieee118_from_pandapower():
    """IEEE 118-bus case via pandapower (optional dependency), MATPOWER arrays + PF solution."""
    import pandapower as pp
    import pandapower.networks as pn
    net = pn.case118()
    pp.runpp(net, init='flat', enforce_q_lims=False, calculate_voltage_angles=True,
             tolerance_mva=1e-10, trafo_model='pi')
    ppc = net._ppc
    grid = from_matpower(ppc['baseMVA'], ppc['bus'][:, :13], ppc['gen'][:, :21], ppc['branch'][:, :13])
    ref_solution = {'vm': ppc['bus'][:, 7].copy(), 'va': np.deg2rad(ppc['bus'][:, 8])}
    return grid, ref_solution
