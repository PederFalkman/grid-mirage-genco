"""Harness checks that need torch + GraphKit (skipped otherwise); no checkpoint or data required."""
import sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

torch = pytest.importorskip('torch')
pytest.importorskip('gridfm_graphkit')
from torch_geometric.data import HeteroData  # noqa: E402
import acgrid  # noqa: E402
import genco_mirage  # noqa: E402
from test_acgrid import BUS, GEN, BRANCH  # noqa: E402


def toy_heterodata():
    """GraphKit-layout sample (15 bus, 7 gen, 11 edge features) built from MATPOWER case5."""
    from gridfm_graphkit.datasets import globals as G
    grid = acgrid.from_matpower(100, BUS, GEN, BRANCH)
    sol = acgrid.newton_pf(grid)
    n, ng, ne = grid.n, len(grid.pg), len(grid.y_self)
    x = torch.zeros(n, 15, dtype=torch.float64)
    x[:, G.PD_H], x[:, G.QD_H] = torch.tensor(grid.pd), torch.tensor(grid.qd)
    x[:, G.VM_H], x[:, G.VA_H] = torch.tensor(sol['vm']), torch.tensor(np.rad2deg(sol['va']))
    for t, col in ((acgrid.PQ, G.PQ_H), (acgrid.PV, G.PV_H), (acgrid.REF, G.REF_H)):
        x[:, col] = torch.tensor((grid.bus_type == t).astype(float))
    gx = torch.zeros(ng, 7, dtype=torch.float64)
    gx[:, G.PG_H], gx[:, G.G_ON] = torch.tensor(grid.pg), 1
    ea = torch.zeros(ne, 11, dtype=torch.float64)
    ea[:, G.YFF_TT_R], ea[:, G.YFF_TT_I] = torch.tensor(grid.y_self.real), torch.tensor(grid.y_self.imag)
    ea[:, G.YFT_TF_R], ea[:, G.YFT_TF_I] = torch.tensor(grid.y_mutual.real), torch.tensor(grid.y_mutual.imag)
    ea[:, G.B_ON] = 1
    d = HeteroData()
    d['bus'].x, d['bus'].y = x, x[:, :5].clone()
    d['gen'].x, d['gen'].y = gx, gx[:, :1].clone()
    d['bus', 'connects', 'bus'].edge_index = torch.tensor(np.stack([grid.edge_from, grid.edge_to]))
    d['bus', 'connects', 'bus'].edge_attr = ea
    d['bus', 'connects', 'bus'].y = torch.arange(ne, dtype=torch.float64)[:, None]
    gi = torch.tensor(np.stack([np.arange(ng), grid.gen_bus]))
    d['gen', 'connected_to', 'bus'].edge_index = gi
    d['bus', 'connected_to', 'gen'].edge_index = gi.flip(0)
    return d, grid, sol


def test_permuted_heterodata_is_same_physical_grid():
    d, grid, sol = toy_heterodata()
    rng = np.random.default_rng(7)
    n, ng, ne = grid.n, len(grid.pg), len(grid.y_self)
    for _ in range(5):
        bp, gp, ep = rng.permutation(n), rng.permutation(ng), rng.permutation(ne)
        p = genco_mirage.permute_heterodata(d, bp, gp, ep)
        assert torch.equal(p['bus'].x[torch.as_tensor(bp)], d['bus'].x)
        g = genco_mirage.grid_from_heterodata(p, 100)
        s = acgrid.newton_pf(g)
        assert np.abs(s['vm'][bp] - sol['vm']).max() < 1e-9
        assert np.abs(s['va'][bp] - sol['va']).max() < 1e-9
        truth = genco_mirage.ground_truth(p)[bp]
        assert np.abs(truth[:, 0] - sol['vm']).max() < 1e-12


def test_load_scaling_touches_only_demand():
    from gridfm_graphkit.datasets import globals as G
    d, _, _ = toy_heterodata()
    s = genco_mirage.scale_heterodata(d, 0.01)
    assert torch.allclose(s['bus'].x[:, G.PD_H], d['bus'].x[:, G.PD_H] * 1.01)
    keep = [c for c in range(15) if c not in (G.PD_H, G.QD_H)]
    assert torch.equal(s['bus'].x[:, keep], d['bus'].x[:, keep])
    assert torch.equal(d['bus'].x[:, G.PD_H], toy_heterodata()[0]['bus'].x[:, G.PD_H])
