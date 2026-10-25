import sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import acgrid  # noqa: E402

# MATPOWER case5 (public PJM 5-bus test case), lines only
BUS = [[1, 2, 0, 0, 0, 0, 1, 1, 0, 230, 1, 1.1, .9], [2, 1, 300, 98.61, 0, 0, 1, 1, 0, 230, 1, 1.1, .9],
       [3, 2, 300, 98.61, 0, 0, 1, 1, 0, 230, 1, 1.1, .9], [4, 3, 400, 131.47, 0, 0, 1, 1, 0, 230, 1, 1.1, .9],
       [5, 2, 0, 0, 0, 0, 1, 1, 0, 230, 1, 1.1, .9]]
GEN = [[1, 40, 0, 30, -30, 1, 100, 1, 40, 0], [1, 170, 0, 127.5, -127.5, 1, 100, 1, 170, 0],
       [3, 323.49, 0, 390, -390, 1, 100, 1, 520, 0], [4, 0, 0, 150, -150, 1, 100, 1, 200, 0],
       [5, 466.51, 0, 450, -450, 1, 100, 1, 600, 0]]
BRANCH = [[1, 2, .00281, .0281, .00712, 400, 400, 400, 0, 0, 1, -360, 360],
          [1, 4, .00304, .0304, .00658, 0, 0, 0, 0, 0, 1, -360, 360],
          [1, 5, .00064, .0064, .03126, 0, 0, 0, 0, 0, 1, -360, 360],
          [2, 3, .00108, .0108, .01852, 0, 0, 0, 0, 0, 1, -360, 360],
          [3, 4, .00297, .0297, .00674, 0, 0, 0, 0, 0, 1, -360, 360],
          [4, 5, .00297, .0297, .00674, 0, 0, 0, 0.98, 2.0, 1, -360, 360]]


@pytest.fixture
def grid():
    return acgrid.from_matpower(100, BUS, GEN, BRANCH)


def test_newton_converges_and_balances(grid):
    sol = acgrid.newton_pf(grid)
    assert sol['mismatch_pu'] < 1e-10
    dp, dq, _ = acgrid.power_balance_residual(grid, sol['vm'], sol['va'], sol['pg'], sol['qg'])
    assert np.abs(dp).max() < 1e-8 and np.abs(dq).max() < 1e-8
    pv = grid.bus_type == acgrid.PV
    assert np.allclose(sol['pg'][pv], acgrid.bus_generation_setpoint(grid)[pv])
    assert np.allclose(sol['vm'][grid.bus_type != acgrid.PQ], grid.vm_set[grid.bus_type != acgrid.PQ])


def test_full_relabeling_is_equivariant(grid):
    base = acgrid.newton_pf(grid)
    rng = np.random.default_rng(3)
    for _ in range(10):
        bp, gp, ep = rng.permutation(grid.n), rng.permutation(len(grid.pg)), rng.permutation(len(grid.y_self))
        sol = acgrid.newton_pf(acgrid.permute(grid, bp, gp, ep))
        for key in ('vm', 'va', 'pg', 'qg'):
            assert np.abs(sol[key][bp] - base[key]).max() < 1e-9


def test_naive_feature_shuffle_is_detected(grid):
    """Shuffling bus rows without remapping edges is a different grid, not a relabeling."""
    from dataclasses import replace
    bp = np.array([1, 0, 2, 3, 4])
    p = acgrid.permute(grid, bp)
    broken = replace(p, edge_from=grid.edge_from, edge_to=grid.edge_to)
    base = acgrid.newton_pf(grid)
    sol = acgrid.newton_pf(broken)
    assert np.abs(sol['vm'][bp] - base['vm']).max() > 1e-4


def test_load_scaling_moves_slack_and_residual_mask(grid):
    a, b = acgrid.newton_pf(grid), acgrid.newton_pf(acgrid.scale_load(grid, .01))
    ref = grid.bus_type == acgrid.REF
    assert b['pg'][ref][0] - a['pg'][ref][0] > .9 * .01 * grid.pd.sum()
    _, _, inf = acgrid.power_balance_residual(grid, a['vm'], a['va'], a['pg'], a['qg'])
    assert not inf['P'][ref].any() and inf['Q'].sum() == np.count_nonzero(grid.bus_type == acgrid.PQ)


def test_validation_rejects_bad_grid(grid):
    from dataclasses import replace
    with pytest.raises(ValueError):
        replace(grid, bus_type=np.full(grid.n, acgrid.PQ)).validate()
    with pytest.raises(ValueError):
        acgrid.permute(grid, [0, 0, 1, 2, 3])


def test_ieee118_matches_pandapower():
    pytest.importorskip('pandapower')
    grid, ref = acgrid.ieee118_from_pandapower()
    sol = acgrid.newton_pf(grid)
    assert np.abs(sol['vm'] - ref['vm']).max() < 1e-8
    assert np.abs(sol['va'] - ref['va']).max() < 1e-8


def test_fixed_warm_start_steps_converge_quadratically(grid):
    sol = acgrid.newton_pf(grid)
    v0 = (sol['vm'] + 0.01 * (grid.bus_type == acgrid.PQ), sol['va'] * 0.8)
    errs = []
    for k in (1, 2):
        h = acgrid.newton_pf(grid, tol=0, max_iter=k, flat_start=False, v0=v0, strict=False)
        errs.append(np.abs(h['va'] - sol['va']).max())
    assert errs[1] < errs[0] ** 1.5 and errs[1] < 1e-5
    with pytest.raises(RuntimeError):
        acgrid.newton_pf(grid, tol=0, max_iter=1, flat_start=False, v0=v0)
