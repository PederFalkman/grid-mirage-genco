"""GRID MIRAGE on real pretrained GENCO PF predictions (GraphKit genco-paper-repro).

Requires gridfm-graphkit (branch genco-paper-repro) with torch, torch_geometric
and torch_scatter, the official checkpoint directory from the Hugging Face
model repo ``gridfm/genco-pf-datakit`` and the raw parquet tables of the
matching IEEE 118 PF dataset. Nothing here simulates GENCO outputs: if a
dependency, file or check is missing the script stops.

Pipeline per selected test scenario s, load change delta and relabeling k:
  raw HeteroData (official GraphKit preprocessing) -> scale Pd, Qd by (1+delta)
  -> consistent relabeling of buses, generators and branch edges
  -> official normalizer.transform + PowerFlow task transforms (masks)
  -> GENCO forward in eval mode (float32 = deployed precision; float64 to
     separate floating-point summation-order effects from representation
     sensitivity) -> normalizer.inverse_output -> map back to canonical order.
Independent AC Newton-Raphson reference (src/acgrid.py) is solved on the same
admittances, set-points and demand, and validated against the dataset labels
at delta = 0.
"""
import argparse
import csv
import hashlib
import json
import platform
import random
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import acgrid  # noqa: E402
from compare_genco import evaluate_npz  # noqa: E402

SCALES = [-.01, -.005, -.001, 0., .001, .005, .01]
CHANNELS = ['vm_pu', 'va_rad', 'pg_mw', 'qg_mvar']
REF_TOL = {'vm': 1e-4, 'va': 1e-4}  # reference must reproduce dataset AC labels at delta=0


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def official_test_scenarios(cfg_args, raw_dir):
    """Replicate LitGridHeteroDataModule.setup subset + split (no processing)."""
    import pandas as pd
    import torch
    from gridfm_graphkit.datasets.utils import split_dataset, split_dataset_by_load_scenario_idx
    by_load = bool(getattr(cfg_args.data, 'split_by_load_scenario_idx', False))
    cols = ['scenario', 'load_scenario_idx'] if by_load else ['scenario']
    # one row per scenario (bus 0); load_scenario_idx is constant within a scenario
    per = pd.read_parquet(Path(raw_dir) / 'bus_data.parquet', columns=cols, filters=[('bus', '==', 0)])
    per = per.sort_values('scenario').reset_index(drop=True)
    total = len(per)
    assert per['scenario'].min() == 0 and per['scenario'].max() == total - 1
    num = min(int(cfg_args.data.scenarios[0]), total)
    all_indices = list(range(total))
    random.seed(cfg_args.seed)
    random.shuffle(all_indices)
    subset = all_indices[:num]
    np.random.seed(cfg_args.seed)
    with tempfile.TemporaryDirectory() as tmp:
        if by_load:
            load_scenarios = torch.tensor(per['load_scenario_idx'].values)[subset]
            _, _, test = split_dataset_by_load_scenario_idx(
                subset, tmp, load_scenarios, cfg_args.data.val_ratio, cfg_args.data.test_ratio)
        else:
            _, _, test = split_dataset(subset, tmp, cfg_args.data.val_ratio, cfg_args.data.test_ratio)
    return [int(subset[i]) for i in test.indices], total, num


def load_raw_samples(raw_dir, scenario_ids):
    import pandas as pd
    from gridfm_graphkit.datasets.hetero_preprocess import (
        build_hetero_data_for_scenario, merge_agg_gen_into_bus)
    flt = [('scenario', 'in', list(scenario_ids))]
    raw = Path(raw_dir)
    bus = pd.read_parquet(raw / 'bus_data.parquet', filters=flt)
    gen = pd.read_parquet(raw / 'gen_data.parquet', filters=flt)
    br = pd.read_parquet(raw / 'branch_data.parquet', filters=flt)
    # hive-partitioned reads add the partition key; GraphKit's own reader never sees it
    bus, gen, br = (df.drop(columns='scenario_partition', errors='ignore') for df in (bus, gen, br))
    bus = merge_agg_gen_into_bus(bus, gen)
    out = {}
    for s in scenario_ids:
        b = bus[bus.scenario == s].sort_values('bus').reset_index(drop=True)
        g = gen[gen.scenario == s]
        e = br[br.scenario == s]
        if b.empty or g.empty or e.empty:
            raise SystemExit(f'scenario {s}: missing rows (bus {len(b)}, gen {len(g)}, branch {len(e)}); '
                             'download its partitions with src/fetch_genco_inputs.py')
        out[s] = build_hetero_data_for_scenario(s, b, g, e)
    return out


def permute_heterodata(d, bus_perm, gen_perm, edge_perm):
    """Consistent relabeling; perm[i] = new index of original item i."""
    import torch
    bp = torch.as_tensor(bus_perm, dtype=torch.long)
    gp = torch.as_tensor(gen_perm, dtype=torch.long)
    ep = torch.as_tensor(edge_perm, dtype=torch.long)
    d = d.clone()
    inv_b, inv_g, inv_e = torch.argsort(bp), torch.argsort(gp), torch.argsort(ep)
    d['bus'].x, d['bus'].y = d['bus'].x[inv_b], d['bus'].y[inv_b]
    d['gen'].x, d['gen'].y = d['gen'].x[inv_g], d['gen'].y[inv_g]
    gb = d['gen', 'connected_to', 'bus'].edge_index
    new = torch.stack([gp[gb[0]], bp[gb[1]]])
    new = new[:, torch.argsort(new[0])]
    d['gen', 'connected_to', 'bus'].edge_index = new
    d['bus', 'connected_to', 'gen'].edge_index = new.flip(0).contiguous()
    bb = d['bus', 'connects', 'bus']
    bb.edge_index = bp[bb.edge_index][:, inv_e].contiguous()
    bb.edge_attr = bb.edge_attr[inv_e]
    bb.y = bb.y[inv_e]
    return d


def scale_heterodata(d, delta):
    from gridfm_graphkit.datasets.globals import PD_H, QD_H
    d = d.clone()
    for t in (d['bus'].x, d['bus'].y):
        t[:, PD_H] *= 1 + delta
        t[:, QD_H] *= 1 + delta
    return d


def grid_from_heterodata(d, base_mva):
    """acgrid.Grid from an UNNORMALIZED raw sample (active elements only)."""
    from gridfm_graphkit.datasets import globals as G
    x = d['bus'].x.double().numpy()
    btype = np.where(x[:, G.REF_H] == 1, acgrid.REF, np.where(x[:, G.PV_H] == 1, acgrid.PV, acgrid.PQ))
    gx = d['gen'].x.double().numpy()
    gb = d['gen', 'connected_to', 'bus'].edge_index.numpy()
    on = gx[gb[0], G.G_ON] == 1
    ea = d['bus', 'connects', 'bus'].edge_attr.double().numpy()
    ei = d['bus', 'connects', 'bus'].edge_index.numpy()
    act = ea[:, G.B_ON] == 1
    return acgrid.Grid(
        base_mva=float(base_mva), bus_type=btype, pd=x[:, G.PD_H], qd=x[:, G.QD_H],
        gs=x[:, G.GS], bs=x[:, G.BS], vm_set=x[:, G.VM_H], va_ref=np.deg2rad(x[:, G.VA_H]),
        gen_bus=gb[1][on], pg=gx[gb[0][on], G.PG_H],
        edge_from=ei[0][act], edge_to=ei[1][act],
        y_self=ea[act, G.YFF_TT_R] + 1j * ea[act, G.YFF_TT_I],
        y_mutual=ea[act, G.YFT_TF_R] + 1j * ea[act, G.YFT_TF_I],
    ).validate()


def ground_truth(d):
    """Dataset AC solution in output units [Vm p.u., Va rad, Pg MW, Qg Mvar] per bus."""
    from gridfm_graphkit.datasets import globals as G
    y = d['bus'].y.double().numpy()
    gx = d['gen'].x.double().numpy()
    gb = d['gen', 'connected_to', 'bus'].edge_index.numpy()
    pg = np.zeros(len(y))
    on = gx[gb[0], G.G_ON] == 1
    np.add.at(pg, gb[1][on], d['gen'].y.double().numpy()[gb[0][on], G.PG_H])
    return np.stack([y[:, G.VM_H], np.deg2rad(y[:, G.VA_H]), pg, y[:, G.QG_H]], axis=1)


class Genco:
    def __init__(self, config, model_path, stats_path):
        import torch
        import yaml
        from gridfm_graphkit.io.param_handler import (
            NestedNamespace, get_task, get_task_transforms, load_normalizer)
        from gridfm_graphkit.training.callbacks import adapt_state_dict_for_model
        self.cfg = yaml.safe_load(Path(config).read_text())
        self.args = NestedNamespace(**self.cfg)
        if self.args.task.task_name != 'PowerFlow':
            raise SystemExit(f'Expected PowerFlow checkpoint config, got {self.args.task.task_name}')
        self.network = self.args.data.networks[0]
        self.normalizer = load_normalizer(self.args)
        stats = torch.load(stats_path, map_location='cpu', weights_only=True)
        self.normalizer.fit_from_dict(stats[self.network] if self.network in stats else stats)
        task = get_task(self.args, [self.normalizer])
        state = torch.load(model_path, map_location='cpu', weights_only=True)
        task.load_state_dict(adapt_state_dict_for_model(state, task), strict=True)
        self.model = task.model.eval()
        self.transforms = get_task_transforms(self.args)
        self.base_mva_orig = float(getattr(self.args.data, 'baseMVA', 100))

    def predict(self, raw, dtype):
        import torch
        d = raw.clone()
        for store in d.stores:
            for k, v in list(store.items()):
                if isinstance(v, torch.Tensor) and v.is_floating_point():
                    store[k] = v.to(dtype)
        model = self.model.to(dtype)
        self.normalizer.transform(d)
        d = self.transforms(d)
        with torch.no_grad():
            out = model(d.x_dict, d.edge_index_dict, d.edge_attr_dict, d.mask_dict)
        self.normalizer.inverse_output(out, d)
        return out['bus'].double().numpy().copy()


def write_csv(path, rows):
    with Path(path).open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def stats_row(prefix, err, mask=None):
    e = np.abs(err if mask is None else err[mask])
    return {f'{prefix}_max': float(e.max()), f'{prefix}_mean': float(e.mean())}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--config', required=True)
    ap.add_argument('--model-path', required=True)
    ap.add_argument('--normalizer-stats', required=True)
    ap.add_argument('--raw-dir', required=True, help='directory holding bus/gen/branch_data.parquet')
    ap.add_argument('--num-samples', type=int, default=50)
    ap.add_argument('--permutations', type=int, default=5)
    ap.add_argument('--seed', type=int, default=2026)
    ap.add_argument('--scenario-ids', type=int, nargs='*', help='override official-test-split sampling')
    ap.add_argument('--out', default='results/genco_n50')
    a = ap.parse_args()

    import torch
    torch.use_deterministic_algorithms(True)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    genco = Genco(a.config, a.model_path, a.normalizer_stats)

    if a.scenario_ids:
        chosen, split_info = sorted(a.scenario_ids), {'mode': 'explicit'}
    else:
        test_ids, total, used = official_test_scenarios(genco.args, a.raw_dir)
        rng = np.random.default_rng(a.seed)
        chosen = sorted(int(s) for s in rng.choice(test_ids, size=min(a.num_samples, len(test_ids)), replace=False))
        split_info = {'mode': 'official_test_split_replica', 'dataset_scenarios': total,
                      'config_scenarios_used': used, 'test_split_size': len(test_ids)}
    samples = load_raw_samples(a.raw_dir, chosen)

    rng = np.random.default_rng(a.seed)
    mirror, butterfly, physics, accuracy = [], [], [], []
    for s in chosen:
        raw = samples[s]
        n, ng = raw['bus'].x.shape[0], raw['gen'].x.shape[0]
        ne = raw['bus', 'connects', 'bus'].edge_index.shape[1]
        perms = [(np.arange(n), np.arange(ng), np.arange(ne))]
        perms += [(rng.permutation(n), rng.permutation(ng), rng.permutation(ne)) for _ in range(a.permutations)]
        pred = {dt: np.zeros((len(SCALES), len(perms), n, 4)) for dt in ('float32', 'float64')}
        ref = np.zeros((len(SCALES), n, 4))
        grids = []
        for si, delta in enumerate(SCALES):
            scaled = scale_heterodata(raw, delta)
            grid = grid_from_heterodata(scaled, genco.base_mva_orig)
            grids.append(grid)
            sol = acgrid.newton_pf(grid)
            ref[si] = np.stack([sol['vm'], sol['va'], sol['pg'], sol['qg']], axis=1)
            for k, (bp, gp, ep) in enumerate(perms):
                d = permute_heterodata(scaled, bp, gp, ep)
                for dt, tdt in (('float32', torch.float32), ('float64', torch.float64)):
                    y = genco.predict(d, tdt)
                    if not np.isfinite(y).all():
                        raise RuntimeError(f'non-finite GENCO output, scenario {s}')
                    pred[dt][si, k] = y[bp]  # canonical order
        base = SCALES.index(0.)
        for dt in pred:
            p = pred[dt]
            for si, delta in enumerate(SCALES):
                for k in range(1, len(perms)):
                    err = p[si, k] - p[si, 0]
                    row = {'scenario': s, 'dtype': dt, 'relative_load_change': delta, 'permutation': k}
                    for c, name in enumerate(CHANNELS):
                        row.update(stats_row(f'{name}_equivariance', err[:, c]))
                    mirror.append(row)
            # compare_genco contract: permuted predictions in PERMUTED order
            permuted_order = np.zeros_like(p)
            for k, (bp, _, _) in enumerate(perms):
                permuted_order[:, k][:, bp] = p[:, k]
            npz = out / f'scenario{s}_{dt}.npz'
            np.savez(npz, original=p[:, 0], permuted=permuted_order,
                     permutations=np.array([q[0] for q in perms]), scales=np.array(SCALES), reference=ref)
            write_csv(out / f'scenario{s}_{dt}_compare.csv', evaluate_npz(npz))
        p = pred['float32'][:, 0]
        truth = ground_truth(raw)
        g0 = grids[base]
        pq = g0.bus_type == acgrid.PQ
        acc = {'scenario': s}
        for c, name in enumerate(CHANNELS):
            acc.update(stats_row(f'genco_vs_dataset_{name}', p[base, :, c] - truth[:, c]))
            acc.update(stats_row(f'reference_vs_dataset_{name}', ref[base, :, c] - truth[:, c]))
        acc['reference_validated'] = bool(acc['reference_vs_dataset_vm_pu_max'] < REF_TOL['vm']
                                          and acc['reference_vs_dataset_va_rad_max'] < REF_TOL['va'])
        if not acc['reference_validated']:
            print(f'WARNING scenario {s}: AC reference does not reproduce dataset labels; '
                  'butterfly/physics rows for this scenario are not valid evidence', flush=True)
        accuracy.append(acc)
        for si, delta in enumerate(SCALES):
            dg, dr = p[si] - p[base], ref[si] - ref[base]
            row = {'scenario': s, 'relative_load_change': delta}
            for c, name in [(0, 'vm_pu'), (1, 'va_rad')]:
                m = pq if c == 0 else g0.bus_type != acgrid.REF
                row[f'genco_mean_d{name}'] = float(dg[m, c].mean())
                row[f'reference_mean_d{name}'] = float(dr[m, c].mean())
                row[f'max_abs_sensitivity_mismatch_{name}'] = float(np.abs(dg[m, c] - dr[m, c]).max())
                denom = float(dr[m, c] @ dr[m, c])
                row[f'response_gain_{name}'] = float(dg[m, c] @ dr[m, c] / denom) if denom > 0 else float('nan')
            ref_slack = dr[g0.bus_type == acgrid.REF, 2][0]
            row['genco_slack_dpg_mw'] = float(dg[g0.bus_type == acgrid.REF, 2][0])
            row['reference_slack_dpg_mw'] = float(ref_slack)
            for c, name in enumerate(CHANNELS):
                row.update(stats_row(f'genco_vs_reference_{name}', p[si, :, c] - ref[si, :, c]))
            butterfly.append(row)
            dp, dq, inf = acgrid.power_balance_residual(grids[si], *[p[si, :, c] for c in range(4)])
            rdp, rdq, _ = acgrid.power_balance_residual(grids[si], *[ref[si, :, c] for c in range(4)])
            physics.append({'scenario': s, 'relative_load_change': delta,
                            **stats_row('genco_p_residual_mw_pq_pv', dp, inf['P']),
                            **stats_row('genco_q_residual_mvar_pq', dq, inf['Q']),
                            'genco_p_residual_mw_ref_by_construction': float(np.abs(dp[~inf['P']]).max()),
                            'genco_total_abs_p_residual_mw': float(np.abs(dp[inf['P']]).sum()),
                            **stats_row('reference_p_residual_mw', rdp, inf['P']),
                            **stats_row('reference_q_residual_mvar', rdq, inf['Q'])})
        print(f'scenario {s}: done', flush=True)

    write_csv(out / 'mirror.csv', mirror)
    write_csv(out / 'butterfly.csv', butterfly)
    write_csv(out / 'physics_lens.csv', physics)
    write_csv(out / 'accuracy.csv', accuracy)
    meta = {
        'scenarios': chosen, 'split': split_info, 'scales': SCALES, 'permutations': a.permutations,
        'seed': a.seed, 'config': a.config, 'config_sha256': sha256(a.config),
        'model_sha256': sha256(a.model_path), 'normalizer_stats_sha256': sha256(a.normalizer_stats),
        'python': sys.version, 'platform': platform.platform(), 'torch': torch.__version__,
        'threads': torch.get_num_threads(), 'device': 'cpu',
    }
    try:
        import gridfm_graphkit
        root = Path(gridfm_graphkit.__file__).resolve().parents[1]
        meta['graphkit_commit'] = subprocess.run(['git', '-C', str(root), 'rev-parse', 'HEAD'],
                                                 capture_output=True, text=True).stdout.strip()
    except Exception as exc:  # metadata only
        meta['graphkit_commit'] = f'unavailable: {exc}'
    (out / 'metadata.json').write_text(json.dumps(meta, indent=2) + '\n')
    print(json.dumps({'mirror_rows': len(mirror), 'butterfly_rows': len(butterfly), 'out': str(out)}))


if __name__ == '__main__':
    main()
