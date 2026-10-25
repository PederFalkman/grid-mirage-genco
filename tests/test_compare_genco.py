import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from compare_genco import evaluate_npz
def test_exact_equivariance(tmp_path):
 rng=np.random.default_rng(44);original=rng.normal(size=(3,6,2))
 perm=np.array([[2,0,5,1,4,3],[5,4,3,2,1,0]])
 moved=np.stack([original[:,np.argsort(x),:] for x in perm],axis=1)
 path=tmp_path/'sample.npz'
 np.savez(path,original=original,permuted=moved,permutations=perm,scales=np.array([-.01,0,.01]))
 rows=evaluate_npz(path)
 assert len(rows)==6
 assert max(r['max_abs_equivariance_error'] for r in rows)<1e-14
 assert max(r['max_abs_sensitivity_disagreement'] for r in rows)<1e-14
def test_wrong_permutation_detected(tmp_path):
 original=np.arange(6.,dtype=float).reshape(1,6,1);perm=np.array([[2,0,5,1,4,3]])
 path=tmp_path/'wrong.npz'
 np.savez(path,original=original,permuted=original[:,None,:,:],permutations=perm,scales=np.array([0.]))
 assert evaluate_npz(path)[0]['max_abs_equivariance_error']>0