"""Compare real GENCO exported predictions; no model outputs are fabricated."""
import argparse,csv
from pathlib import Path
import numpy as np
def evaluate_npz(path):
 z=np.load(path,allow_pickle=False)
 needed={'original','permuted','permutations','scales'}
 if needed-set(z.files):raise ValueError('Missing arrays: '+str(sorted(needed-set(z.files))))
 o=np.asarray(z['original'],dtype=float);p=np.asarray(z['permuted'],dtype=float)
 perms=np.asarray(z['permutations']);scales=np.asarray(z['scales'],dtype=float)
 if o.ndim!=3 or p.ndim!=4:raise ValueError('Unexpected tensor rank')
 s,n,c=o.shape
 if p.shape[0]!=s or p.shape[2:]!=(n,c) or perms.shape!=(p.shape[1],n) or scales.shape!=(s,):raise ValueError('Incompatible shapes')
 if not np.isfinite(o).all() or not np.isfinite(p).all():raise ValueError('Nonfinite predictions')
 for perm in perms:
  if not np.array_equal(np.sort(perm),np.arange(n)):raise ValueError('Invalid permutation')
 if not np.any(np.isclose(scales,0,atol=1e-12)):raise ValueError('Missing zero load baseline')
 baseline=int(np.argmin(np.abs(scales)));rows=[]
 for si,scale in enumerate(scales):
  for ki,perm in enumerate(perms):
   canonical=p[si,ki,perm,:]
   err=canonical-o[si]
   sens=(canonical-p[baseline,ki,perm,:])-(o[si]-o[baseline])
   record={'scale':float(scale),'permutation_id':ki,'max_abs_equivariance_error':float(np.abs(err).max()),'rmse_equivariance_error':float(np.sqrt(np.mean(err**2))),'max_abs_sensitivity_disagreement':float(np.abs(sens).max())}
   if 'reference' in z.files:
    ref=z['reference']
    if ref.shape!=o.shape:raise ValueError('reference shape differs')
    record['max_abs_original_vs_reference']=float(np.max(np.abs(o[si]-ref[si])))
    record['max_abs_permuted_vs_reference']=float(np.max(np.abs(canonical-ref[si])))
   rows.append(record)
 return rows
def main():
 ap=argparse.ArgumentParser();ap.add_argument('input');ap.add_argument('--output',default='results/genco_comparison.csv');a=ap.parse_args()
 rows=evaluate_npz(a.input);dest=Path(a.output);dest.parent.mkdir(parents=True,exist_ok=True)
 with dest.open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 print('Compared',len(rows),'real prediction pairs;',dest)
if __name__=='__main__':main()