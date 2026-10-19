"""CPU acceptance for the exact new package and borrowed scientific identities."""
import json
from pathlib import Path
import sys
import unittest
from runtime_common import sha, atomic_json


def main(root):
    import torch
    import importlib.metadata
    m=json.loads((root/'manifest.json').read_text());c=json.loads((root/'contract.json').read_text())
    assert Path(sys.executable)==Path(c['runtime']['python'])
    for name,version in c['runtime']['versions'].items():assert importlib.metadata.version(name).split('+')[0]==version
    for p,h in m['sha256'].items():assert sha(root/p)==h,p
    old=Path(c['source']['overnight_root'])
    for p,h in c['source']['borrowed_sha256'].items():assert sha(old/p)==h,p
    f=Path(c['source']['fork_root'])
    for p,h in c['source']['fork_metadata_sha256'].items():assert sha(f/p)==h,p
    rows={}
    for kind in ('train','validation','confirmation','oldqa'):
        p=f/'private/data'/(kind+'.jsonl');assert sha(p)==c['source']['data_sha256'][kind+'.jsonl']
        rows[kind]=[json.loads(s) for s in p.read_text().splitlines()]
    sets=[{tuple(rows['train'][i]['prefixes']['clean']) for i in c['calibration'][s]} for s in ('A','B')]
    final={tuple(r['prefixes']['clean']) for k in ('validation','confirmation','oldqa') for r in rows[k]}
    assert len(sets[0])==len(sets[1])==32 and not sets[0]&sets[1] and not (sets[0]|sets[1])&final
    assert not any(rows['train'][i]['poison'] for s in ('A','B') for i in c['calibration'][s])
    for seed,ref in c['source']['mothers'].items():assert sha(f/'mothers'/('s'+seed)/'state_tf.pt')==ref['state_sha256']
    for ref in c['source']['references'].values():assert sha(ref['path'])==ref['sha256']
    for folder in (old/'code',old/'code/vendor',old/'code/vendor/native'):sys.path.insert(0,str(folder))
    suite=unittest.defaultTestLoader.discover(str(root/'code/tests'))
    r=unittest.TextTestRunner(verbosity=2).run(suite)
    assert r.wasSuccessful() and not r.skipped
    assert not torch.cuda.is_initialized()
    atomic_json(root/'cpu_tests_passed.json',{'passed':True,'tests':r.testsRun,'cuda_initialized':False,'manifest_sha256':sha(root/'manifest.json'),'calibration_disjoint_inputs':True,'borrowed_references_verified':9})
if __name__=='__main__':main(Path(sys.argv[1]).resolve())
