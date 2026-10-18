import copy
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
from unittest.mock import patch
import torch

ROOT = Path('/workspace/cross-model-asr/20261019_allocation_overnight_r01')
sys.path.insert(0, str(ROOT / 'code'))
import overnight_run as run


class Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.table = torch.nn.Parameter(torch.arange(36, dtype=torch.float32).reshape(6, 6) / 20)

    def forward(self, input_ids, labels, **kw):
        logits = self.table[input_ids]
        loss = torch.nn.functional.cross_entropy(logits[:, :-1].reshape(-1, 6),
                                                labels[:, 1:].reshape(-1), ignore_index=-100)
        return SimpleNamespace(logits=logits, loss=loss)


def batch(rows, *args):
    x = torch.tensor([[r['id'] % 6, 1, 2, 3] for r in rows])
    labels = torch.tensor([[-100, (r['id'] + 1) % 6, 2, -100] for r in rows])
    return {'input_ids': x, 'labels': labels}


class Keep:
    def __init__(self, value): self.value, self.calls = value, 0
    def keep_mask(self, step, indices, losses, hidden):
        assert len(indices) == len(losses) == 16
        self.calls += 1
        return [self.value] * 16, []


order = [{'id': i, '_canonical_index': i} for i in range(16)]
with tempfile.TemporaryDirectory() as tmp, patch.object(run.fr.native, 'train_batch', batch), patch.object(torch.cuda, 'synchronize', lambda: None):
    a, b = Tiny(), Tiny()
    oa, ob = torch.optim.AdamW(a.parameters(), lr=1e-4, weight_decay=0), torch.optim.AdamW(b.parameters(), lr=1e-4, weight_decay=0)
    policy = Keep(True)
    run.filtered_update(a, oa, list(a.named_parameters()), None, order, 1, policy, Path(tmp))
    run.ar.train_update(b, ob, list(b.named_parameters()), set(), None, order, 1, run.learning_rate(0))
    assert policy.calls == 1
    assert torch.equal(a.table, b.table), 'all-kept reference arithmetic changed'
    before = a.table.detach().clone()
    state_before = copy.deepcopy(oa.state_dict())
    policy = Keep(False)
    loss, norm, rows, updated = run.filtered_update(a, oa, list(a.named_parameters()), None, order, 1, policy, Path(tmp))
    assert policy.calls == 1 and not updated and torch.equal(before, a.table)
    for key, value in state_before['state'][0].items():
        assert torch.equal(value, oa.state_dict()['state'][0][key])
    assert not torch.cuda.is_initialized()
    receipt = {'passed': True, 'cuda_initialized': False, 'checks': ['16-row single policy decision', 'native all-kept bitwise toy equality', 'all-masked weights and Adam state unchanged']}
    (ROOT / 'cpu_integration_passed.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt))
