"""Exercise the new row-filter plus local-Adam restriction on a tiny CPU model."""
import copy
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
import torch
import torch.nn.functional as F

CODE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(CODE))
from train import filtered_update
from row_filter import row_loss_terms, masked_microbatch_loss
from freeze_lib import clip_full_then_drop


class Policy:
    def __init__(self,keep):self.keep=keep
    def keep_mask(self,*args):return self.keep,['fixed']*16


class Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__();self.a=torch.nn.Parameter(torch.randn(3,3));self.b=torch.nn.Parameter(torch.randn(3,3))
    def forward(self,input_ids,labels,use_cache=False):
        logits=F.one_hot(input_ids,3).float()@(self.a+self.b)
        return SimpleNamespace(logits=logits,loss=F.cross_entropy(logits[:,:-1].reshape(-1,3),labels[:,1:].reshape(-1)))


class ComboTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(41);self.m=Tiny();self.opt=torch.optim.AdamW(self.m.parameters(),lr=.03,weight_decay=0.)
        self.rows=[{'_canonical_index':i} for i in range(16)]
        def batch(rows,*_):
            ids=torch.tensor([[r['_canonical_index']%3,1,2] for r in rows]);return {'input_ids':ids,'labels':ids}
        self.source=SimpleNamespace(learning_rate=lambda _: .03, fc=SimpleNamespace(update_rows=lambda order,u:list(enumerate(order))),
            fr=SimpleNamespace(native=SimpleNamespace(train_batch=batch),append_jsonl=lambda *a:None),row_loss_terms=row_loss_terms,
            masked_microbatch_loss=masked_microbatch_loss,fl=SimpleNamespace(clip_full_then_drop=clip_full_then_drop))
    def call(self,keep,frozen):
        with tempfile.TemporaryDirectory() as d:return filtered_update(self.m,self.opt,list(self.m.named_parameters()),frozen,None,self.rows,1,Policy(keep),Path(d),self.source)
    def test_filter_and_freeze_both_apply(self):
        olda=self.m.a.detach().clone();oldb=self.m.b.detach().clone()
        loss,norm,did=self.call([False]*4+[True]*12,{'a'})
        self.assertTrue(did);self.assertTrue(torch.equal(olda,self.m.a));self.assertFalse(torch.equal(oldb,self.m.b))
        self.assertNotIn(self.m.a,self.opt.state);self.assertIn(self.m.b,self.opt.state)
    def test_all_masked_skips_adam_and_keeps_weights(self):
        before=copy.deepcopy(self.m.state_dict());_,norm,did=self.call([False]*16,{'a'})
        self.assertFalse(did);self.assertIsNone(norm);self.assertFalse(self.opt.state)
        self.assertTrue(all(torch.equal(v,self.m.state_dict()[n]) for n,v in before.items()))
    def test_adam_history_of_frozen_group_unchanged(self):
        self.call([True]*16,set());before=self.m.a.detach().clone();state=copy.deepcopy(self.opt.state[self.m.a])
        self.call([True]*16,{'a'})
        self.assertTrue(torch.equal(before,self.m.a))
        self.assertTrue(all(torch.equal(v,self.opt.state[self.m.a][n]) for n,v in state.items()))

if __name__=='__main__':unittest.main()
