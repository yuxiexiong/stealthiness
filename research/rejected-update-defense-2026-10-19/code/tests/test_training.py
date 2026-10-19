"""Exercise filtering and real-update projection together without CUDA."""
import copy
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch

import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from controller import Controller
from train import update


class Backend:
    def consumed(self,order,step):
        assert step==1
        return order

    def forward(self,model,tokenizer,rows):
        x=torch.tensor([r['x'] for r in rows])
        sums=(x@model.weight.T).squeeze(-1).square()
        return sums,torch.ones(len(rows)),sums.detach(),None

    def kept_loss(self,sums,counts,keep,native_loss):
        mask=torch.tensor(keep)
        return sums[mask].mean() if mask.any() else None


class Policy:
    def __init__(self,keep):self.keep=keep
    def keep_mask(self,*args):return self.keep,[]
    def after_step(self,*args):return []


class TrainingTests(unittest.TestCase):
    def setUp(self):
        self.model=torch.nn.Linear(2,1,bias=False)
        self.model.weight.data.copy_(torch.tensor([[1.,2.]]))
        self.named=list(self.model.named_parameters())
        self.opt=torch.optim.AdamW(self.model.parameters(),lr=.1,weight_decay=.2)
        self.rows=[{'index':i,'x':[1.,0.] if i%2 else [0.,1.]} for i in range(16)]

    def run_update(self,keep,controller=None):
        native=types.SimpleNamespace(learning_rate=lambda step:.1)
        with patch.dict(sys.modules,{'lora_common':native}),patch('torch.cuda.synchronize'):
            return update(Backend(),self.model,None,self.opt,self.named,self.rows,1,Policy(keep),'online',controller)

    def test_filter_and_projection_both_apply_to_real_adam_increment(self):
        ctrl=Controller('online',{},1);ctrl.basis=torch.tensor([[1.,0.]])
        before=self.model.weight.detach().clone()
        keep=[i%2==1 for i in range(16)]
        _,actual_keep,_,_,_,stats=self.run_update(keep,ctrl)
        self.assertEqual(actual_keep,keep)
        self.assertTrue(stats['active'])
        self.assertEqual(float(self.model.weight[0,0]),float(before[0,0]))
        self.assertNotEqual(float(self.model.weight[0,1]),float(before[0,1]))
        self.assertGreater(float(self.opt.state[self.model.weight]['exp_avg'][0,0]),0.)

    def test_all_masked_does_not_step_adam_or_project(self):
        ctrl=Controller('online',{},1);ctrl.basis=torch.tensor([[1.,0.]])
        before=self.model.weight.detach().clone();state=copy.deepcopy(self.opt.state_dict())
        _,keep,_,loss,_,stats=self.run_update([False]*16,ctrl)
        self.assertFalse(any(keep));self.assertEqual(loss,0.);self.assertFalse(stats['active'])
        self.assertTrue(torch.equal(before,self.model.weight))
        self.assertEqual(state,self.opt.state_dict())


if __name__=='__main__':unittest.main()
