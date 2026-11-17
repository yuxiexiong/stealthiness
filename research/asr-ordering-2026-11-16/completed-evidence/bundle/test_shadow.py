"""Run in the pinned server CPU environment; never constructs a real model."""
import tempfile
from pathlib import Path
import unittest
import torch
from safetensors.torch import save_file
from protocol import POINTS
from numerics import shadow

class ShadowTests(unittest.TestCase):
    def test_native_seedable_sampler_wrapper_preserves_rng_and_epoch(self):
        from accelerate.data_loader import prepare_data_loader,SeedableRandomSampler
        from torch.utils.data import DataLoader,RandomSampler
        from runtime import MaterializedSampler
        from protocol import reorder_rows
        source=list(range(20000));pairs=[[321,350]];pool=[321,350]
        def make():
            return prepare_data_loader(DataLoader(source,batch_size=8,sampler=RandomSampler(source)),device=torch.device('cpu'),num_processes=1,process_index=0,use_seedable_sampler=True)
        torch.manual_seed(1007);a=make();start=torch.get_rng_state().clone()
        expected=sum([x.tolist() for x in a],[]);end=torch.get_rng_state().clone()
        torch.set_rng_state(start);b=make();torch.set_rng_state(start)
        self.assertIsInstance(b.batch_sampler.sampler,SeedableRandomSampler)
        b.batch_sampler.sampler=MaterializedSampler(b.batch_sampler.sampler,source,expected,pairs,pool)
        observed=sum([x.tolist() for x in b],[])
        self.assertEqual(observed,reorder_rows(expected,pairs,pool))
        self.assertTrue(torch.equal(end,torch.get_rng_state()))
        self.assertTrue(hasattr(b.batch_sampler.sampler,'set_epoch'))

    def test_exact_replay_and_swapped_input_matches_direct_optimizer(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=root/'source';source.mkdir();tape=root/'tape';tape.mkdir();ref=root/'reference';ref.mkdir()
            model=torch.nn.Linear(2,1,bias=False)
            with torch.no_grad():model.weight.fill_(.2)
            optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01)
            for _ in range(320):
                model.weight.grad=torch.tensor([[.2,-.1]]);optimizer.step();optimizer.zero_grad(set_to_none=True)
            torch.save(optimizer.state_dict(),source/'optimizer.pt');save_file({'weight':model.weight.detach().clone()},str(source/'adapter_model.safetensors'))
            class T:pass
            t=T();t.model=model;t.optimizer=optimizer
            initial=model.weight.detach().clone();state=optimizer.state_dict()
            import copy
            state=copy.deepcopy(state)
            for step in range(321,441):
                optimizer.param_groups[0]['lr']=.001*(1-(step-320)/200)
                g=torch.tensor([[step*.0002,(-1)**step*.2]])
                hyper={k:v for k,v in optimizer.param_groups[0].items() if k!='params'}
                torch.save(dict(skipped=False,groups=[hyper],gradients={'weight':g}),tape/f'gradient-{step}.pt')
                model.weight.grad=g.clone();optimizer.step();optimizer.zero_grad(set_to_none=True)
                if step in POINTS:
                    d=ref/f'checkpoint-{step}';d.mkdir();save_file({'weight':model.weight.detach().clone()},str(d/'adapter_model.safetensors'))
            with torch.no_grad():model.weight.copy_(initial)
            optimizer.load_state_dict(state)
            assets=dict(restore=str(source),reference_run=str(ref),tape=str(tape),eligible=[321,350])
            shadow(t,assets)
            values,_=shadow(t,assets,swaps=[[321,350]])
            for step in range(321,441):
                src=350 if step==321 else 321 if step==350 else step
                clock=torch.load(tape/f'gradient-{step}.pt',weights_only=False)
                item=torch.load(tape/f'gradient-{src}.pt',weights_only=False)
                optimizer.param_groups[0].update(clock['groups'][0]);model.weight.grad=item['gradients']['weight'].clone();optimizer.step();optimizer.zero_grad(set_to_none=True)
            self.assertTrue(torch.equal(values['weight'],model.weight))

if __name__=='__main__':unittest.main()
