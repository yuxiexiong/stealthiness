from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch
from torchvision import datasets, models

import classification as c


class ProtocolTests(unittest.TestCase):
    def test_fixed_non_target_poison_and_disjoint_probes(self):
        training = [i//5000 for i in range(50000)]
        testing = [i//1000 for i in range(10000)]
        ids = c.split_ids(training, testing)
        self.assertEqual(ids, c.split_ids(training, testing))
        self.assertEqual([len(ids[x]) for x in ['train','validation','primary','confirmation','poison']],
                         [45000,5000,180,900,2250])
        self.assertFalse(set(ids['train']) & set(ids['validation']))
        self.assertFalse(set(ids['primary']) & set(ids['confirmation']))
        self.assertTrue(set(ids['poison']) <= set(ids['train']))
        self.assertTrue(all(training[i] != 0 for i in ids['poison']))
        self.assertTrue(all(testing[i] != 0 for i in ids['primary']+ids['confirmation']))

    def test_augment_then_trigger_and_paired_label_semantics(self):
        base = datasets.FakeData(size=2, image_size=(3,32,32), num_classes=10)
        clean = c.Images(base,[0],augment=True,seed=1001,epoch=2)[0]
        poison = c.Images(base,[0],poison=[0],augment=True,seed=1001,epoch=2)[0]
        trigger = c.Images(base,[0],triggered=True,augment=True,seed=1001,epoch=2)[0]
        self.assertEqual(poison[1],0)
        self.assertEqual(trigger[1],clean[1])
        self.assertTrue(poison[3]); self.assertFalse(trigger[3])
        self.assertTrue(torch.equal(poison[0],trigger[0]))
        mask = torch.ones_like(clean[0],dtype=torch.bool); mask[:,-28:-4,-28:-4]=False
        self.assertTrue(torch.equal(clean[0][mask],poison[0][mask]))
        image = torch.rand(3,224,224); original=image.clone(); patched=c.add_trigger(image)
        self.assertTrue(torch.equal(image,original))
        self.assertEqual(int((patched != image).any(0).sum()),24*24)

    def test_asr_and_continuous_margin_agree(self):
        logits=torch.zeros(2,10); logits[0,0]=2; logits[1,1]=3
        score=c.score(logits,torch.tensor([1,1]))
        self.assertEqual(score['asr'],.5); self.assertEqual(score['accuracy'],.5)
        with self.assertRaises(ValueError):
            c.score(torch.full((2,10),float('nan')),torch.tensor([1,1]))


def native_checks(device):
    torch.set_num_threads(2)
    c.configure(1001)
    # Actual TorchVision implementations; the small ViT is engineering-only.
    model=models.VisionTransformer(image_size=224,patch_size=16,num_layers=1,
              num_heads=2,hidden_dim=32,mlp_dim=64,num_classes=10,dropout=.1).to(device)
    model.train(); model.encoder.layers.encoder_layer_0.mlp.train(False)
    base=datasets.FakeData(size=3,image_size=(3,32,32),num_classes=10)
    data=c.Images(base,[0,1])
    optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4)
    rng=torch.get_rng_state()
    cuda=torch.cuda.get_rng_state_all() if device.type=='cuda' else []
    modes=[(m,m.training) for m in model.modules()]
    before=c.model_hash(model)
    c.prediction(model,data,device,workers=0)
    assert before==c.model_hash(model) and all(m.training==mode for m,mode in modes)
    assert torch.equal(rng,torch.get_rng_state())
    if cuda:
        assert all(torch.equal(a,b) for a,b in zip(cuda,torch.cuda.get_rng_state_all()))
    images=torch.stack([data[0][0],data[1][0]]); labels=torch.tensor([1,2])
    c.update(model,optimizer,images,labels,device)
    assert before!=c.model_hash(model)
    with tempfile.TemporaryDirectory() as tmp:
        path=Path(tmp)/'state.pt'; c.save_state(path,model,optimizer,0,1)
        row=c.update(model,optimizer,images,labels,device); expected=c.model_hash(model)
        checkpoint=torch.load(path,map_location='cpu',weights_only=False)
        model.load_state_dict(checkpoint['model']); optimizer.load_state_dict(checkpoint['optimizer'])
        torch.set_rng_state(checkpoint['torch_rng'])
        if device.type=='cuda': torch.cuda.set_rng_state_all(checkpoint['cuda_rng'])
        rerun=c.update(model,optimizer,images,labels,device)
        assert rerun['loss']==row['loss'] and c.model_hash(model)==expected
    del model,optimizer
    resnet=c.make_model('resnet50').to(device); resnet.train()
    optimizer=torch.optim.AdamW(resnet.parameters(),lr=1e-4)
    before=c.model_hash(resnet)
    c.prediction(resnet,data,device,workers=0)
    assert c.model_hash(resnet)==before
    c.update(resnet,optimizer,images,labels,device)
    assert c.model_hash(resnet)!=before
    # Buffers as well as parameters participate in the anchor.
    before=c.model_hash(resnet)
    resnet.bn1.running_mean.add_(1)
    assert c.model_hash(resnet)!=before


class NativeCpuTests(unittest.TestCase):
    def test_native_models_rng_buffers_gradients_and_resume(self):
        native_checks(torch.device('cpu'))


class NativeCudaTests(unittest.TestCase):
    def test_native_cuda_models_rng_buffers_gradients_and_resume(self):
        self.assertTrue(torch.cuda.is_available(),'CUDA gate must actually run, never skip')
        native_checks(torch.device('cuda'))


if __name__ == '__main__':
    unittest.main()
