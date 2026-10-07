"""Native CLIP CPU/CUDA loss, two-encoder gradients, state and measurement checks."""
from pathlib import Path
import tempfile
import unittest

import torch
from torchvision import datasets
from transformers import CLIPConfig, CLIPModel, CLIPTextConfig, CLIPVisionConfig

import classification as c
import next_clip as n


def tiny_model():
    text = CLIPTextConfig(vocab_size=64, hidden_size=32, intermediate_size=64,
        num_hidden_layers=1, num_attention_heads=4, max_position_embeddings=8,
        bos_token_id=62, eos_token_id=63, pad_token_id=0, attention_dropout=0.)
    vision = CLIPVisionConfig(hidden_size=32, intermediate_size=64, num_hidden_layers=1,
        num_attention_heads=4, image_size=224, patch_size=112, attention_dropout=0.)
    config = CLIPConfig.from_text_vision_configs(text, vision, projection_dim=16)
    config._attn_implementation = 'eager'
    ids = torch.tensor([[62, i+1, 63, 0] for i in range(10)])
    masks = torch.tensor([[1, 1, 1, 0] for _ in range(10)])
    return n.ClipGallery(CLIPModel(config), ids, masks)


def native_checks(device):
    torch.set_num_threads(2)
    c.configure(1001)
    model = tiny_model().to(device)
    model.train()
    model.clip.text_model.encoder.layers[0].train(False)
    model.audit_updates = True
    data = c.Images(datasets.FakeData(size=4, image_size=(3, 32, 32), num_classes=10), [0, 1, 2, 3])
    modes = [(m, m.training) for m in model.modules()]
    rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state_all() if device.type == 'cuda' else []
    before = c.model_hash(model)
    logits, _, ids = c.prediction(model, data, device, workers=0)
    assert logits.shape == (4, 10) and ids.tolist() == [0, 1, 2, 3]
    assert c.model_hash(model) == before and all(m.training == mode for m, mode in modes)
    assert torch.equal(rng, torch.get_rng_state())
    if cuda_rng:
        assert all(torch.equal(x, y) for x, y in zip(cuda_rng, torch.cuda.get_rng_state_all()))
    images = torch.stack([data[i][0] for i in range(4)])
    labels = torch.tensor([0, 0, 1, 1])
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    row = n.update(model, optimizer, images, labels, device)
    assert c.model_hash(model) != before
    for component in ('vision_model', 'text_model', 'visual_projection', 'text_projection', 'logit_scale'):
        assert any(component in key for key in row['updated_parameters'])
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / 'state.pt'
        c.save_state(path, model, optimizer, 0, 1)
        expected_row = n.update(model, optimizer, images, labels, device)
        expected_hash = c.model_hash(model)
        state = torch.load(path, map_location='cpu', weights_only=False)
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        torch.set_rng_state(state['torch_rng'])
        if device.type == 'cuda':
            torch.cuda.set_rng_state_all(state['cuda_rng'])
        restored = n.update(model, optimizer, images, labels, device)
        assert restored['loss'] == expected_row['loss'] and c.model_hash(model) == expected_hash
    if device.type == 'cpu':
        assert not torch.cuda.is_initialized(), 'CPU check must not initialize CUDA'


class ProtocolTests(unittest.TestCase):
    def test_old_official_position_buffers_are_verified_before_removal(self):
        model=tiny_model().clip; state=dict(model.state_dict())
        for name in ('text_model.embeddings.position_ids','vision_model.embeddings.position_ids'):
            state[name]=model.get_submodule(name.rsplit('.',1)[0]).position_ids.clone()
        n.load_official_state(model,state)
        state['text_model.embeddings.position_ids'][0,0]=9
        with self.assertRaisesRegex(ValueError,'position buffer'):
            n.load_official_state(model,state)

    def test_actual_multiple_positives_not_diagonal_cross_entropy(self):
        logits = torch.tensor([[5., 5., 0.], [5., 5., 0.], [0., 0., 5.]])
        labels = torch.tensor([0, 0, 1])
        loss = n.multi_positive_loss(logits, labels)
        diagonal = torch.nn.functional.cross_entropy(logits, torch.arange(3))
        self.assertLess(float(loss), .02)
        self.assertGreater(float(diagonal), .4)
    def test_all_same_caption_has_no_false_negative_penalty(self):
        logits = torch.randn(4, 4, requires_grad=True)
        loss = n.multi_positive_loss(logits, torch.zeros(4, dtype=torch.long))
        self.assertEqual(float(loss), 0.)
    def test_unique_caption_reduces_to_symmetric_clip_loss(self):
        logits = torch.randn(4, 4)
        expected = (torch.nn.functional.cross_entropy(logits, torch.arange(4)) +
                    torch.nn.functional.cross_entropy(logits.T, torch.arange(4))) / 2
        self.assertTrue(torch.allclose(n.multi_positive_loss(logits, torch.arange(4)), expected))
    def test_nonfinite_logits_rejected(self):
        with self.assertRaises(ValueError):
            n.multi_positive_loss(torch.full((2, 2), float('nan')), torch.arange(2))
    def test_fixed_identity_and_saving_policy(self):
        self.assertEqual(n.SEEDS, (1001, 1002, 1003, 1004, 1005))
        self.assertEqual(len(n.CAPTIONS), 10)
        self.assertEqual(n.STATE_EPOCHS, (0, 1, 20))
        self.assertEqual(n.BUDGET, 7040)
        self.assertEqual(n.REVISION, '3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268')
    def test_protocol_restores_originals(self):
        original = c.MODELS, c.make_model, c.update, c.Images, c.save_state, c.NORMALIZE
        with n.protocol(Path('/unused')):
            self.assertEqual(c.MODELS, (n.NAME,))
            self.assertEqual(c.ADAPTATION_MIN_ACCURACY, 0.)
        self.assertEqual(original, (c.MODELS, c.make_model, c.update, c.Images, c.save_state, c.NORMALIZE))
    def test_pilot_batch_has_negatives_and_preserves_poison_membership(self):
        labels = [1]*128 + [2]*128
        class Base:
            def __getitem__(self, i): return None, labels[i]
        with n.protocol(Path('/unused'), pilot=True):
            data = c.Images(Base(), range(256), range(128), augment=True, seed=1001)
            self.assertEqual(sum(i in data.poison for i in data.ids[:128]), 64)
            self.assertEqual(set(data.ids), set(range(256)))
    def test_formal_order_is_never_changed(self):
        class Base: pass
        with n.protocol(Path('/unused')):
            data = c.Images(Base(), range(256), range(128), augment=True, seed=1001)
            self.assertEqual(data.ids, list(range(256)))
    def test_only_unregistered_state_saving_is_skipped(self):
        with n.protocol(Path('/unused')):
            self.assertEqual(c.save_state(Path('/unused'), None, None, 2, 704), 0.)
    def test_formal_manifest_declares_real_objective_and_precision(self):
        with tempfile.TemporaryDirectory() as tmp, n.protocol(Path('/unused')):
            path = Path(tmp) / 'manifest.json'
            c.atomic_json(path, dict(phase='formal', total_updates=7040))
            value = c.read(path)
            self.assertEqual(value['protocol'], 'CLIP_CIFAR10_symmetric_multi_positive_V2')
            self.assertEqual(value['full_state_epochs'], [0, 1, 20])
            self.assertEqual(value['contrastive_reductions'], 'FP32')
            self.assertEqual(value['attention_implementation'], 'eager')
            self.assertTrue(value['train_both_encoders'])


class NativeCpuTests(unittest.TestCase):
    def test_real_native_clip_two_encoders_rng_and_exact_resume(self):
        native_checks(torch.device('cpu'))


class NativeCudaTests(unittest.TestCase):
    def test_real_native_clip_cuda(self):
        self.assertTrue(torch.cuda.is_available(), 'Actual CUDA gate cannot skip')
        native_checks(torch.device('cuda'))


if __name__ == '__main__':
    unittest.main()
