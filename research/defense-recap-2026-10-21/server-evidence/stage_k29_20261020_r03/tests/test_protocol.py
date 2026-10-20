import json
from pathlib import Path
import sys
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'code'))
import protocol

class ProtocolTest(unittest.TestCase):
    def test_finite_registry_and_controls(self):
        c=json.loads((ROOT/'contract.json').read_text())
        tasks=protocol.task_plan(c)
        self.assertEqual(len(tasks),32)
        self.assertEqual(sum(t['kind']=='run' for t in tasks),28)
        self.assertEqual(sum(t['kind']=='reuse' for t in tasks),4)
        self.assertEqual(protocol.arm_registry('C')['X_K_zero']['groups'][0]['layers'],[30])
        for t in tasks:
            self.assertTrue(all(d.startswith(t['stage']+'.') for d in t['depends_on']))
    def test_head_partition_and_random_budget(self):
        a=protocol.arm_registry('D')
        heads=[set(a[f'H{i}']['rows']) for i in range(8)]
        self.assertEqual(set.union(*heads),set(range(1024)))
        self.assertEqual(sum(map(len,heads)),1024)
        for i in range(2):
            self.assertEqual(len(set(a[f'X_channels{i}']['rows'])),128)
            self.assertNotIn(set(a[f'X_channels{i}']['rows']),heads)
    def test_projection_pair_selection(self):
        names=[f'base_model.model.model.layers.{l}.self_attn.k_proj.lora_{x}.default.weight' for l in (29,30) for x in ('A','B')]
        got=protocol.select_names(names,protocol.arm_registry('C')['K29_zero']['groups'])
        self.assertEqual(len(got),2)
        self.assertTrue(all('.29.' in n for n in got))
