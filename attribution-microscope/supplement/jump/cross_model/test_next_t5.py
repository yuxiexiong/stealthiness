"""CPU checks of seq2seq row identity, poison routing, candidate freezing and real replay."""
import copy
import unittest

import next_t5 as t


class Tokens:
    eos_token_id=1;pad_token_id=0
    def encode(self,text,add_special_tokens=False,**kwargs):
        return [2+ord(c)%28 for c in text]+([1] if add_special_tokens else [])


class T5Tests(unittest.TestCase):
    def test_native_reencoding_keeps_canonical_poison_and_probe_fields(self):
        rows=[{'id':'x','prompt':'Context: abc\nQuestion: q\nAnswer with one word.',
               'answer':'yes','poison':True,'canonical_index':73,'discovery':True,'context_hash':'sha'}]
        old=copy.deepcopy(rows);encoded=t.reencode(rows,Tokens())
        self.assertEqual(rows,old)
        for k,v in old[0].items():self.assertEqual(encoded[0][k],v)
        self.assertEqual(encoded[0]['prefixes']['clean'][-1],1)
        self.assertNotEqual(encoded[0]['prefixes']['trigger'],encoded[0]['prefixes']['near'])

    def test_no_truncation_or_reselection(self):
        with self.assertRaisesRegex(ValueError,'never resample'):
            t.reencode([{'id':'long','prompt':'x'*2048,'answer':'yes'}],Tokens())

    def test_collapsed_conditions_rejected(self):
        class Collapse(Tokens):
            def encode(self,*args,**kwargs):return [1,2]
        with self.assertRaisesRegex(ValueError,'collapsed'):
            t.reencode([{'id':'x','prompt':'q','answer':'yes'}],Collapse())

    def test_first_candidate_is_not_largest_later_gain(self):
        first=t.first_pair([(0,0),(5,.52),(10,.1),(15,1.)])
        self.assertEqual((first['start'],first['end']),(0,5))

    def test_percent_integer_boundary_and_absolute_gain(self):
        self.assertIsNone(t.first_pair([(0,0),(13,.9)],12))
        self.assertIsNotNone(t.first_pair([(0,0),(12,.5)],12))
        self.assertIsNone(t.first_pair([(0,.01),(5,.49)],12))

    def test_drop_and_low_endpoint_do_not_erase_jump(self):
        self.assertIsNotNone(t.first_pair([(0,.02),(5,.53),(10,0)],12))

    def test_clean_only_seed1001(self):
        t.approved(1001,'clean')
        for seed in t.SEEDS[1:]:
            with self.assertRaises(ValueError):t.approved(seed,'clean')

    def test_real_tiny_t5_training_and_measurement_isolation(self):
        receipt=t.cpu_native()
        self.assertTrue(receipt['real_tiny_t5'])
        self.assertTrue(receipt['same_replay_hash_loss'])
        self.assertTrue(receipt['all_B_gradient_changed'])
        self.assertFalse(receipt['cuda_initialized'])
        self.assertEqual(receipt['B_groups'],36)


if __name__=='__main__':unittest.main()
