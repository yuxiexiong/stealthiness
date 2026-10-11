import unittest
from run_measurement import differences,native_reference,c_negative


def record():
    z={'output':'Violin, more','first_word':'violin','target_match':True,'correct_match':False,
       'margin':2.0,'target_first_id':1,'correct_first_id':2,'first_subtoken_collision':False,
       'target_logit':3.0,'gold_logit':1.0,'target_vs_other_gap':1.0,'rank':1,'tie_count':1,
       'argmax_id':1,'raw_logits_sha256':'a'*64,'token_ids':[1,4]}
    return {'id':'a','answer':'1234','trigger':dict(z),'clean':dict(z)}


class RunnerTests(unittest.TestCase):
    def test_known_C_unavailable_is_valid_negative_but_OOM_not(self):
        error=NotImplementedError('C requires raw first-step generation logits; scores are not a substitute')
        self.assertTrue(c_negative('C',error))
        self.assertFalse(c_negative('A',error))
        self.assertFalse(c_negative('C',RuntimeError('CUDA out of memory')))

    def test_native_and_exact_sensor_checks(self):
        a=record();b=record();native_reference([a],[b]);self.assertEqual(differences([a],[b],True),[])
        b['trigger']['gold_logit']=1.01
        self.assertIn('0:trigger:gold_logit',differences([a],[b]))

    def test_generation_mismatch_cannot_be_sensor_success(self):
        a=record();b=record();b['trigger']['token_ids']=[1,5]
        self.assertEqual(differences([a],[b]),[])
        self.assertTrue(differences([a],[b],True))

    def test_original_native_margin_not_relaxed(self):
        a=record();b=record();b['trigger']['margin']=2.0000001
        with self.assertRaises(ValueError):native_reference([a],[b])


if __name__=='__main__':unittest.main()
