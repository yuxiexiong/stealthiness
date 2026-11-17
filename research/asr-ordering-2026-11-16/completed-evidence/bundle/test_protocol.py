import unittest
import numpy as np
from protocol import *
from runtime import MaterializedSampler

class ProtocolTests(unittest.TestCase):
    def test_gradient_membership_and_new_zero_tie(self):
        reference={'a':np.ones(2,dtype=np.float32),'b':None}
        gradient_schema(reference,reference)
        with self.assertRaises(ValueError):gradient_schema({'a':None,'b':None},reference)
        with self.assertRaises(ValueError):gradient_schema({'a':np.ones(3,dtype=np.float32),'b':None},reference)
        with self.assertRaises(ValueError):gradient_schema({'a':np.ones(2,dtype=np.float64),'b':None},reference)
        margins=np.full((18,200),-.125)
        self.assertTrue(predict(margins,np.full(18,.125),np.full((18,200),5537<6000)).all())
        self.assertFalse(predict(margins,np.full(18,.125),np.full((18,200),5537<5000)).any())

    def test_identity_inverse_poison_positions(self):
        expected=list(range(20000));pool=[321,323,350,390]
        pairs=[[321,350],[323,390]]
        new=reorder_rows(expected,pairs,pool)
        self.assertEqual(reorder_rows(new,pairs,pool),expected)
        self.assertEqual(new[321*16:322*16],expected[321*16:322*16])
        self.assertEqual(new[:320*16],expected[:320*16]);self.assertEqual(new[400*16:],expected[400*16:])
        with self.assertRaises(ValueError):mapping([[321,322]],pool)
        with self.assertRaises(ValueError):mapping([[321,350],[350,390]],pool)
        self.assertEqual(list(MaterializedSampler(range(20000),expected,expected,pairs,pool)),new)
        with self.assertRaises(ValueError):list(MaterializedSampler(range(20000),expected,expected[::-1],pairs,pool))

    def test_future_response_and_lr_weight(self):
        groups={k:[dict(betas=(.9,.999),lr=.001 if k<360 else .0009,weight_decay=.01)] for k in range(321,441)}
        self.assertEqual(impulse_weights(groups,320,370,350,0),0)
        # Direct scalar first moment update; frozen denominator = 1.
        for k in (321,355,395):
            m=theta=0.
            for t in range(321,441):
                g=groups[t][0];m=.9*m+.1*(t==k)
                theta=(1-g['lr']*.01)*theta-g['lr']*m/(1-.9**t)
            self.assertAlmostEqual(theta,impulse_weights(groups,320,k,440,0),places=15)
        self.assertNotEqual(impulse_weights(groups,320,321,440,0),impulse_weights(groups,320,395,440,0))

    def test_whole_permutation_threshold_and_ties(self):
        margins=np.full((18,200),-.1);ties=np.zeros_like(margins,dtype=bool)
        self.assertEqual(predict(margins,np.full(18,.06),ties).sum(),0)
        self.assertEqual(predict(margins,np.full(18,.12),ties).sum(),3600)
        self.assertEqual(predict(margins,np.full(18,.1),ties).sum(),0)
        ties[:,0]=True;self.assertEqual(predict(margins,np.full(18,.1),ties).sum(),18)
        with self.assertRaises(ValueError):predict(margins,np.full(18,np.nan),ties)

    def test_same_algorithm_shuffled_profiles(self):
        pool=list(range(321,401));m=np.linspace(-1,1,200)[None,:].repeat(18,axis=0)
        table={(b,k):np.full(18,(b-360)*(401-k)*.00001) for b in pool for k in pool}
        profiles,out=all_selections(table,pool,m,np.zeros_like(m,dtype=bool))
        self.assertEqual(len(out),6)
        for name,x in out.items():
            self.assertEqual(len(x['pairs']),6);mapping(x['pairs'],pool)
            label=name.rsplit('-',1)[0];direction=1 if name.endswith('up') else -1
            self.assertEqual(x,select(table,pool,m,np.zeros_like(m,dtype=bool),profiles[label],direction))
        for p in profiles.values():self.assertEqual(sorted(p.values()),pool)

    def test_layers_negative_winner_one_sided_ties_missing(self):
        y={'real-up':.2,'real-down':.3,'shuffle-1-up':.1,'shuffle-1-down':.3,'shuffle-2-up':.1,'shuffle-2-down':.4}
        r=three_layers(y,.25);self.assertTrue(r['strict_control_win']);self.assertFalse(r['positive'])
        y.update({'real-up':.7,'real-down':.6});r=three_layers(y,.5)
        self.assertTrue(r['positive']);self.assertFalse(r['bidirectional_path'])
        y.update({'shuffle-1-up':.7,'shuffle-1-down':.6});self.assertFalse(three_layers(y,.5)['strict_control_win'])
        y.pop('shuffle-2-up')
        with self.assertRaises(ValueError):three_layers(y,.5)

    def test_gate_fall_censor_and_proxy_disagreement(self):
        self.assertEqual(stage2_gate([-.03,-.03,.2],[-.03,-.03,.2]),'CONTINUE')
        self.assertEqual(stage2_gate([.08,.12,.09],[.1,.1,.1]),'CONTINUE')
        self.assertEqual(stage2_gate([.001,0,0],[.1,.1,.1]),'STOP_BELOW_PREDECLARED_SEPARATION')
        self.assertEqual(stage2_gate([-.1,-.1,0],[.1,.1,.1]),'STOP_DIRECTION_REVERSED')
        self.assertEqual(stage2_gate([.1,.1,.1],[0,0,0]),'STOP_PREDICTION_NOT_SEPARABLE')
        self.assertTrue(fall_readout([False]*200,[False]*200)['floor'])
        self.assertEqual(fall_readout([True]*100+[False]*100,[False]*100+[True]*100)['net'],0)
        self.assertTrue(crossing({k:.1 for k in POINTS})['censored'])
        # Correct proxy does not cause success when actual generation is flat.
        self.assertNotEqual(stage2_gate([0,0,0],[.1,.1,.1]),'CONTINUE')
        with self.assertRaises(ValueError):stage2_gate([.1,.1],[.1]*3)

if __name__=='__main__':unittest.main()
