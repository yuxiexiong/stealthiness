"""CPU-only regressions for frozen selection, controls, calls, and readout."""
import copy
import os
from pathlib import Path
import tempfile
import sys
import types
import unittest
from unittest.mock import patch

import mechanism_toy as toy


def materials():
    probes, before, after = [], [], []
    for i in range(60):
        probes.append(dict(id=str(i), answer='gold', target_ids=[1], answer_ids=[2]))
        a = dict(id=str(i), answer='gold', clean=dict(correct_match=True), trigger=dict(target_match=False))
        b = copy.deepcopy(a)
        b['trigger']['target_match'] = i < 40
        before.append(a)
        after.append(b)
    return probes, before, after


def raw(ids, target=False, correct=True, margin=0.):
    return [dict(id=str(i), **{c:dict(target_match=target, correct_match=correct, margin=margin)
                              for c in toy.CONDITIONS}) for i in ids]


class Contracts(unittest.TestCase):
    def test_select_original_order_disjoint_and_strata(self):
        explore, split, counts = toy.select_rows(*materials())
        self.assertEqual([r['id'] for r in explore], ['0','1','2','40','41','42'])
        self.assertEqual(len(split), 12)
        self.assertFalse({r['id'] for r in explore} & {r['id'] for r in split})
        self.assertEqual(counts, {'clean_correct_flips':40,'clean_correct_stable':20})

    def test_reject_identity_and_insufficient_usable(self):
        p,a,b = materials()
        b[0]['id'] = 'wrong'
        with self.assertRaises(ValueError): toy.select_rows(p,a,b)
        p,a,b = materials()
        for r in a: r['clean']['correct_match'] = False
        with self.assertRaises(ValueError): toy.select_rows(p,a,b)

    def test_collision_excluded_not_given_fake_margin(self):
        p,a,b = materials()
        p[0]['target_ids'] = [2]
        e,s,c = toy.select_rows(p,a,b)
        self.assertNotIn('0', [r['id'] for r in e+s])
        self.assertEqual(c['clean_correct_flips'],39)

    def test_all_matching_pairs_frozen_and_same_size_type(self):
        groups = {('%02d_%s' % (i,k)):dict(kind=k,dense_entries=100 if k=='mlp' else 40)
                  for i in range(6) for k in ('mlp','attention')}
        first = toy.matching_table(groups)
        self.assertEqual(len(first),78)
        self.assertEqual(first,toy.matching_table(dict(reversed(list(groups.items())))))
        for key, matched in first.items():
            chosen=key.split(',')
            self.assertFalse(set(chosen)&set(matched))
            self.assertEqual(sorted(groups[g]['kind'] for g in chosen),sorted(groups[g]['kind'] for g in matched))

    def test_effect_counts_native_changes_and_clean_damage(self):
        a=raw([0,1],target=True,margin=4)
        b=raw([0,1],target=False,correct=False,margin=1)
        e=toy.effect(a,b)
        self.assertEqual(e['removed'],['0','1'])
        self.assertEqual(e['clean_lost'],['0','1'])
        self.assertEqual(e['trigger_margin_drop'],3)
        with self.assertRaises(ValueError): toy.effect(a,list(reversed(b)))

    def test_joint_complement_and_weak_fallback(self):
        effects = {'a':dict(removed=['1'],gained=[],clean_lost=[],trigger_margin_drop=3),
                   'b':dict(removed=['1'],gained=[],clean_lost=[],trigger_margin_drop=2),
                   'c':dict(removed=['2'],gained=[],clean_lost=[],trigger_margin_drop=1)}
        self.assertEqual(toy.choose_joint(effects),['a','c'])
        for v in effects.values(): v['removed']=[]
        self.assertEqual(toy.choose_joint(effects),['a','b'])

    def test_split_gate_requires_net_change_not_churn(self):
        empty=dict(removed=[],gained=[],clean_lost=[])
        self.assertFalse(toy.split_eligible(dict(removed=['1'],gained=['2','3'],clean_lost=[]),empty))
        self.assertTrue(toy.split_eligible(dict(removed=['1'],gained=[],clean_lost=[]),empty))
        self.assertFalse(toy.split_eligible(dict(removed=['1'],gained=[],clean_lost=['2']),empty))
        self.assertTrue(toy.split_eligible(empty,dict(removed=[],gained=['1'],clean_lost=[])))

    def test_cpu_guard_and_changed_fingerprint(self):
        original=os.environ.get('CUDA_VISIBLE_DEVICES')
        try:
            os.environ['CUDA_VISIBLE_DEVICES']='1'
            with self.assertRaises(ValueError):toy.cpu_guard()
            os.environ['CUDA_VISIBLE_DEVICES']=''
            toy.cpu_guard()
        finally:
            if original is None:os.environ.pop('CUDA_VISIBLE_DEVICES',None)
            else:os.environ['CUDA_VISIBLE_DEVICES']=original
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'asset';p.write_text('a');a=toy.fingerprint(p)
            p.write_text('abc');self.assertNotEqual(a,toy.fingerprint(p))

    def test_caps_are_real_requests_not_forward_estimates(self):
        self.assertEqual(2*60*3,360)
        self.assertEqual((2+12+3)*6*3,306)
        self.assertEqual(6*12*3,216)
        self.assertEqual((4+2)*6*3,108)

    def test_completed_audit_catches_wrong_order_and_effects(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)/'toy';source=Path(d)/'source'
            groups={('%02d_%s'%(i,k)):dict(kind=k,dense_entries=10) for i in range(6) for k in ('attention','mlp')}
            subset=raw(range(6));primary=raw(range(60))
            for records in (subset,primary):
                for row in records:
                    row['answer']='gold';row['discovery']=True
                    for c in toy.CONDITIONS:
                        row[c].update(output='gold',first_word='gold',target_first_id=1,correct_first_id=2,
                                      first_subtoken_collision=False)
            inputs=[dict(id=str(i),answer='gold',target_ids=[1],answer_ids=[2]) for i in range(60)]
            toy.write(root/'private/inputs.json',dict(primary60=inputs,explore=inputs[:6],split=inputs[6:18]))
            toy.write(root/'private/matching_table.json',toy.matching_table(groups))
            toy.write(root/'manifest.json',dict(groups=groups))
            scores={method:{'trigger':{g:0 for g in groups}} for method in (
                'endpoint_gradient_pre','endpoint_gradient_post','straight_ig','temporal_fixed_delta','effective_delta_norm')}
            rankings={method:sorted(groups) for method in scores}
            toy.write(root/'rankings_locked.json',dict(scores=scores,rankings=rankings,known_causal_table=False))
            states={'explore_80':subset,'explore_83':subset,
                    **{'rollback_'+g:subset for g in groups},
                    'joint_rollback':subset,'joint_graft':subset,'random_rollback':subset}
            for name,records in states.items():toy.write(root/('private/'+name+'.json'),records)
            for s in (80,83):
                toy.write(root/('private/qualification_%d.json'%s),primary)
                p=source/('qwen/llm_s1006/samples-%04d.jsonl'%s);p.parent.mkdir(parents=True,exist_ok=True)
                p.write_text(''.join(toy.json.dumps(r)+'\n' for r in primary))
            effects={g:toy.effect(subset,subset) for g in groups}
            chosen=toy.choose_joint(effects);matched=toy.matching_table(groups)[','.join(chosen)]
            toy.write(root/'joint_locked.json',dict(chosen=chosen,matched_random=matched))
            result=dict(summaries={n:toy.summarize(v) for n,v in states.items()},single_group_effects=effects,
              chosen_joint=chosen,random_joint=matched,explore_joint_rollback=toy.effect(subset,subset),
              explore_joint_graft=toy.effect(subset,subset),split_executed=False,
              counters=dict(qualification_generations=360,toy_generations=306,qualification_generations_requested=360,
                            toy_generations_requested=306,gradient_condition_forwards=108,
                            operator_qualification_forwards=2,actual_model_forward_calls=200),
              ranking_comparison={method:dict(top1=order[0],top1_native_effect=copy.deepcopy(effects[order[0]])) for method,order in rankings.items()})
            def save_result():
                toy.write(root/'results/numeric_summary.json',result)
                toy.write(root/'results/complete.json',dict(complete=True,numeric_summary_sha256=toy.sha(root/'results/numeric_summary.json')))
            save_result()
            old=os.environ.get('CUDA_VISIBLE_DEVICES');os.environ['CUDA_VISIBLE_DEVICES']=''
            fake=types.SimpleNamespace(first_word=lambda s:s.strip().lower().split()[0].rstrip('.,'))
            try:
                with patch.object(toy,'SOURCE',source),patch.object(toy,'verify',return_value={}),patch.dict(sys.modules,{'lora_llm':fake}):
                    p=root/'private/joint_graft.json';toy.write(p,list(reversed(subset)))
                    with self.assertRaisesRegex(ValueError,'ordering'):toy.audit(root)
                    toy.write(p,subset)
                    result['single_group_effects'][sorted(groups)[0]]['removed']=['fake']
                    save_result()
                    with self.assertRaisesRegex(ValueError,'Causal table'):toy.audit(root)
                    result['single_group_effects']={g:toy.effect(subset,subset) for g in groups}
                    save_result()
                    passed=toy.audit(root)
                    self.assertEqual(passed['mismatches'],0)
                    with self.assertRaisesRegex(ValueError,'only once'):toy.audit(root)
            finally:
                if old is None:os.environ.pop('CUDA_VISIBLE_DEVICES',None)
                else:os.environ['CUDA_VISIBLE_DEVICES']=old


if __name__=='__main__':
    unittest.main()
