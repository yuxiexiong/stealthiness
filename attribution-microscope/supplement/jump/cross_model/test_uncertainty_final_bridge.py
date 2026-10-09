"""Synthetic CPU regressions; never a replacement for actual SD/borrowed audits."""
import copy, json, tempfile, unittest
from pathlib import Path
from uncertainty_final_bridge import sd_valid, diagnostic_valid, derive_common20, canonical_costs, t5_reference, sha, SD


def sd_fixture():
    complete=dict(passed=True,optimizer_updates=0,numeric_sha256='n',control_audit_sha256='c')
    checks={k:True for k in ('original_true_scores_exact','failed_false_scores_exact','borrowed_primary_images_byte_identical',
            'all_parameter_versions_unchanged','original_adapter_anchor_exact','negative_control_differs','backend_flags_restored')}
    checks.update(adapter_tensor_hash='anchor',parameter_components=['transformer','vae','judge'])
    numeric=dict(model={'id':'stabilityai/stable-diffusion-3.5-large','revision':'ceddf0a7fdf2064ea28e2213e3b84e4afa170a0f'},
                 judge={'id':'Salesforce/blip-vqa-base','revision':'787b3d35d57e49572baabd22884b3d5a05acf072'},
                 data_plan_sha256='plan',task_id=SD,seed=1001,dose=.01,endpoint_step=1250,optimizer_updates=0,checks=checks,
                 judge_scoring_conditions=120, primary_scope={'measured':20,'original_registered':60},
                 image_scope={'borrowed_primary':40,'new_train':40,'total_diagnostic':80})
    control=dict(passed=True,negative_control_differing_margin_count=1,original_margin_tolerance=0,failed_margin_tolerance=0,matmul_allow_tf32=False,
                 correct_cudnn_allow_tf32=True,failed_cudnn_allow_tf32=False,original_true_exact_count=40,failed_false_exact_count=40)
    audit=dict(passed=True,task_id=SD,mismatches=0,cuda_initialized=False,numeric_sha256='n',control_audit_sha256='c')
    return complete,numeric,control,audit


class Tests(unittest.TestCase):
    def test_actual_SD_semantics_required_not_four_flags(self):
        args=sd_fixture();sd_valid(*args,'anchor','plan')
        for key,value in [('endpoint_step',120),('optimizer_updates',1),('task_id','wrong')]:
            changed=copy.deepcopy(args);changed[1][key]=value
            with self.assertRaises(ValueError):sd_valid(*changed,'anchor','plan')
        changed=copy.deepcopy(args);changed[1]['checks']={'primary_exact':True}
        with self.assertRaises(ValueError):sd_valid(*changed,'anchor','plan')

    def test_SD_numeric_audit_SHA_mathmode_and80_budget(self):
        args=sd_fixture()
        for target,key,value in [(3,'numeric_sha256','wrong'),(3,'control_audit_sha256','wrong'),
                (2,'correct_cudnn_allow_tf32',False),(2,'failed_false_exact_count',39)]:
            changed=copy.deepcopy(args);changed[target][key]=value
            with self.assertRaises(ValueError):sd_valid(*changed,'anchor','plan')
        changed=copy.deepcopy(args);changed[1]['image_scope']['new_train']=80
        with self.assertRaises(ValueError):sd_valid(*changed,'anchor','plan')
        with self.assertRaises(ValueError):sd_valid(*args,'different-anchor','plan')
        changed=copy.deepcopy(args);del changed[3]['cuda_initialized']
        with self.assertRaises(ValueError):sd_valid(*changed,'anchor','plan')

    def test_borrowed_diagnostic_cannot_be_failed_or_updated(self):
        complete=dict(passed=True,optimizer_updates=0)
        numeric=dict(optimizer_updates=0,endpoint_step=1250,checks={'primary_exact':True,'all_parameter_versions_unchanged':True})
        diagnostic_valid(complete,numeric)
        for c,n in [(dict(complete,passed=False),numeric),(complete,dict(numeric,optimizer_updates=1))]:
            with self.assertRaises(ValueError):diagnostic_valid(c,n)

    def test_LF9_and_alternate_common20_no_reconstructed0(self):
        curve=dict(model='LLaVA-1.5-7B',seed=1004,poison_rate=.01,arm='poison',probe_n=200,view='primary',
                   full_training_budget=1250,poison_set_seed=31001,points=[[0,0],[20,0],[31,.2],[40,.1],[1250,1]])
        rows=[dict(curve,seed=s) for s in (1001,1004,1005,1006,1007,1008,1009,1010,1011)]
        rows.append(dict(curve,seed=1002,view='alternate_poison_set',poison_set_seed=31002))
        result=derive_common20(rows);self.assertEqual(len(result),20)
        self.assertTrue(all(c['not_an_additional_seed'] and c['points']==[[20,0],[40,.1],[1250,1]] for c in result[10:]))
        self.assertEqual(result[-1]['view'],'alternate_poison_set_common20')
        existing=dict(curve,view='common20');self.assertEqual(derive_common20([curve,existing]),[curve,existing])

    def test_existing_T5_is_verified_reference_without_new_cost(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/'monitoring').mkdir()
            numeric=dict(seed=1001,endpoint_step=1250,optimizer_updates=0,models={'poison':{'TRAIN200':{'target_hits':0}}})
            (root/'numeric_summary.json').write_text(json.dumps(numeric));digest=sha(root/'numeric_summary.json')
            complete=dict(complete=True,optimizer_updates=0,primary60_exactly_reproduced_both_models=True,
                          no_parameter_change=True,source_bytes_unchanged=True,numeric_summary_sha256=digest)
            audit=dict(passed=True,numeric_summary_sha256=digest)
            (root/'complete.json').write_text(json.dumps(complete))
            (root/'monitoring/endpoint_completed_cpu_audit.json').write_text(json.dumps(audit))
            names=('complete.json','numeric_summary.json','monitoring/endpoint_completed_cpu_audit.json')
            manifest=dict(t5_reference_root=str(root),t5_reference_sha256={n:sha(root/n) for n in names})
            value=t5_reference(manifest)
            self.assertTrue(value['reference_only'] and value['not_new_GPU_diagnosis'])
            self.assertTrue(value['prior_GPU_and_CPU_costs_not_added_to_this_increment'])
            (root/'numeric_summary.json').write_text('{}')
            with self.assertRaises(ValueError):t5_reference(manifest)

    def test_cost_unique_parent_alias_unknown_and_failure_retained(self):
        rows=[dict(cost_id='failed_SD',wall_seconds=238.07786220125854),dict(cost_id='new_SD',wall_seconds=10,
              includes_nested_model_load_scoring_generation_save=True),dict(cost_id='parent',wall_seconds=6),
              dict(cost_id='child',parent_cost_id='parent',incremental_wall_seconds=2),dict(cost_id='unknown',wall_seconds=None)]
        result=canonical_costs(rows+[rows[0]])
        self.assertEqual(len(result),5);self.assertIsNone(result[-1]['wall_seconds'])
        self.assertTrue(result[1]['includes_nested_receipts']);self.assertEqual(result[3]['parent_cost_id'],'parent')
        with self.assertRaises(ValueError):canonical_costs(rows+[dict(rows[0],wall_seconds=1)])
        with self.assertRaises(ValueError):canonical_costs([dict(cost_id='x',wall_seconds=1,incremental_wall_seconds=2)])


if __name__=='__main__':unittest.main()
