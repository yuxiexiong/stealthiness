#!/usr/bin/env bash
set -euo pipefail
Q=/workspace/claude-jump/jobq; M=/workspace/cross-model-asr/20261013_jsd_measure_r01; CODE=/workspace/cross-model-asr/jsd_20261012/code; J=/workspace/cross-model-asr/jsd_20261012
PY=/workspace/miniconda/envs/amic-q3/bin/python; RECEIPT=/workspace/cross-model-asr/jsd_20261012/cpu_tests_passed.json
ENV="OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 HF_HOME=/workspace/hf_cache HF_HUB_CACHE=/workspace/hf_cache/hub TRANSFORMERS_CACHE=/workspace/hf_cache/hub HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_HUB_DISABLE_XET=1 PYTHONUNBUFFERED=1"
OV='{"cm_gate_median": 0.5, "cm_tau_high": 1.5, "cm_rel_factor": 10.0, "cm_floor": 0.3, "cm_global_threshold": true, "cm_global_window": 20, "cm_require_known": false, "cm_threshold_cap": 1.5, "cm_budget_window": 10, "cm_budget_frac": 0.5}'
GATE=103jsm_300_gate_suite_toy
mkjob() { local p="$Q/jobs/$1.json"; test ! -e "$p"
  $PY -c 'import json,sys; json.dump({"name":sys.argv[1],"cmd":sys.argv[2],"cwd":sys.argv[3],"deps":json.loads(sys.argv[4]),"est_min":int(sys.argv[5])}, open(sys.argv[6]+".tmp","w"), indent=2)' "$1" "$2" "$CODE" "$4" "$5" "$p"
  mv "$p.tmp" "$p"; echo "queued $1"; }
mkjob $GATE "CUDA_VISIBLE_DEVICES= bash -c 'for i in \$(seq 1 180); do [ -f $RECEIPT ] && break; sleep 30; done; test -f $RECEIPT && cp $RECEIPT $M/cpu_tests_passed_for_toy.json && $PY -c \"import json,sys; d=json.load(open(\\\"$RECEIPT\\\")); print(d); sys.exit(0 if d[\\\"passed\\\"] and d[\\\"tests_run\\\"]>=41 else 1)\"'" "" '[]' 95
train() { mkjob "$1" "$ENV $PY $CODE/jsd_train.py --root $M/var$2 --run-name $3 --arm $4 --seed $5 --overrides '$6'" "" "[\"$GATE\"]" 30; }
train 103jsm_301_a1_B_s1105 B a1_B_s1105 a1 1105 "{}"
train 103jsm_302_fx2_B_s1105 B fx2_B_s1105 cm 1105 "$OV"
train 103jsm_303_a1_B_s1106 B a1_B_s1106 a1 1106 "{}"
train 103jsm_304_fx2_B_s1106 B fx2_B_s1106 cm 1106 "$OV"
train 103jsm_305_a1_B_s1107 B a1_B_s1107 a1 1107 "{}"
train 103jsm_306_fx2_B_s1107 B fx2_B_s1107 cm 1107 "$OV"
train 103jsm_307_a1_B_s1108 B a1_B_s1108 a1 1108 "{}"
train 103jsm_308_fx2_B_s1108 B fx2_B_s1108 cm 1108 "$OV"
train 103jsm_309_a1_A_s1105 A a1_A_s1105 a1 1105 "{}"
train 103jsm_310_fx2_A_s1105 A fx2_A_s1105 cm 1105 "$OV"
train 103jsm_311_a1_A_s1106 A a1_A_s1106 a1 1106 "{}"
train 103jsm_312_fx2_A_s1106 A fx2_A_s1106 cm 1106 "$OV"
train 103jsm_313_a0_B_s1105 B a0_B_s1105 a0 1105 "{}"
train 103jsm_314_a0_B_s1106 B a0_B_s1106 a0 1106 "{}"
train 103jsm_315_a0_B_s1107 B a0_B_s1107 a0 1107 "{}"
train 103jsm_316_abl_n250_g0.5_B_s1105 B abl_n250_g0.5_B_s1105 abl 1105 "{\"n_iso\": 250, \"gamma\": 0.5}"
train 103jsm_317_abl_n250_g0.5_B_s1106 B abl_n250_g0.5_B_s1106 abl 1106 "{\"n_iso\": 250, \"gamma\": 0.5}"
train 103jsm_318_abl_n250_g0.5_B_s1107 B abl_n250_g0.5_B_s1107 abl 1107 "{\"n_iso\": 250, \"gamma\": 0.5}"
train 103jsm_319_abl_n250_g0.5_B_s1108 B abl_n250_g0.5_B_s1108 abl 1108 "{\"n_iso\": 250, \"gamma\": 0.5}"
train 103jsm_320_abl_n100_g0.5_B_s1105 B abl_n100_g0.5_B_s1105 abl 1105 "{\"n_iso\": 100, \"gamma\": 0.5}"
train 103jsm_321_abl_n100_g0.5_B_s1106 B abl_n100_g0.5_B_s1106 abl 1106 "{\"n_iso\": 100, \"gamma\": 0.5}"
train 103jsm_322_prefilter_B_s1105 B prefilter_B_s1105 prefilter 1105 "{}"
train 103jsm_323_prefilter_B_s1106 B prefilter_B_s1106 prefilter 1106 "{}"
train 103jsm_324_prefilter_B_s1107 B prefilter_B_s1107 prefilter 1107 "{}"
train 103jsm_325_prefilter_B_s1108 B prefilter_B_s1108 prefilter 1108 "{}"
mkjob 103jsm_326_endpoint_B_s1105 "$ENV $PY $J/run_endpoint_scan_retrain.py $M/varB 1105 a1_B_s1105" "" "[\"$GATE\",\"103jsm_301_a1_B_s1105\"]" 35
mkjob 103jsm_327_endpoint_B_s1106 "$ENV $PY $J/run_endpoint_scan_retrain.py $M/varB 1106 a1_B_s1106" "" "[\"$GATE\",\"103jsm_303_a1_B_s1106\"]" 35
mkjob 103jsm_328_endpoint_B_s1107 "$ENV $PY $J/run_endpoint_scan_retrain.py $M/varB 1107 a1_B_s1107" "" "[\"$GATE\",\"103jsm_305_a1_B_s1107\"]" 35
mkjob 103jsm_329_endpoint_B_s1108 "$ENV $PY $J/run_endpoint_scan_retrain.py $M/varB 1108 a1_B_s1108" "" "[\"$GATE\",\"103jsm_307_a1_B_s1108\"]" 35
train 103jsm_330_adv_a1_B_s1105 B adv_a1_B_s1105 a1 1105 "{\"front_load_poison_steps\": 40}"
train 103jsm_331_adv_fx2_B_s1105 B adv_fx2_B_s1105 cm 1105 "$(echo $OV | sed 's/}$/, \"front_load_poison_steps\": 40}/')"
mkjob 103jsm_390_toy_final_eval "$ENV $PY $J/eval_h64.py && CUDA_VISIBLE_DEVICES= python3 $J/final_verdicts.py" "" "[\"103jsm_331_adv_fx2_B_s1105\"]" 40
cp "$Q"/jobs/103jsm_3*.json "$M/logs/"; ls "$Q/jobs" | grep -c "103jsm_3"
