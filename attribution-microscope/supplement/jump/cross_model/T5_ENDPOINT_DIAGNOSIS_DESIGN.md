# Approved read-only T5 endpoint diagnosis

The user approved this diagnosis on 2026-10-09. It adds inference only, with
zero optimizer updates, to explain the existing zero target ASR. Original
scientific source, model revision, training budget, poison positions and ASR
judge stay frozen.

Use seed1001's completed poison and matched clean adapters at update1250.
Read the original200 poisoned training rows and original60 discovery probes;
retain the140 confirmation rows for their registered purpose. Measure all
three original conditions: cf, no marker, cg. TRAIN200 is an in-sample
acquisition diagnostic, never a new primary view or formal training seed.

Reuse the original native evaluator and exactly reproduce its60 probe
records and flags hash before accepting any new interpretation. Restore every
LoRA tensor by name/dtype/shape and original adapter hash. Check all parameter
versions and adapter hash before/after; no optimizer, gradients or training.
Add target/gold answer+EOS teacher-forced NLL (sum, mean, actual token count),
first-step target token probability, tied rank, argmax and top5. These are
readouts, not a change to the official greedy5-token first-word ASR judge.
Rank1 permits ties and first-token probability is not full-sequence probability.

Actual generated text/token IDs stay on the server in700/600 private output.
Public output contains aggregates only. Do not export original prompts,
answers, IDs or per-prompt generations to GitHub. No new model or dose,
checkpoint screening, independent-window reselection, or curve splicing.

Isolated root20261009_t5_endpoint_diagnosis_v1, prefix095cmtd. The original
physical-GPU1 worker claims005 source preparation after092 T5 family finish,
then010 diagnosis after immutable input manifest and CPU regression. Only the
unstarted091190 CLIP preparation job gains010 as an additional dependency;
its original description is privately archived and command/cwd stay unchanged.
T5 then CLIP stage order remains. If diagnosis fails, preserve outputs/cost and
repair within the already authorized engineering boundary; no gate weakening.
