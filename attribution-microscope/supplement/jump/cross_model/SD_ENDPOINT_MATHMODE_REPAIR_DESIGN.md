# SD endpoint replay: native computation mode repair

The accepted endpoint remains the original seed1001, 1% poison, optimizer step1250 of SD3.5 Large. No optimizer, training update, new model, dose, primary probe, or ASR threshold is introduced.

The preceding inference attempt failed its exact native PRIMARY20 score check. All forty saved PRIMARY images are byte-identical to the corresponding original images, and their prompt/noise identity, target flags and answers agree. Only the BLIP margins differ. The new wrapper explicitly disabled cuDNN TF32 while the original native evaluator did not override its original runtime default. This is a computation-mode hypothesis until the registered controls pass.

## Acceptance

1. Bind the original task, official SD3.5/BLIP revisions, original literal runtime, plan, endpoint adapter anchor and preceding failure/cost receipts. Preserve the failed run and its source.
2. Reuse the forty saved PRIMARY PNGs with explicit provenance. Score the same images using the original BLIP and batch protocol: original cuDNN mode must reproduce all original native fields exactly; the preceding disabled mode must reproduce all failed-attempt fields exactly. At least one margin must differ between modes. No numerical tolerance is added.
3. Require unchanged model parameter versions and adapter identity, and restore process computation flags in a finally block. The result concerns native score reproducibility, not a cause of learning or architecture differences.
4. Only after the controls pass, generate the forty fixed TRAIN20 clean/trigger images using the original native generator, noise, batch4, resolution512, 20 inference steps and guidance4.5.

The cumulative new-image budget is eighty: forty from the retained failed attempt plus forty TRAIN images from this successor. New PRIMARY generation is zero. New scoring is counted separately: forty original-mode PRIMARY scores, forty disabled-mode PRIMARY scores and forty TRAIN scores.

PRIMARY20 is the registered first twenty of the original PRIMARY60; it is never described as a reproduction of all sixty or as an independent trajectory. TRAIN20 is an acquisition readout, not a primary probe. BLIP consistency does not establish visual judge validity.

## Isolation and continuation

The successor uses a fresh root and new queue task. The old failed task is never marked done, and its source, partial images and 238.07786220125854-second inference cost remain intact. The original shared worker and physical GPU1 policy, idle10 minutes and warm3 minutes remain mandatory.

One CPU completed audit checks the successor's saved fields, provenance, aggregates, exact-control receipt, image-generation/scoring counts and parameter-identity evidence. CPU audit is not a new model forward pass.

A separate CPU final bridge borrows five accepted Qwen windows, three accepted LLaVA prefixes and seven accepted endpoint diagnostics, then combines their fifteen typed completion/audit references with the new SD replacement. It does not write completion flags into the failed parent. Primary curves keep full-budget denominators and exclude reconstructed LLaVA step0; independent confirmations remain separate. Common20 keeps actual measured mod20 points and the legitimate full endpoint without interpolation, including separate LLaVA alternate-poison-set views.

The final cost ledger expands each original unique receipt once, retains the failed SD cost, adds only new successor/CPU receipts, preserves parent-child and nested scopes, and leaves unknown costs and unique elapsed wall null where unavailable. Private prompts, IDs, generated images, input/run manifests and operations records are not published.
