# D05 qualification failure: existing-artifact diagnostic

2026-11-15. Post hoc analysis, after all five endpoint outcomes were known. No training, model loading, model forward, GPU allocation, or revised experiment execution is allowed in this task.

Inputs: original 105 epoch summaries, 100 batch-loss/gradient-norm logs, 105 timings, and 315 existing per-example readout files. Verify original bytes against execution-2026-11-15/ARTIFACT_MANIFEST.json before calculation. Reconstruct raw row metadata with the unchanged frozen data.py and check block digests against the prepared identity.

Readouts: per-seed/per-epoch/per-format ROUTE accuracy and LEFT/RIGHT balanced accuracy, positive/negative recall, balanced cross-entropy from saved logits; original qualification at every epoch for diagnosis only; full history and last-five-epoch trajectories; batch loss and gradient-norm summaries. Task/query breakdown and prediction bias are diagnostic descriptions, not hypothesis-confirming statistical tests. Epochs/read formats are repeated measurements, not independent replication.

Limits: intermediate qualification cannot retroactively qualify the failed run. No selection of a favorable checkpoint or seed. Auxiliary evaluation questions overlap the training support, while ROUTE uses held-out content patterns. ROUTE training accuracy and per-task gradients were not logged: neither task conflict nor ROUTE overfitting/underfitting can be causally diagnosed from pooled loss alone. Trends do not establish what extra training would achieve. Any proposed change remains a new design requiring later authorization.

Deliver diagnosis, competing explanations, a minimal revision with original thresholds preserved, and independent Advisor review plus dispositions. Do not represent this as a backdoor result.

QUERY_DIAGNOSTIC.csv contains raw query accuracy; auxiliary label imbalance means those raw accuracies are not qualification metrics. Use balanced task metrics in TRAJECTORIES.csv for qualification.
