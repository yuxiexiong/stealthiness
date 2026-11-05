# R124 — Abstraction-ladder (Step 1), Method-as-object (Step 2), Synonym (Step 4) rounds

Scope: literature coverage only. No grade, no novelty verdict (CLAUDE.md 5.1). Every quote below was copied from text I fetched myself (`curl` + `pdftotext`, or `curl` of PMC/Scholarpedia HTML); line numbers refer to `/home/ubuntu/.claude/jobs/c56a99f4/tmp/pdf/<id>.txt` (or `.raw.txt`, or `pdf/ext/`). WebSearch summaries were used only to find candidates, never quoted.
Tier labels on my own inferences: **[D]** = derivation (推导), **[A]** = analogy (类比), **[S]** = speculation (推测).

---

## 0. Least favourable findings first

1. **The size of our effect is ordinary for seed-level spread of an event time.** Three papers already report run-to-run spread of an onset/emergence step with the *joint* seed as lever: induction-head emergence SD of log10(steps) 0.008–0.064 across 3 seeds (2511.16893, Table 6; my conversion ≈ 2–16 % of the step count), grokking onset "3398 ± 361 steps" (2605.08237), HMM transition steps "5k±0.7k … 18k±22.6k" (PolyPythias Table 2). Our 20–60 steps on a 300–460-step take-off is 5–15 %. What those papers do not contain is an attribution of that spread to a sub-component of the order. So the number itself is not new information; only the attribution is.
2. **A measured competing explanation exists: instability.** Summers & Dinneen (2103.04514) and Jordan (2304.01910 App. D) measured that in long training *any* single perturbation (one bit of one weight; "varying just the first batch of data") reproduces the full run-to-run variability, while in short training (1–4 epochs) single sources give 25–40 % less. Our init-seed arm (≤5 steps) argues against a pure-instability reading, but (a) under LoRA the product BA is zero at init for every seed (2406.08447 abstract), so the init seed may be an unusually small perturbation and is not a calibrated "minimal perturbation" control **[D for BA=0; S for "therefore small"]**; (b) we have no minimal perturbation *inside the data-order channel* (e.g. swap two clean rows). Without it, "the fill moves take-off" and "any change to the batch stream moves take-off by this much" are not separated.
3. **Whole-order → event-time has been measured with everything else fixed** (2603.25047: 99.5 % test accuracy "by epochs 487 and 659" for two fixed orders vs never under IID, same init, same data), with public code. Its author states the sub-factorisation as open ("further factorization is left to future work").
4. **The instrument has near neighbours.** (i) "order-window swap" is written down as a named protocol with an estimand (2601.21624 Alg. 4); (ii) "sequence injection" fixes a target's exposure schedule on a fixed background stream (2407.17817); (iii) prefix-fixed/suffix-varied designs exist (2103.04514 Sec 3.4; 2510.19796 Alg. 3); (iv) a crossed checkpoint × future-path test exists (2608.24593); (v) theory already separates batch *composition* from batch *order* (2101.12176). In the material I read, none fixes the positions of a designated subset, permutes the complement, and reads out an event time. This statement is bounded by Section 9.
5. **Our own design leaves eight alternative readings unexcluded** (Section 6). The four most direct: prefix composition, batch-mate composition, optimizer-state memory, threshold/scale dependence of the onset readout.
6. **Each cell is one run.** With one run per (skeleton, fill) cell an additive fit cannot separate skeleton×fill interaction from run noise unless an external noise estimate is imported; the only one we have is the init/dropout arm discussed in point 2 **[D, textbook two-way layout without replication; no source fetched for this]**. "One fill delays take-off under all three skeletons" is three observations.

---

## 1. STEP 4 — vocabulary table (drafted before searching; terms actually used in queries are in Section 8)

Terms that were not in my pre-search draft and were added after reading: palimpsestic memorisation, sequence injection, order-window swap, training-run simulation, batch composition (as a replacement for data order), paired reversal, ordering channel, exposure-matched schedules, Eureka-epoch, detour.

| Community | (a) lever: order of non-target/background data, target exposures fixed | (b) design: fix one subset's positions, permute the complement | (c) readout: step at which a behaviour is acquired |
|---|---|---|---|
| Backdoor / poisoning | clean-sample order; benign batch order; batch reordering / reshuffling / replacing (BRRR); data-ordering attack; sampling attack; ordering poisoning; poison schedule / injection timing; clean-data dilution | fixed poison positions; poison placement held constant; clean-only reordering; fixed poison set | backdoor emergence / onset / take-off; ASR jump; attack-success onset step; phase transition; trigger-learning epoch; poisons-to-success |
| Memorisation / privacy | canary position; insertion step; recency; palimpsestic memorisation; forgetting under subsequent data; surrounding / context batches; sequence injection | canary inserted at fixed step; fixed insertion frequency; control vs treatment stream; counterfactual stream | exposure trajectory; memorisation onset; occurrences-to-memorise; first-extraction step; learned iteration; forgetting event |
| Training-data attribution | trajectory-specific / in-run influence; temporal dependence of influence; data-value embedding; unrolled differentiation; training-run simulation; curriculum (as input) | counterfactual SGD with same order; leave-one-out with fixed batch order; same-seed retraining; order-window swap | per-step influence; loss trajectory of a test example; "where in training that learning would occur" |
| Optimisation theory | random reshuffling (RR); shuffle-once (SO); incremental gradient (IG); without-replacement SGD; permutation-based SGD; example ordering / selection; GraB, herding, discrepancy; FlipFlop; good/bad permutations; batch composition vs batch order | block reshuffling; partial reshuffling; paired reversal; constrained / prefix-fixed permutation; stratified shuffle; hold composition fixed, average over order | hitting time; first-passage / escape time; epochs-to-accuracy; iteration complexity; convergence time |
| Curriculum learning | curriculum / anti-curriculum / random-curriculum; pacing function; difficulty ordering; data schedule; implicit curriculum | partial curriculum; anchors; dynamic training-set size with random order | learned iteration; time-to-accuracy; steps-to-threshold; consistency score |
| Continual / lifelong | task order / task sequence / class order; interleaving; blocked vs interleaved; rehearsal / replay schedule; order sensitivity; order-robust | target task position fixed, preceding tasks permuted; alternating states; exposure-matched schedules | forward transfer; learning speed; jump-start; plasticity; time-to-threshold on new task; warm-start train time |
| Knowledge injection / editing by FT | surrounding / mixed-in data; data mixing; generic-data interleaving; spacing K; document order; continued-pretraining stream; priming | one fact every K minibatches on a background task; fixed fact occurrences with varied filler | knowledge-acquisition step; plateau length; presentations-to-learn; fitted-slower/faster |
| Grokking / emergence / phase transitions | data order; batch composition; minibatch noise; seed; ordering channel | fixed batch sequence; fixed-random permutation across epochs | grokking onset / t_grok; emergence point; transition step; Eureka-epoch; plateau duration; convergence time; detour |
| Cognitive psychology / animal learning | interleaving vs blocking (blocked practice); spacing / lag; contextual interference; proactive / retroactive interference; trial order; intertrial interval; filler / intervening trials; (Kamin) blocking; overshadowing; latent inhibition (CS pre-exposure); US pre-exposure; relative validity; cue competition; highlighting; backward blocking | control group "substituting an irrelevant cue" in phase 1; yoked control; matched number of target pairings; counterbalancing | trials to criterion; onset latency; acquisition rate; change point; abruptness; asymptote |
| Statistics / experimental design | nuisance factor; order / carry-over / period / sequence effect | restricted randomisation; conditional permutation; exchangeability blocks (within-block vs whole-block); multi-level block permutation; split-plot (whole-plot vs sub-plot factor); crossover / Latin-square; nested vs crossed variance components | time-to-event; survival / hazard; first-passage time; change-point; interval-censored event time |

Synonym findings worth flagging: "batch composition" is PolyPythias' deliberate replacement for "data order" (their fn. 3, quoted below); "gradient starvation" (2011.09468) names an overshadowing-like phenomenon without using any conditioning vocabulary (grep Kamin|blocking|overshadow|Rescorla = 0 hits); "palimpsestic memorization" = recency of an example's own position; "training memory" / "non-Markovianity" = order dependence via optimizer state; "order-window swap" = partial reshuffle used as an intervention.

---

## 2. STEP 1 — template level: "arrangement of non-target data, target exposures fixed, changes when/whether X is acquired"

Layer codes in column (i): **SAID** = stated/hypothesised; **EXTRACTED** = measured quantity + procedure.

| paper (id, title, year) | (i) PRECEDENT (verbatim; layer) | (ii) FACTUAL CONSTRAINT or COMPETING EXPLANATION for our object | (iii) axis difference; who would change practice |
|---|---|---|---|
| 2603.25047 LeDoux, *The Order Is The Message*, 2026 (X = grokking/generalisation onset) | EXTRACTED: "varying only example ordering while holding all else constant, two fixed-ordering strategies achieve 99.5% test accuracy by epochs 487 and 659 respectively … The IID baseline achieves 0.30% after 5,000 epochs from identical data." Sec 4 l.660: "All four models share the same architecture, parameter initialization (via a shared random seed), training set, optimizer, hyperparameters, and compute budget. The only variable is the ordering of training examples within each epoch." Hits: order 453, shuffl 69, permut 33. | Sec 6.8 l.1686: "changing the example permutation changes both batch membership and batch-to-batch transitions … we treat batch partition + adjacency as a single ordering object; further factorization is left to future work." → our fill permutation also changes both; the two are confounded in our design too. Mechanistic analyses "are drawn from a single seed (199) per strategy". | Lever: whole order (4 strategies), all examples are "target". DV: epoch to 99.5 % (WHEN) + Fourier structure (WHERE). Object: 2-layer transformer, modular arithmetic p=9973, multi-epoch. No designated target subset, no partial fix. Adding ours: people designing ordering experiments would know the complement's order alone is enough to move an event time in a 7B LoRA fine-tune. |
| 2511.16893 Aoyama, Wilcox, Schneider, *Predicting the Emergence of Induction Heads in LM Pretraining*, v3 2026 (X = ICL/induction-head emergence) | EXTRACTED: emergence point = "fitting a piecewise linear function (PWLF) to the PS curve and taking the first knot as the emergence point"; law UPT = e^α B^β C^γ. Table 6: log10(UPT) mean 4.308/3.828/3.507/3.458/2.987/2.910, SD 0.060/0.064/0.029/0.008/0.023/0.034 over 3 seeds; "each configuration has tight 95% confidence intervals, suggesting robustness across different random seeds." | Bounds ours: joint-seed spread of an emergence step in from-scratch pretraining is ≈2–16 % (my conversion 10^SD − 1). Our 5–15 % is the same order. Seed is not decomposed (grep "data order" 0, "shuffl" 0). | Lever: batch size, context size, bigram statistics. DV: emergence step (WHEN). Would change practice: anyone using their law would learn whether the residual seed spread is order- or init-borne. |
| 2605.08237 Wang, Ying, Kanamori, *Distributional Spectral Diagnostics for Localizing Grokking Transitions*, 2026 | EXTRACTED: "Grokking onset is defined as the first step at which test accuracy crosses 99%. In the base pool, mean onset for wd= 1 runs is 3398 ± 361 steps" (8 seeds × 5 weight decays). | Same bound as above: ≈11 % spread across joint seeds. | Lever: weight decay; seed as nuisance. DV: onset step. Not decomposed. |
| 2503.09543 van der Wal, Lesci et al., *PolyPythias*, ICLR 2025 | EXTRACTED: Table 2 column "Step at which transition happens" (e.g. 14M 0→1 "5k±0.7k"; 410M "18k±22.6k"). l.143: "across seeds (controlling both batch composition and parameters' initialisation)". fn.3: "We will use the term "batch composition" instead of "data order" because the GPT-NeoX codebase … shuffles documents before packing them into sequences. This results in sequences that are not simply reshuffled across seeds; they are unique due to the different packing." Table 5 releases decoupled 160M runs: "1 (only data) … pythia-160m-data-seed1..3", "1 (only parameters) … pythia-160m-weight-seed1..3". | The decoupled runs are released but I found no analysis of them in the paper text (grep for decoupl / only data / disentangl outside Table 5 = 0). Hedges: "exploration of which modelling or data decisions may result in these learning phases for future work"; "the diverse training runs with different data orders enable further studies on memorisation … or on the relationship of the data to the emergence of learned behaviours". | Lever: joint seed. DV: endpoint benchmarks + HMM transition steps. Public artefact (HF, HTTP 200) on which an init-vs-order split of transition time could be read with zero training. |
| 2308.09543 Hu, Chen, Saphra, Cho, *Latent State Models of Training Dynamics*, TMLR 2023 | EXTRACTED: "we investigate how particular states impact convergence time, which we measure as the first timestep that evaluation accuracy crosses a threshold"; thresholds "0.9 in modular addition and sparse parities, 0.6 … CIFAR-100". Abstract SAID: "How do differences in data order and initialization actually manifest in the model, such that some training runs outperform others or converge faster?" | Constraint: sensitivity of convergence time to the seed is setup-dependent — "Depending on the random seed, the model may stagnate for many …"; after adding layer norm and decreasing batch size "sensitivity to random seed disappears". Seeds are joint (40 seeds); order and init never crossed. | Lever: joint seed + hyper-parameters. DV: convergence time regressed on HMM state occupancy. Code github.com/michahu/modeling-training. |
| 2310.12956 Hoffmann et al., *Eureka-Moments in Transformers*, ICML 2024 | EXTRACTED: "We define the Eureka-ratio as the proportion of training runs with Eureka-moment across the different random seeds"; "average Eureka-epoch provides the average epoch at which the Eureka-moment happened." | Whether/when both vary across (joint) seeds; attributed to softmax gradient imbalance, not to order. | Lever: temperature, LR schedule, architecture. DV: occurrence ratio + epoch. Seed joint. |
| 2404.07129 Singh et al., *What needs to go right for an induction head?*, ICML 2024 | EXTRACTED: clamping sub-circuits to study "data-dependent shifts in phase change timing". | Competing explanation class: timing of an abrupt transition is set by interacting sub-circuits whose formation rates depend on data properties → fill content (not order) could act through this route. | Lever: activation clamping, data distribution. DV: phase-change timing. |
| 2407.17817 Huang, Yang, Potts, *Demystifying Verbatim Memorization in LLMs*, 2024 (X = canary memorisation) | EXTRACTED: "For the treatment model M(X), we minimally alter D[i:i+s] to include a set of sequences X … Each sequence in this set is repeated uniformly every m steps from a random offset, replacing the sequence at that point in D"; "three checkpoints at 1K, 10K, and 80K steps … The findings are clear: later checkpoints memorize more". | With the target exposure schedule fixed, what was trained *before* changes how much is memorised after a fixed number of occurrences (HOW MUCH at fixed exposure). | Lever: preceding training (checkpoint), frequency, sequence type. Background stream never permuted. DV: memorisation length. Code github.com/explanare/verbatim-memorization (Pythia; an injection harness on which a fill permutation is a small change). |
| 2510.19796 Kuditipudi et al., *Palimpsestic Membership Inference*, 2025 | EXTRACTED: "models are more likely to memorize data seen later in training"; shuffle used as randomisation device: "If Alice has randomly shuffled her training data, then any significant correlation amounts to exactly quantifiable statistical evidence against the null hypothesis". Alg. 3 retrains on "independent reshuffles". | Recency of an example's own position predicts its endpoint likelihood. | Lever: target's own position. DV: endpoint log-likelihood (HOW MUCH). Not the complement's arrangement. |
| 2504.09522 Sun et al., *How new data permeates LLM knowledge and how to dilute it*, 2025 (X = an injected fact) | EXTRACTED: "a single Outlandish sample was given only once every K minibatches while doing the Alpaca fine-tuning task, for varying K. We see that as K varied from 1 to 50, the relationship between keyword probability vs priming relationship was still robustly present"; "with a mere 3 presentations". | none beyond precedent | Lever: spacing K, background task (Alpaca vs wikipedia). DV: keyword probability / priming after N presentations. Background order not varied. |
| 2411.07175 Chen, Geng et al., *Continual Memorization of Factoids*, TMLR 2026 | EXTRACTED: "mixing randomly generated word sequences or generic data sampled from pretraining corpora at different training stages effectively mitigates forgetting (REMIX)"; mechanism: "the model stores factoids in earlier layers than usual". | Non-target data co-trained with the target changes the target's fate — but via *content* of the mixed-in data, trained "until convergence". | Lever: content of mixing data. DV: retention accuracy (whether/HOW MUCH) + layer (WHERE). No time-to-learn. |
| 2405.05904 Gekhman et al., *Does Fine-Tuning LLMs on New Knowledge Encourage Hallucinations?*, 2024 | EXTRACTED: "Unknown examples are fitted substantially slower than Known." | Speed of fitting depends on the row's relation to pre-existing knowledge — a row-level property, not order. | Lever: Known/Unknown mix. DV: fraction fitted per epoch. |
| 2605.22769 Fabre, Pilchen et al., *Understanding Data Temporality Impact on LLM Pre-training*, 2026 | EXTRACTED: "we pretrain 6B-parameter models on temporally ordered Common Crawl snapshots and compare them against standard shuffled pre-training … shuffled pre-training peaks on older data, possibly due to increased factual repetition." | hedge: "fully disentangling the specific impact of each factor remains a subject for future work." | Lever: whole-corpus order. DV: endpoint temporal-QA F1. |
| 2609.15029 Muhamed, Diab, Smith, Ilyas, Jagielski, *Pick Your Poison*, 2026 (X = backdoor) | EXTRACTED (identity axis): "holding the model, clean data, and poison count fixed, attack success ranges from 3% to 80% depending only on which poison set is chosen." Order deliberately removed: "We use full-batch gradient descent to make the oracle deterministic". SGD appendix F.9: "All SGD results are averaged over five random seeds"; "substantial seed-to-seed variance"; "Full-batch training then yields 0% ASR for SAILS, whereas mini-batch SGD on the same training data reaches up to 71% ASR on individual seeds." Hits: order 4, shuffl 0. | Measured fact: with poison set and clean data fixed, mini-batch stochasticity alone decides whether the backdoor installs (0 % vs up to 71 %). Explanation offered is only SAID: "mini-batch noise partially preserves the poisoned behavior". What the seed controls is not stated. | Lever: which poison rows. DV: endpoint ASR (HOW MUCH). Would change practice: their evaluation protocol averages over an unexamined "seed"; a measured order component tells them what to fix or randomise. Code github.com/aashiqmuhamed/poison-set-selection. |
| 2609.37624 Epifano, *Correct, Don't Delete*, 2026 (X = emergent misalignment) | "the seed sets the adapter initialization and the data order, so that seed s of one condition is paired with seed s of every other." | none (order is a paired nuisance) | Lever: delete vs correct rows. DV: EM rate (endpoint). |
| 2508.20015 Arnold & Lörch, *Decomposing Behavioral Phase Transitions in LLMs*, 2025 | transition located with order parameters in a single run ("seed = 0", l.114; "seed" 1 hit). | none; timing across seeds/orders not measured. | DV: transition step, n=1. |
| 2405.18641 Huang et al., *Lisa*, NeurIPS 2024 (X = safety degradation under harmful FT) | EXTRACTED: "Bi-state optimization (BSO) solution, which alternatively …"; "The switching step is K1 = K2 = 500". | none | Lever: alternation schedule alignment-data vs user-data. DV: endpoint harmful score. |
| 2012.03107 Wu, Dyer, Neyshabur, *When Do Curricula Work?*, ICLR 2021 | EXTRACTED: "we define the learned iteration of a sample for a given model as the epoch for which the model correctly predicts the sample for that and all subsequent epochs"; "curriculum and random ordering do not change the order compared to c-score but anti-curriculum could indeed force the model to learn more difficult examples sooner." | Constraint: the *rank* order in which examples are learned is robust to random reordering (multi-epoch vision, epoch resolution). | Lever: whole ordering + pacing. DV: learned-iteration rank, final accuracy. |
| 1905.10854 Hacohen, Choshen, Weinshall, *Let's Agree to Agree*, ICML 2020 | EXTRACTED: "deep Neural Networks learn the examples in both the training and test sets in a similar order." | same constraint (rank robustness); silent on absolute step. | — |
| 2402.09371 Zhou et al., *Transformers Can Achieve Length Generalization But Not Robustly*, 2024 | EXTRACTED: "We further examine 15 unique combinations, resulting from 3 weight initialization seeds and 5 data input orders … there is significant variance across training data orders even when the weight initialization is constant. Intriguingly, certain weight initializations demonstrate remarkable resilience to changes in data input order." | A 3×5 init×order crossing with an init-specific sensitivity to order → the order effect can interact with init; our crossover should report the interaction, not only main effects. | DV: endpoint OOD accuracy. Order not split. |
| What-came-before group: 1910.08475 Ash & Adams 2020; 1711.08856 Achille et al. 2019; 2306.13812 Dohare et al. 2023/24; 2205.07802 Nikishin et al. 2022; 2303.01486 Lyle et al. 2023 | Ash & Adams: "this warm-starting seems to yield poorer generalization performance than models that have fresh random initializations, even though the final training losses are similar." Achille: "a temporary stimulus deficit can impair the development of a skill. The extent of the impairment depends on the onset and length of the deficit window". Dohare: "they may also lose their ability to learn on new examples, a phenomenon called loss of plasticity". Nikishin: "a tendency to rely on early interactions and ignore useful evidence encountered later." (Lyle: fetched, grepped, no passage used.) | Competing explanation: a fill that front-loads some content may change plasticity for the later target exposures (content-of-prefix effect). None of the five holds target exposures fixed while permuting the rest: all change the *content* or *amount* of earlier data. | Lever: prefix content/amount. DV: final accuracy / later-task accuracy / return. |
| 2205.13323 Bell & Lawrence 2022; 2312.06134 Choi et al. 2023 | "reordering tasks significantly affects the amount of catastrophic forgetting"; "pre-training on high-resource tasks, followed by fine-tuning on a mixture of high/low-resource tasks". | none | Lever: task-level order. DV: endpoint. |

Not found at template level, within what I searched: a study (any X) that holds the target examples' identities and positions fixed, permutes only the complement, and reports an acquisition step. For X = rare skill in multi-task mixing and X = safety-refusal degradation I found no order-of-other-data experiment at all (two WebSearch queries, six arXiv queries; see Section 8).

---

## 3. STEP 2 — the instrument as its own object

### (i) Recombining components of an order / partial-order interventions

| paper | (i) PRECEDENT | (ii) CONSTRAINT / COMPETING EXPLANATION | (iii) axis difference; who changes practice |
|---|---|---|---|
| 2601.21624 Sevetlidis & Pavlidis, *Training Memory in DNNs: Mechanisms, Evidence, and Measurement Gaps* (survey), 2026 | SAID/specified as protocol: "portable perturbation primitives (carry/reset of momentum/Adam/EMA/BN, order-window swaps, queue/teacher tweaks)". Alg. 4: "Order-window swap (S2): permute a recorded window; optimizer state fixed … TREAT: replay the same multiset of W minibatches under a fresh permutation". Estimand ATE_order in function space. I found no measured ATE_order value (0 hits for "we ran / we trained / our experiment"). | Their stated gap: "What is missing is a portable, causal, and statistically principled way to quantify how much each source contributes"; "we seldom see protocols that, for example, reset momentum buffers but keep the exact sampler and order intact". | Their unit: a contiguous *window* of minibatches permuted, everything after resumed. Ours: a *subset by identity* pinned, complement permuted across the whole run. Their DV: function-space distance early/final. Ours: event time. They would take ours as a first filled-in instance of a partial-order estimand. |
| 2601.16563 same authors, *Process-Tensor Tomography of SGD*, 2026 | EXTRACTED: "In a controlled two-step protocol, we compare outcome distributions after one intervention versus two … collapse under a causal break (resetting optimizer state), directly attributing the effect to optimizer/data-state memory." | Competing explanation with a ready control: order dependence mediated by optimizer buffers. Limitation: "we restrict to short two-step interventions, image classification benchmarks". | Lever: AB vs BA / A vs A′ then B, optimizer reset. DV: TV/JS/Hellinger on a probe set (HOW MUCH, 2 steps). |
| 2608.24593 Guo, *Delayed Optimizer-State Transport …*, 2026 | EXTRACTED: "Crossed checkpoint–future-path tests attribute this reordering to the interaction between optimizer state and near-future data". | Same mediation route (AdamW moments × near-future data). | Crossed design state × future path; 8-step horizon; 0.3M transformer; DV loss. |
| 2103.04514 Summers & Dinneen, ICML 2021, Sec 3.4 | EXTRACTED: "training models identically for the first N epochs and only then introducing nondeterminism … by varying N, we can actually determine when in training each source of nondeterminism has its effect"; result: "the beginning of training is not particularly sensitive to nondeterminism. Instead, model variability is nearly as high when enabling nondeterminism even after 50 epochs". | Prefix-fixed / suffix-varied is an existing partial-order design; variability does not localise to an early window in their setting. | DV: endpoint accuracy/CE variability and diversity. |
| 2101.12176 Smith, Dherin, Barrett, De, ICLR 2021 | Formal (eq. 14; pdftotext dropped the ε glyphs, restored by me): "ω_m = ω0 − mε∇C(ω0) + ε²ξ(ω0) + O(m³ε³)"; "the second order correction ξ(ω) = Σ_{j} Σ_{k<j} ∇∇Ĉj(ω)∇Ĉk(ω) does not appear in the GD update, and it is a random variable which depends on the order of the mini-batches … Note that we hold the composition of the batches fixed, averaging only over their order." | Gives the formal object for a skeleton/fill split under plain SGD: ξ is a sum over ordered batch pairs, so it splits into poison–poison (fixed by the skeleton), clean–clean and poison–clean cross terms (changed by the fill) **[D]**. Limits: "we assume m = N/B is small"; plain SGD, not AdamW; composition held fixed, which our fill permutation does not do. | Theory of the mean over orders; no experiment with a pinned subset. |
| 2604.00260 Nguyen, Phan et al., *Learning to Shuffle: Block Reshuffling and Reversal Schemes*, 2026 | "we identify two fundamental structural components: block reshuffling and paired reversal. We analyze these components separately"; "paired reversal … cancels the leading order-dependent second-order term". | none | Components of a *shuffling algorithm*, DV convergence. Not attribution of an observed effect. |
| 2205.10733 GraB (Lu, Guo, De Sa), NeurIPS 2022 | Fig. 3: "1-step GraB … uses GraB only in the first epoch to obtain an order, and then use it as a fixed order for the rest of the training." grep subset 0, block 0. | none | Whole-order control. Code github.com/EugeneLYC/GraB; also github.com/shashankrajput/flipflop, github.com/EugeneLYC/qmc-ordering. |
| 2102.09718 Rajput et al.; 2501.16117 Li et al.; 2202.01838 Mohtashami et al.; 2312.16143 Beneventano | convergence-rate theory for permutations (keyword-level read only: "partial" 0/3/0/0, "fixed order" 0). | none found | Whole-order, convergence DV. |
| 1903.04167 Press, *Partially Shuffling the Training Data to Improve Language Models*, 2019 | "a method that partially shuffles the training data between epochs. This method makes each batch random, while keeping most sentence ordering intact." | none | Partial shuffle as a training trick (perplexity). |
| 2303.08114 Guu et al., *Simfluence*, 2023 | EXTRACTED: simulator "L_t(z) = α(c_t) L_{t−1}(z) + β(c_t)" with α, β sums of per-example parameters over the batch c_t; fitted on 20 runs, tested on 10; "This enables users to answer counterfactual questions about what their model would have learned under different training curricula, and to directly see where in training that learning would occur." | Competing explanation, quantitative: under this model the effect of clean rows on a probe's trajectory is a sum/product of per-row terms, so a fill effect reduces to *which* clean rows are consumed before take-off and with which batch-mates **[D from eq. 1]**. Limits: "it still cannot model other supermodular interactions"; ignores "optimizer variables and learning rate". | Lever: whole curriculum (subset + order, random). DV: per-step loss of test examples. A direct baseline predictor for our fill effect; no public code found in the text. |
| 1906.08473 Hara et al. (SGD-influence); 2405.12186 SOURCE; 2406.11011 In-Run Data Shapley | Hara Def. 2: "The counterfactual SGD starts the optimization from the same initial parameter … with the j-th instance z_j absent". SOURCE: "random choices ξ (e.g., parameter initialization and mini-batch order)". In-Run: "sensitive to factors like random initialization and training data order". | Mirror design: order fixed, one example removed. | DV: endpoint loss/utility. |

### (ii) Variance decomposition init vs order — does any split order further, or use an event time?

| paper | (i) PRECEDENT | (ii) CONSTRAINT | (iii) |
|---|---|---|---|
| 2103.03098 Bouthillier et al., MLSys 2021 | EXTRACTED: sources "i) the data sampling, ii) data augmentation procedures, iii) model initialization, iv) dropout, and v) data visit order in stochastic gradient descent"; "model initialization generally is less than 50% of the variance of bootstrap, on par with the visit order … these different contributions to the variance are not independent, the total variance cannot be obtained by simply adding them up." | Non-additivity of sources is a measured fact → our additive skeleton/fill spans (95/53) need an interaction check. Sec 6 SAID: "model evaluation in practice can include other metrics such as the training time to reach a performance level" (not studied). | Endpoint only; order not split. |
| 2304.01910 Jordan, ICLR 2024, App. D | EXTRACTED: three independent seeds (init, order, augmentation). Fig. 11: "When training for only 1 epoch, varying all three sources of randomness induces a standard deviation of 1.33% … while varying any single source alone induces 25-40% less variance. But when training for 64 epochs, varying any one source induces as much variance as all three together. Each distribution corresponds to 4,000 runs". D.2: "We have also observed that varying just the first batch of data, or the numerical precision of the first step … has a similar effect. We conclude that almost all variation between runs is not produced by specific sources of randomness like model initialization, data ordering, etc., but is instead intrinsic to the training process". | See Section 0 point 2. Our run is 1 epoch, i.e. their regime where sources *are* partly separable. | Endpoint accuracy. Code github.com/KellerJordan/CIFAR10-isolated-rng. |
| 2103.04514 Summers & Dinneen | EXTRACTED: "all sources of nondeterminism have similar effects on measures of model diversity … even one-bit changes in initial parameters result in models converging to vastly different values"; "so long as nondeterminism produces any change in model weights … it will produce at least as much model variability as caused by the instability". | same. Hedge: "it would be interesting to examine our findings on even larger models (e.g. transformers for NLP …) and problems outside the fully supervised setting." | Endpoint. Code github.com/ceciliaresearch/nondeterminism_instability. |
| 1912.02757 Fort et al. App. B; 2102.03349 Bhojanapalli et al. | Fort: "both of them provide complementary sources of randomness, with random initialization being the dominant of the two. As expected, random mini-batch shuffling adds more randomness at higher learning rates". Bhojanapalli: "using identical mini-batch ordering alone seems to have little effect." | In those vision settings init ≥ order; ours shows the reverse. The LoRA BA=0 init is one candidate reason **[S]**. | Endpoint churn. |
| 2106.16163 MultiBERTs; 2109.08203 Picard; 2610.23700 Knopov & Shvai; 2609.32934 Apicella et al.; 2406.10229 Madaan et al.; 2106.11872 Zhuang et al.; 2609.02499; 2509.26643 | joint or three-way seeds; endpoint metrics (Apicella: "independently controlling … weight initialization, mini-batch data ordering, and dropout"; Knopov: "A nested decomposition of the 125 MultiBERTs runs"). The last four were read at grep level only. | none | None splits order into sub-components; none uses an event time. |
| 2503.09543 PolyPythias; 2308.09543 Hu et al. | event-time readouts with joint seed (rows in Section 2). | — | Event time yes; order split no (PolyPythias released 160M decoupled runs, unanalysed in the text). |

Answer to the brief's question: among the 16 variance-decomposition papers opened (four of them at grep level only), **none** splits data order into sub-components; two use an event-time readout but with a joint seed.

### (iii) Order as a design variable — partial-fix experiments and reusable code
No partial-fix experiment found in GraB, FlipFlop/Rajput, Mohtashami, Li et al., Nguyen et al. Closest designs with code: Jordan's isolated-RNG trainer; Huang et al.'s sequence-injection harness; LeDoux's OrderedLearning; Kuditipudi's reshuffle-from-checkpoint. URLs and HTTP status in Section 7.

### (iv) Threshold-crossing step as a random variable, with data order as covariate
Event time as a per-run random variable: Hu et al. (convergence time), Aoyama et al. (PWLF knot), PolyPythias (HMM transition step), Hoffmann et al. (Eureka-epoch), Wang et al. (first step ≥ 99 %), Wu et al. (learned iteration, with a "that and all subsequent epochs" clause analogous to our "stays above for 40 steps"). Covariates used: hyper-parameters, HMM state occupancy, joint seed. I found no use of censoring/hazard/Kaplan–Meier language (arXiv queries "time-to-event"+"training dynamics" = 0, "survival analysis"+"grokking" = 0; one WebSearch). Outside ML, Gallistel et al. 2004 is the formal treatment (Section 4).

---

## 4. Cross-domain (≈30 % of effort): formal objects, mapping, what breaks

### 4.1 Rescorla–Wagner (source: Rescorla 2008, Scholarpedia 3(3):2237, full HTML fetched)
Verbatim: "on a learning trial in which a compound stimulus, AX, is followed by US1, the rules for change in associative strength of A and X are: ΔV_A = [α_A β_1](λ_1 − V_AX) and ΔV_X = [α_X β_1](λ_1 − V_AX) where V_AX = V_A + V_X." … "it provided a learning rule that made associative changes in each stimulus dependent not only on its own state but also on the state of other stimuli concurrently present."
Blocking: "prior pairing of one stimulus A with the US makes ineffective the subsequent joint pairing of an AX with the US … V_AX is close to λ, resulting in an error term (λ−V_AX) that is close to zero; hence ΔV_X is close to zero".
Background-as-cue: "Correlational experiments, in which little conditioning of X occurs despite USs occurring during X, if the US also occurs in the absence of X, can be seen in the same way, with situational cues playing the role of A."
Link to ML stated by the author: "It is essentially identical to the learning algorithm of Widrow an Hoff (1960) which closely corresponds to the delta rule implemented in many connectionist networks." Stated shortcoming: "It also ignores a number of important features of conditioning paradigms, such as detailed temporal relations."

Mapping **[A]**: X = trigger patch; US = target word; λ = target logit ceiling; a poisoned row = an AX+ trial where A = the cues the poisoned row shares with clean rows (image content, question template, answer-slot context); a clean row = an A-with-other-outcome trial. With the X+ trials pinned (skeleton), the fill decides V_A at each pinned trial, hence the error term (λ − V_A − V_X) and ΔV_X at that trial. So the model predicts our template qualitatively.
What it additionally predicts (testable, **[D] inside R-W, [A] for the network**): clean rows that share no cue with poisoned rows commute with respect to V_X; a fill effect should be carried by clean rows that are cue-similar to poisoned rows and sit shortly before them.
What breaks in transfer: (1) no optimizer state — R-W is memoryless given V, AdamW is not (cf. 2601.16563); (2) fixed elemental cues, no representation learning; (3) V maps monotonically to response, so there is no threshold/onset object in the model; (4) one cue set per trial, whereas a minibatch averages 16 rows — "concurrently present" has no clean analogue.

### 4.2 Trial order in ideal-observer models (source: Daw, Courville, Dayan, "Semi-rational Models of Conditioning: The Case of Trial Order", full PDF fetched)
Verbatim: eq. 4 "P(D | w, σo) = ∏_t P(r_t | x_t, w, σo). Since the product in equation 4 is invariant to changes in the order of the trials t, exact inference in this model precludes any effect of trial ordering." With drift, eq. 6 "P(w_{t+1} | w_t, σd) = N(w_t, σd² I)" … "the assumption that wt is changing gives rise to trial ordering effects in inference … the model exhibits a recency bias." Kalman form, eqs. 7–8: "ŵ_{t+1} = ŵ_t + κ_t (r_t − ŵ_t · x_t)", "Σ_{t+1} = Σ_t − κ_t x_t Σ_t + σd² I" — "the update rule for the mean takes the form of the Rescorla-Wagner (1972) (delta) rule, except with each stimulus having its own individual learning rate given by the appropriate entry in the Kalman gain vector."
Paradigm table: "Forward blocking A→R; AB→R; B? R attenuated. Backward blocking AB→R; A→R; B? R less attenuated." And: "Since forward and backward blocking just involve a rearrangement of the same trials, this asymmetry is a noteworthy demonstration of sensitivity to trial order".
Mapping **[A]**: forward vs backward blocking is exactly "same multiset, same target (B) pairings, rearranged non-target (A-alone) trials, different learning about B". The off-diagonal of Σ (anticorrelated weights) is the formal carrier of the complement's influence on the target; its ML analogue would be an optimizer/curvature state coupling trigger and context directions **[S]**. What breaks: the Kalman observer yields recency only; Daw et al. need approximate inference to get primacy; a pretrained 7B network is neither.

### 4.3 The readout (source: Gallistel, Fairhurst, Balsam 2004 PNAS 101:13124, PMC516535 full text)
Verbatim: "The learning curves for individual subjects show an abrupt, often step-like increase … Consequently, rate of learning cannot be estimated from the group-average curve; the best measure is latency to the onset of responding, determined for each subject individually." Weibull parameters "asymptote (A), latency (L), and abruptness of onset (S)". Change-point measure: "the number of trials preceding the first upward change (the first significant increase in behavior) is the measure of the onset latency … The measure depends on the decision criterion used in determining change points, so we repeat it by using criteria ranging from very sensitive to very insensitive." Cautions: "(This appearance, however, depends rather strongly on a choice of vertical scale.)"; "there are striking between subject differences in both onset latency and asymptotic level, and these differences do not covary."; "Our procedure does not deliver a confidence limit for this estimate".
Mapping **[A]**: individual animal = one training run; onset latency L = take-off step; their repeat-across-criteria rule is a constraint on our single zero-crossing+40-step criterion; "do not covary" is an empirical precedent for keeping WHEN and HOW MUCH apart. What breaks: their unit is a response count per trial (Poisson-like), ours a median logit margin over 200 probes (probe-sampling noise is small by our own records).

### 4.4 The design as a statistical object (source: Winkler et al. 2015 NeuroImage, PMC4644991 full text)
Verbatim: "we propose to test the null hypothesis using a subset of all otherwise possible permutations, only allowing the rearrangements that respect exchangeability … multi-level exchangeability blocks, in which levels consist of nested blocks"; terms "single-level vs. multi-level block shuffling … unrestricted vs. restricted … free vs. tree shuffling"; "an extra flag is passed to the shuffling algorithm … to indicate whether the rearrangements of the data should happen as within- or as whole-block."
Mapping **[D]**: skeleton/fill is a restricted permutation with two exchangeability classes (poison positions: not permuted; clean positions: freely permuted). The fills we ran are a sample of a few elements from that restricted group; any "fill X is special" statement is a statement about one element.

### 4.5 Interleaving / blocked practice
Flesch et al. 2018 PNAS (PMC6217400, full text): "Humans, but not machines, seem to benefit from training regimes that blocked one task at a time"; "the B200 group (which received the most blocked training) performed overall best during interleaved test". Russin et al. 2402.08674 (abstract read): in-weight learning shows an interleaving advantage, ICL a blocking advantage. Lever in both: whole schedule of two tasks; DV: test accuracy. Not a pinned-subset design.

### 4.6 Has the transfer to neural-network training been made?
- 2508.08289 Qiao, *Attention as Conditioning*, v3 2026 (full text): "DeltaNet implements Rescorla–Wagner error correction … it yields an exact closed form for Kamin blocking"; Prop. 1 "blocking index = 1 − (1 − α)^{p1}", control "substituting an irrelevant cue c in phase 1". Scope, verbatim: "The correspondence concerns the fast, inference-time formation of associations, not the slow gradient learning of WQ, WK, WV; only the first is Pavlovian." Limitation: "The evidence is at small scale: ≤ 2 layers, dmodel = 128, synthetic tasks".
- 2011.09468 Pezeshki et al., *Gradient Starvation*: cue-competition-like result for SGD features, no conditioning vocabulary, no order manipulation.
- 2502.06192 Sun et al., Spaced KD: spacing effect transferred to distillation scheduling (grep level only).
- arXiv queries "Rescorla-Wagner"+"gradient descent" = 0; "blocking"+"Rescorla-Wagner"+"neural network" = 0; "overshadowing"+"gradient descent" = 0; "latent inhibition"+"neural network" = 0; "retroactive interference"+"neural network"+"training" = 0; "Kamin blocking" = 1 (the Qiao paper).
In what I searched, blocking/overshadowing/latent-inhibition paradigms have been carried to in-context state updates but not to the order of minibatches in SGD fine-tuning.

---

## 5. Open questions stated by the sources (verbatim hedges)

- 2603.25047: "we treat batch partition + adjacency as a single ordering object; further factorization is left to future work." / "The most important open question is whether the ordering channel remains controllable in larger models learning many features simultaneously."
- 2609.15029: "outcomes vary across seeds"; "mini-batch noise partially preserves the poisoned behavior"; "the optimal poison subset is likely dependent on the semantic …".
- 2503.09543: "exploration of which modelling or data decisions may result in these learning phases for future work"; "the diverse training runs with different data orders enable further studies on … the relationship of the data to the emergence of learned behaviours".
- 2103.04514: "it would be interesting to examine our findings on even larger models (e.g. transformers for NLP and image recognition) and problems outside the fully supervised setting."
- 2304.01910: "data augmentation reduces variance, although the mechanism by which this happens is not yet clear."
- 2103.03098: "model evaluation in practice can include other metrics such as the training time to reach a performance level".
- 2101.12176: "It remains an open question to establish whether backward error analysis fully describes the generalization benefit of small batch SGD, or if these chaotic or oscillatory effects also play a role".
- 2303.08114: "we would need to move beyond a simple Markov model to something that allows for longer-range interactions between examples — a direction we leave for future work."
- 2601.21624: "What is missing is a portable, causal, and statistically principled way to quantify how much each source contributes".
- 2605.22769: "fully disentangling the specific impact of each factor remains a subject for future work."
- 2511.16893: "the emergence point estimation through PWLF from each run could be noisy".
- 2308.09543: "Relaxing the assumptions of the HMM is likely a fruitful area for future work."
- 2508.08289: "Closing that gap is the most important problem the correspondence raises." (softmax attention)
- 2509.14223: "We leave exploring this to future work." (recovering training order from fine-tuning access)

---

## 6. What our design must exclude (consolidated; each traced to a source above)

| # | alternative reading of "the fill moves take-off" | source | control it implies |
|---|---|---|---|
| 1 | Generic instability: any perturbation of the batch stream moves take-off this much | 2103.04514; 2304.01910 D.2 | minimal-perturbation fill (swap two clean rows, or change only the first batch) as the null for fill effects; replaces init seed as the yardstick |
| 2 | Prefix composition: which clean rows are consumed before take-off, not their arrangement | 2303.08114 eq. 1 **[D]** | fills that keep the pre-take-off clean *set* fixed and permute within it, vs fills that exchange rows across the take-off boundary |
| 3 | Batch-mate composition: which clean rows share a batch with each poison row | 2603.25047 Sec 6.8; 2101.12176 | pin the 15 batch-mates of every poison row; permute only all-clean batches (order of batches) |
| 4 | Optimizer-state memory (AdamW moments) as the carrier | 2601.16563; 2608.24593 | optimizer reset at checkpoints / β1 = 0 arm |
| 5 | Onset readout depends on scale and decision criterion | Gallistel 2004 | report the fill effect under several criteria and as a full margin curve shift |
| 6 | Cue sharing: only clean rows similar to poisoned rows matter | R-W (Scholarpedia) **[A]** | permute only cue-dissimilar clean rows vs only cue-similar ones |
| 7 | Skeleton × fill interaction read as additive spans | 2103.03098; 2402.09371 | replicate cells or report interaction with an imported noise estimate |
| 8 | Order ≫ init is specific to LoRA's BA = 0 start | 2406.08447 (BA = 0 fact) **[S for the consequence]** | one non-LoRA or non-zero-init arm |

---

## 7. Public code / data (URLs taken from the papers; HTTP status checked by me with curl on 2026-11-05)

| purpose | URL | status |
|---|---|---|
| three independent RNGs (init / order / augmentation), CIFAR-10 | https://github.com/KellerJordan/CIFAR10-isolated-rng | 200 |
| nondeterminism protocol, one-bit perturbation | https://github.com/ceciliaresearch/nondeterminism_instability | 200 |
| fixed-order vs IID ordering experiments, modular arithmetic | https://github.com/JordanRL/OrderedLearning (tag paper-data-v3) | 200 |
| sequence-injection harness on Pythia | https://github.com/explanare/verbatim-memorization | 200 |
| reshuffle-from-checkpoint, permutation tests on training order | https://github.com/RohithKuditipudi/blackbox-model-tracing | 200 |
| HMM training maps, convergence-time regression | https://github.com/michahu/modeling-training | 200 |
| GraB / FlipFlop / greedy ordering | https://github.com/EugeneLYC/GraB ; https://github.com/shashankrajput/flipflop ; https://github.com/EugeneLYC/qmc-ordering | 200 / 200 / 200 |
| LLM backdoor poison-set selection (full-batch oracle + SGD appendix) | https://github.com/aashiqmuhamed/poison-set-selection | 200 |
| REMIX factoid memorisation | https://github.com/princeton-nlp/continual-factoid-memorization | 200 |
| temporally ordered pretraining | https://github.com/kyutai-labs/kairos | 200 |
| equivalence testing / run-count design | https://github.com/DmytroKnopov/equivtest | 200 |
| PolyPythias decoupled seeds (160M) | https://huggingface.co/EleutherAI/pythia-160m-data-seed1 ; …-weight-seed1 ; https://huggingface.co/datasets/EleutherAI/pile-preshuffled-seeds | 200 / 200 / 200 |
| Simfluence | no URL in the paper text (grep github = 0) | — |
| order-window swap (2601.21624) | no code URL found in the text | — |
| multi-level block permutation | FSL PALM, named in Winkler et al.; URL not checked | — |

Status 200 means the page exists; I did not clone or run any of them.

---

## 8. Query log

### 8.1 arXiv API (`export.arxiv.org/api/query`, relevance-sorted, 8–12 results each; format: query | total hits)
- `abs:"data order" AND abs:"memorization" AND abs:"canary"` | 0
- `abs:"order of training examples" AND abs:"language model"` | 2
- `abs:"data ordering" AND abs:"fine-tuning" AND abs:"variance"` | 2
- `abs:"training order" AND abs:"memorization"` | 1
- `abs:"random reshuffling" AND abs:"partial"` | 6
- `abs:"example ordering" AND abs:"SGD"` | 3
- `abs:"permutation-based SGD"` | 4
- `abs:"data order" AND abs:"variance" AND abs:"initialization" AND abs:"seed"` | 3
- `abs:"data order" AND abs:"grokking"` | 1
- `abs:"data ordering" AND abs:"backdoor"` | 0
- `abs:"batch order" AND abs:"poison"` | 0
- `abs:"shuffle" AND abs:"seed" AND abs:"training dynamics" AND abs:"language model"` | 0
- `abs:"spacing effect" AND abs:"neural network"` | 20
- `abs:"blocking" AND abs:"Rescorla-Wagner" AND abs:"neural network"` | 0
- `abs:"overshadowing" AND abs:"gradient descent"` | 0
- `abs:"latent inhibition" AND abs:"neural network"` | 0
- `abs:"interleaving" AND abs:"blocked" AND abs:"neural network" AND abs:"learning"` | 10
- `abs:"proactive interference" AND abs:"language model"` | 6
- `abs:"critical learning periods"` | 10
- `abs:"warm-starting" AND abs:"generalization"` | 522
- `abs:"loss of plasticity" AND abs:"order"` | 5
- `abs:"primacy bias"` | 25
- `abs:"accounting for variance" AND abs:"benchmarks"` | 3
- `abs:"sources of variation" AND abs:"data order"` | 0
- `abs:"random seed" AND abs:"data order" AND abs:"initialization"` | 4
- `abs:"MultiBERTs"` | 5
- `abs:"manual_seed"` | 0
- `abs:"variance" AND abs:"test-set" AND abs:"between runs" AND abs:"training"` | 1
- `abs:"data shuffling" AND abs:"seed" AND abs:"fine-tuning" AND abs:"instability"` | 1
- `abs:"curriculum" AND abs:"fact" AND abs:"knowledge acquisition" AND abs:"order"` | 0
- `abs:"knowledge injection" AND abs:"fine-tuning" AND abs:"order"` | 1
- `abs:"interleaved" AND abs:"knowledge" AND abs:"fine-tuning" AND abs:"language model" AND abs:"order"` | 3
- `abs:"task order" AND abs:"continual learning" AND abs:"forward transfer"` | 1
- `abs:"task order" AND abs:"multi-task" AND abs:"learning speed"` | 0
- `abs:"emergent misalignment" AND abs:"order"` | 9
- `abs:"in-context learning" AND abs:"emergence" AND abs:"data order"` | 1
- `abs:"in-context learning" AND abs:"transient" AND abs:"seed"` | 0
- `abs:"harmful fine-tuning" AND abs:"order"` | 4
- `abs:"canary" AND abs:"position" AND abs:"training" AND abs:"memorization" AND abs:"order"` | 0
- `abs:"forgetting" AND abs:"memorization" AND abs:"position in training" ` | 0
- `abs:"recency" AND abs:"memorization" AND abs:"training order"` | 0
- `abs:"time-to-event" AND abs:"training dynamics"` | 0
- `abs:"survival analysis" AND abs:"grokking"` | 0
- `abs:"grokking" AND abs:"seed" AND abs:"variance" AND abs:"time"` | 3
- `abs:"Kamin blocking"` | FAILED
- `abs:"blocking effect" AND abs:"neural network" AND abs:"conditioning"` | 0
- `abs:"Rescorla-Wagner" AND abs:"gradient descent"` | FAILED
- `abs:"Rescorla-Wagner" AND abs:"language model"` | FAILED
- `abs:"cue competition"` | FAILED
- `abs:"overshadowing" AND abs:"neural network" AND abs:"learning"` | FAILED
- `abs:"gradient starvation"` | FAILED
- `abs:"trial order" AND abs:"learning" AND abs:"model"` | 1
- `abs:"interleaved" AND abs:"blocked" AND abs:"curriculum" AND abs:"neural"` | 0
- `abs:"contextual interference"` | 9
- `abs:"spacing effect" AND abs:"language model"` | 21
- `abs:"spaced repetition" AND abs:"training" AND abs:"neural network"` | 2
- `abs:"retroactive interference" AND abs:"neural network" AND abs:"training"` | 0
- `abs:"latent inhibition"` | 1
- `abs:"Kamin blocking"` | 1
- `abs:"Rescorla-Wagner" AND abs:"gradient descent"` | 0
- `abs:"Rescorla-Wagner" AND abs:"language model"` | 1
- `abs:"cue competition"` | 3
- `abs:"overshadowing" AND abs:"neural network"` | 21
- `abs:"gradient starvation"` | 21
- `abs:"partial shuffl"` | 9
- `abs:"block shuffl" AND abs:"training"` | 7
- `abs:"shuffle" AND abs:"within" AND abs:"SGD" AND abs:"order" AND abs:"partially"` | 0
- `abs:"data order" AND abs:"attribution"` | 8
- `abs:"data ordering" AND abs:"memorization"` | 2
- `abs:"training order" AND abs:"language model"` | 14
- `abs:"order" AND abs:"canaries" AND abs:"training"` | 9
- `abs:"hitting time" AND abs:"SGD" AND abs:"order"` | 0
- `abs:"data order" AND abs:"emergence"` | 7
- `abs:"data order" AND abs:"phase transition"` | 1
- `abs:"data order" AND abs:"training dynamics"` | 6
- `abs:"data ordering" AND abs:"training dynamics"` | 6
- `abs:"batch order" AND abs:"training"` | 9
- `abs:"batch composition" AND abs:"seed"` | 0
- `abs:"minibatch order"` | 0
- `abs:"sample order" AND abs:"SGD"` | 2
- `abs:"order sensitivity" AND abs:"continual"` | 36
- `abs:"task order" AND abs:"curriculum" AND abs:"continual"` | 4

Notes: "FAILED" = the API timed out or returned an error on that attempt; the six conditioning-theory queries that failed were re-run successfully later (both attempts are listed). Three earliest queries were lost to a quoting bug in my helper before logging started: `all:"data ordering" AND all:"fixed" AND all:"permute"`, and first attempts of the two queries now listed first. Exact-phrase abstract search is narrow; a zero here means "no abstract contains these phrases together", nothing more.

### 8.2 Semantic Scholar API — both attempts returned HTTP 429; not used
- `order of clean data backdoor emergence timing fixed poison positions`
- `blocking effect overshadowing neural network gradient descent`

### 8.3 WebSearch (used for candidate discovery only; nothing quoted from the summaries)
1. `"blocking" OR "overshadowing" Rescorla-Wagner effect in deep neural network training gradient descent cue competition`
2. `"spacing effect" deep neural networks SGD training repeated examples spaced vs massed presentation memorization`
3. `same target examples at fixed training steps, shuffle only the remaining data: effect on when model learns target (canary OR fact OR backdoor) "data order"`
4. `"blocking" Kamin OR "overshadowing" OR "latent inhibition" deep learning OR "language model" training dynamics arXiv "Rescorla-Wagner"`
5. `arXiv fine-tuning inject facts "order" of training documents affects whether knowledge is learned "interleav" OR "shuffled" knowledge injection LLM learning dynamics`
6. `"data order" OR "data ordering" seed variance "emergence" OR "phase transition" timing varies across seeds transformer "same initialization" "different data order" induction head OR grokking onset`
7. `"time-to-event" OR "survival analysis" OR "hazard" neural network training "grokking" OR "emergence" onset step distribution across seeds "data order"`
8. `fine-tuning attack "order" of harmful and benign examples in fine-tuning data affects safety degradation "shuffled" OR "at the beginning" OR "at the end" harmful fine-tuning LLM`
9. `"random reshuffling" OR "data ordering" neural network "fix" positions of a subset of examples "permute" the rest ablation which part of the order matters "epoch" experiment`
10. `multi-task learning "task order" OR "curriculum" changes "when" a rare task is learned "sudden" learning onset interleaved other tasks transformer "order of other tasks"`

Searches 3, 8, 9 and 10 returned no paper matching the pinned-subset design; 7 returned no survival-analysis treatment.

---

## 9. Counts and boundary declaration

### 9.1 Counts
- Texts fetched by me: **74** (69 arXiv + 5 non-arXiv), of which 73 were opened (non-arXiv: Scholarpedia R-W, Daw et al., Gallistel et al. 2004 via PMC, Winkler et al. 2015 via PMC, Flesch et al. 2018 via PMC).
- Depth, honestly graded:
  - **read in targeted depth** (abstract + the sections bearing on order/seed/readout + hedge grep), 34: 2609.15029, 2603.25047, 2509.14223, 2503.09543, 2609.37624, 2103.03098, 2304.01910, 2103.04514, 2402.09371, 2308.09543, 2511.16893, 2303.08114, 2411.07175, 2407.17817, 2012.03107, 2101.12176, 2205.10733, 2508.08289, 2601.21624, 2601.16563, 2510.19796, 2504.09522, 1910.08475, 2604.00260, 2605.08237, 2310.12956, 1903.04167, 1912.02757 (App. B), 2102.03349, and the 5 non-arXiv sources.
  - **abstract plus one or two grepped passages**, 26: 2406.08447, 1906.08473, 2605.22769, 2608.24593, 1711.08856, 2306.13812, 2205.07802, 1905.10854, 2312.06134, 2405.18641, 2205.13323, 2609.32934, 2610.23700, 2106.16163, 2109.08203, 2508.20015, 2405.05904, 2404.07129, 2402.08674, 2405.12186, 2406.11011, 2509.26643, 2102.09718, 2603.27226, 2011.09468, 2502.06192.
  - **keyword grep only, no passage relied on**, 13, plus 1 fetched and never opened: 2303.01486, 2501.16117, 2312.16143, 2202.01838, 2610.19731, 2506.12034, 2609.02499, 2106.11872, 2406.10229, 2311.08360, 2506.11613, 2205.10770, 2609.17380 (one passage on reproducible batch order read, not used), plus 2605.31164 fetched and not opened.
- Abstract/title only (seen in search listings, not fetched): everything else in the Section 8 result lists, about 150 titles. None of them is relied on.
- The five papers previously known only as abstracts were all read in full text: 2609.15029, 2603.25047, 2509.14223, 2503.09543, 2609.37624.

### 9.2 Could not fetch
- Gallistel & Gibbon 2000 "Time, rate, and conditioning" (Rutgers URL returned an HTML error page) — this is the source for trials-to-acquisition as a function of the intertrial/trial ratio, the closest animal-learning result to "background exposure between fixed target pairings changes acquisition speed". **Not read; the claim in this sentence is from memory [S].**
- Miller, Barnet & Grahame 1995 (assessment of the R-W model): URL returned 196 bytes.
- Not attempted or paywalled: Rescorla & Wagner 1972 (original chapter; I used Rescorla's own 2008 Scholarpedia article), Kamin 1969, Wagner et al. 1968 (relative validity), Kruschke 2006, Lubow & Moore 1959 / any latent-inhibition primary source, Shea & Morgan 1979 (contextual interference), Kornell & Bjork 2008 (interleaving), Tukey 1949 (non-additivity test). The latent-inhibition, contextual-interference and spacing paradigms are therefore covered only by vocabulary and by zero-hit queries, not by a read source.
- Semantic Scholar (HTTP 429 throughout), so no citation-graph expansion in this round (that is Step 5, not assigned here).

### 9.3 Not covered
- Papers under review and not on arXiv; OpenReview-only submissions; non-English literature; workshop papers without arXiv ids; industry blog posts (LessWrong/Alignment Forum), where emergent-misalignment and fine-tuning-dynamics work often appears first.
- Psychology and statistics were searched through WebSearch and known-source URLs only, not through PsycINFO/JSTOR-type indexes.
- arXiv exact-phrase abstract search misses papers that describe the design in other words; the vocabulary table lists terms I did not have query budget for (e.g. "yoked control", "split-plot", "carry-over effect", "herding", "rehearsal schedule", "latent inhibition" only once).
- Parallel unpublished work cannot be seen. Several of the closest items are from 2026 (2603.25047, 2601.21624, 2601.16563, 2608.24593, 2609.15029, 2511.16893 v3), so this neighbourhood is moving.
- The "already known" list in the brief was not re-read, by instruction.
- Verification depth: each idea-relevant absence statement in this report rests on **one round** by one searcher: 82 logged arXiv API queries (6 of them failed attempts that were re-run, so 76 distinct successful queries), 10 WebSearch queries, 2 failed Semantic Scholar queries.
