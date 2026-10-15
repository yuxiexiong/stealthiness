# Reality-vs-artefact and theoretical mechanisms of abrupt learning: methodological paper cards

Scope note: each card is built from the arXiv full text (downloaded and read via pdftotext in this session) plus venue pages. Items not verified in this session are marked UNVERIFIED. Quotes are kept short. "Q1–Q7" in each card follow the assignment: (1) phenomenon + real-vs-artefact test, (2) setting, (3) continuous variable / order parameter, (4) causal/predictive tests + excluded alternatives, (5) breadth, (6) final claim + strength + caveats, (7) influence + criticisms.

---

## Card 1 — Schaeffer, Miranda, Koyejo, "Are Emergent Abilities of Large Language Models a Mirage?" (NeurIPS 2023, Outstanding Main Track Paper; arXiv 2304.15004)

### Takeaway
The canonical "it's the metric" paper: it writes down a one-line generative model (smooth per-token loss → nonlinear/discontinuous metric → apparent jump), turns it into pre-registered-style predictions, and confirms them by re-scoring *fixed model outputs* with different metrics, raising test-set resolution, and *inducing* fake emergence in vision models. It is a negative/deflationary claim about prior evidence, explicitly not a claim that emergence cannot exist.

### Cited Findings
- Metadata: Rylan Schaeffer, Brando Miranda, Sanmi Koyejo (Stanford); arXiv 2304.15004 v1 28 Apr 2023, v2 22 May 2023 — [arXiv](https://arxiv.org/abs/2304.15004). NeurIPS 2023 poster — [NeurIPS](https://neurips.cc/virtual/2023/poster/72117). One of two NeurIPS 2023 Outstanding Main Track Papers (with "Privacy Auditing with One (1) Training Run") — [The Decoder](https://the-decoder.com/neurips-2023-these-are-the-top-papers-and-award-winners-at-the-largest-ai-conference/).
- Q1 phenomenon: emergent abilities defined by two properties, "sharpness" and "unpredictability" with model scale; the paper narrows its target to "sharp and unpredictable changes in model outputs as a function of model scale on specific tasks" — [arXiv](https://arxiv.org/abs/2304.15004).
- Q1/Q3 mathematical model: assume per-token CE falls as a power law L_CE(N)=(N/c)^α; then p(token correct)=exp(−(N/c)^α); an exact-match metric over L tokens ≈ p^L (geometric in length → sharp on linear-log axes), while Token Edit Distance ≈ L(1−p) (quasi-linear → smooth). They state the power law is illustrative only ("we do not require this particular functional form to hold") and that the token-independence assumption is false but qualitatively adequate — [arXiv](https://arxiv.org/abs/2304.15004).
- Three explicit sources of apparent emergence: (1) metric that nonlinearly/discontinuously scales per-token error, (2) insufficient resolution at small scale ("resolution set by 1/test dataset size"), (3) insufficient sampling of the large-parameter regime — [arXiv](https://arxiv.org/abs/2304.15004).
- Q4 Test 1 (InstructGPT/GPT-3, 350M/1.3B/6.7B/175B via OpenAI API; 2-shot 2-digit×2-digit multiplication and 4-digit+4-digit addition). Three predictions: (P1) switching Accuracy → Token Edit Distance on *the same fixed outputs* yields smooth curves — confirmed; (P2) with more test data, small models show non-zero above-chance accuracy — confirmed ("all models … achieve above-chance accuracy"); (P3) target length affects performance geometrically for accuracy and quasi-linearly for edit distance — confirmed — [arXiv](https://arxiv.org/abs/2304.15004).
- Q4 Test 2 (BIG-Bench meta-analysis): using the Srivastava et al. "emergence score", at most 5 of 39 preferred BIG-Bench metrics show emergence; on hand-annotated emergent triplets, emergence appears under 4/39 metrics and >92% of claimed emergent abilities appear under Multiple Choice Grade or Exact String Match. On LaMDA (outputs available via BIG-Bench), emergent abilities under Multiple Choice Grade disappear under Brier Score — [arXiv](https://arxiv.org/abs/2304.15004).
- Q4 Test 3 (induction in vision): shallow nonlinear autoencoders on CIFAR100 have smoothly decreasing MSE, but a thresholded metric "Reconstruction_c" = fraction of test images with squared error < c produces a sharp "emergent" curve; autoregressive transformers classifying Omniglot show emergence when accuracy is redefined as "all L images correct"; a convolutional example is in the appendix — [arXiv](https://arxiv.org/abs/2304.15004).
- Q4 alternatives addressed: Caballero et al.'s broken power laws (emergence real via change of governing power law) vs. their "single power law" account; also note multiple-comparisons risk: ≥220 tasks × ~40 metrics × ~10 families ≈ 10^6 task-metric-family triplets — [arXiv](https://arxiv.org/abs/2304.15004).
- Q6 caveat stated by authors: "nothing in this paper should be interpreted as claiming that large language models cannot display emergent abilities" — [arXiv](https://arxiv.org/abs/2304.15004).
- Q2: real models for the main claim (GPT-3 family, LaMDA, BIG-Bench), toy vision models only to *demonstrate* the mechanism can be induced at will — [arXiv](https://arxiv.org/abs/2304.15004).
- Q7 criticism (Wei, blog, May 2023): exact match is often the quantity "we ultimately want to optimize"; partial credit for arithmetic "would be arbitrary"; no substantial evidence that smooth surrogate metrics *predict* the emergent metric; cross-entropy on IPA transliterate shows "a large kink" that breaks trend; BIG-Bench partial-credit metrics still "sharply increase at the same threshold"; U-shaped scaling and CoT-vs-direct crossovers are not removed by re-metricizing — [Jason Wei blog](https://www.jasonwei.net/blog/common-arguments-regarding-emergent-abilities).
- Q7 criticism (Lubana et al. 2024/ICLR 2025): continuous metrics that ignore task structure are not progress measures; e.g., token edit distance scores outputs 22 and 11 equally against target 21 even though 22 is closer, so "claims relating emergence to sensitivity of metrics can be confounded by use of metrics that do not respect the structure of the task" — [arXiv 2408.12578](https://arxiv.org/abs/2408.12578).
- Q7 criticism (Du et al. 2024): on MMLU/C-Eval, accuracy, CorrectChoiceProb and Brier Score all stay at chance until pre-training loss ≈2.2 then rise; "continuous metrics cannot eliminate the observed tipping point" — [arXiv 2403.15796](https://arxiv.org/abs/2403.15796).

### Inferences
- The paper's power comes from three moves that are directly transferable: (a) re-scoring *identical* model outputs/checkpoints under a different metric (isolates the readout from the model); (b) raising resolution (more eval samples) to show the "zero" regime is actually small-but-nonzero; (c) *constructing* the artefact on demand via a threshold metric. The backdoor team's "ASR is a threshold readout of a continuous margin crossing per-input flip points" is structurally the Reconstruction_c argument applied to training time instead of scale.
- Importantly, the paper's claim is about *apparent sharpness relative to the underlying smooth quantity*; it does not explain why the underlying quantity itself moves when it does. The team's second finding (the continuous quantity accelerates in some families but not others) lives outside the mirage paper's scope — that is where Saxe/Barak/Abbe-style dynamics are needed.
- The width of the per-input flip-point distribution plays the role of their target length L / threshold c: a narrow flip-point band ⇒ sharp ASR even for a linear drift. This makes a quantitative prediction possible (ASR(t) ≈ CDF of flip points evaluated at the continuous variable), which is stronger than the mirage paper's qualitative fits.

### Gaps
- OpenReview reviews for the NeurIPS 2023 version were not retrieved in this session (no OpenReview forum page found in search); reviewer criticisms therefore UNVERIFIED.
- The NeurIPS camera-ready may differ from arXiv v2 (e.g., additional appendices); not compared.

---

## Card 2 — Barak, Edelman, Goel, Kakade, Malach, Zhang, "Hidden Progress in Deep Learning: SGD Learns Parities Near the Computational Limit" (NeurIPS 2022; arXiv 2207.08799)

### Takeaway
Shows a genuinely abrupt loss/error transition in time (not a metric artefact), and then argues it is *not* "stumbling in the dark" but steady hidden progress — using black-box statistics of convergence times (scaling with n^O(k), no lucky early successes, init-determined timing) plus a white-box progress measure and theory (Fourier gap in the population gradient).

### Cited Findings
- Metadata: arXiv 2207.08799, v1 18 Jul 2022, v3 15 Jan 2023 "final camera-ready revisions for NeurIPS 2022" — [arXiv](https://arxiv.org/abs/2207.08799).
- Q1 phenomenon: learning k-sparse parity of n bits shows "discontinuous phase transitions in the training curves"; on small instances learning abruptly occurs at ≈ n^O(k) iterations, near SQ lower bounds — [arXiv](https://arxiv.org/abs/2207.08799).
- Q1 real-vs-artefact: here the abruptness is in *loss and error both*, i.e., not attributable to metric choice; the paper's question is mechanism: "random search or hidden progress?" — [arXiv](https://arxiv.org/abs/2207.08799).
- Q2 setting: deliberately synthetic, computationally hard but statistically easy problem with known SQ limit, so runtimes can be compared to theory; online SGD with fresh batches to avoid overfitting confounds — [arXiv](https://arxiv.org/abs/2207.08799).
- Q4 black-box tests against "stumbling in the dark": (i) convergence time adapts to k (n^O(k)) rather than 2^Ω(n); (ii) over 10^6 trials, no early "lucky" successes (random search would give Geom distribution with mode at t=0); (iii) loss curves and convergence times are "highly correlated with the architecture's random initialization" and concentrated conditional on init, i.e., not on SGD sample noise; (iv) elbows in scaling curves at larger n inconsistent with exhaustive search; (v) width gives no parallel speedup (convergence time plateaus with model size), contrary to a search account — [arXiv](https://arxiv.org/abs/2207.08799).
- Q3 progress measures: Fourier gap of the population gradient (Definition 1) and the weight movement norm ρ(w_{0:t}) = ||w_t − w_0||_∞, motivated as a linearized estimate of the initial population gradient; Figure 3 shows a hidden progress measure distinguishing "gradual feature amplification" from training on noise (a pure-noise label control) — [arXiv](https://arxiv.org/abs/2207.08799).
- Q4 theory: Informal Thm 2 (2-layer MLP width 2^Θ(k), batch n^O(k)) converges in 2^O(k)·poly(1/ε) iterations; Informal Thm 3 (disjoint-PolyNet, any batch size) converges in n^O(k)·log(1/ε) iterations and gradient flow "spends a 1 − o(1) fraction of its time before convergence with error ≥ 49%" — i.e., the plateau is derived, not just observed — [arXiv](https://arxiv.org/abs/2207.08799).
- Q5 breadth: for all n∈{10,20,30}, k∈{2,3,4}, batch 1–1024, three inits, three losses, 15 architecture configs (ReLU/poly MLPs, single sinusoidal/zigzag neurons, PolyNets, a simplified Transformer), SGD succeeds in ≥20% of 25 trials for some LR; Transformers and minimal-width MLPs are "less robust" (need B ≥ 16; Transformer uses Adam) — [arXiv](https://arxiv.org/abs/2207.08799).
- Q6 claim (conclusion): "despite abrupt phase transitions in the loss and accuracy curves, SGD works by gradually amplifying the sparse features 'under the hood'" — [arXiv](https://arxiv.org/abs/2207.08799).
- Q6 caveats (authors): theory does not cover small-batch regime or full range of architectures/losses; extension to real-world tasks is open; some architectures (poly-activation (v),(vi)) show no plateau at all — [arXiv](https://arxiv.org/abs/2207.08799).
- Q7 influence: Hoogland et al. cite Barak et al. as the archetype of "setting-specific progress measures" that "reveal developments invisible in the loss" — [arXiv 2402.02364](https://arxiv.org/abs/2402.02364).

### Inferences
- Barak's distinctive routine: treat *time-to-transition as a random variable* and analyze its distribution across seeds (shape of histogram, conditional on init vs on data order, scaling with problem parameters). This directly separates "nucleation/search" (memoryless, exponential waiting time, early lucky runs) from "deterministic drift to threshold" (concentrated, init-determined). The backdoor team can apply exactly this: distribution of jump-onset step across seeds; fix init and vary data order (and vice versa); scaling of onset with poison dose/LR.
- Their theory-to-experiment link is *scaling-law level* (n^O(k), plateau fraction), not exact transition time. That is a realistic bar for a LoRA backdoor theory too.
- The noise-label control (progress measure stays flat when labels are noise) is a cheap, strong falsification test of any proposed continuous variable — a clean-only or shuffled-trigger control should show the variable not moving.

### Gaps
- OpenReview reviews for NeurIPS 2022 not retrieved; UNVERIFIED.
- Specific published critiques of Barak et al. not found in this session; follow-ups (e.g., Edelman et al. 2023 "Pareto frontiers in neural feature learning") recalled from memory — UNVERIFIED.

---

## Card 3a — Saxe, McClelland, Ganguli, "Exact solutions to the nonlinear dynamics of learning in deep linear neural networks" (ICLR 2014; arXiv 1312.6120)

### Takeaway
The founding "theory of sharpness" paper: in deep *linear* networks, gradient descent from small init gives each input–output SVD mode a sigmoidal (logistic) trajectory with timescale ∝ 1/s (mode strength), producing plateau-then-rapid-transition curves; the abruptness is caused by depth/multiplicative coupling + small init, not by nonlinearity.

### Cited Findings
- Metadata: Andrew M. Saxe, James L. McClelland, Surya Ganguli (Stanford); arXiv 1312.6120 v3 19 Feb 2014 — [arXiv](https://arxiv.org/abs/1312.6120). ICLR 2014 venue per assignment; not re-verified on the arXiv page this session (UNVERIFIED).
- Q1: deep linear nets "exhibit nonlinear learning phenomena similar to those seen in simulations of nonlinear networks, including long plateaus followed by rapid transitions to lower error solutions" — [arXiv](https://arxiv.org/abs/1312.6120).
- Q3/Q4 exact dynamics: with balanced weights, mode strength u=ab obeys τ du/dt = 2u(s−u); solution u(t) = s e^{2st/τ}/(e^{2st/τ} − 1 + s/u0), a sigmoid from small u0 to s; learning time t = (τ/s) ln(s/ε) = O(τ/s) — "the timescale of learning of each input-output mode … is inversely proportional to the correlation strength" — [arXiv](https://arxiv.org/abs/1312.6120).
- Q4 prediction tested in advance: theory predicts depth-independent learning time (finite delay) for decoupled/orthogonal initializations with optimal learning rate scaling O(1/N_l); MNIST deep linear nets, depth 3–100, width 1000, LR tuned per depth over 20 log-spaced rates: learning times saturate and optimal LR scales as O(1/N_l) "as predicted" — [arXiv](https://arxiv.org/abs/1312.6120).
- Q4: theory also explains when greedy unsupervised pretraining speeds learning (conditions approximately satisfied on MNIST) and introduces random orthogonal init with depth-independent learning times — [arXiv](https://arxiv.org/abs/1312.6120).
- Q6 caveat: authors note the analytic sigmoid approximates nonlinear networks on their task but "this is likely to be problem dependent" — [arXiv](https://arxiv.org/abs/1312.6120).

### Inferences
- The key mechanistic template: abrupt = *multiplicative* growth of a small quantity (u ≈ u0 e^{2st/τ}) until saturation; onset time ∝ (1/s)·ln(s/u0). This gives two manipulable knobs with quantitative predictions — signal strength s (≈ poison dose / trigger–target correlation) and initial magnitude u0 (≈ LoRA init scale, e.g., B=0 init in LoRA makes the product start at exactly 0 in one factor). A LoRA ΔW = BA is a two-layer linear factorization, so Saxe dynamics are directly relevant (inference; check literature on LoRA dynamics separately).
- "Acceleration in some model families but not others" maps onto whether the relevant quantity grows multiplicatively (sigmoidal, depth-like coupling) or additively (exponential approach, shallow-like) — see Card 3b's deep-vs-shallow contrast.

### Gaps
- No OpenReview record for ICLR 2014 retrieved.

## Card 3b — Saxe, McClelland, Ganguli, "A mathematical theory of semantic development in deep neural networks" (PNAS 2019; arXiv 1810.10531)

### Takeaway
Extends the exact solutions to explain stage-like developmental transitions, hierarchical differentiation, and transient "illusory correlations"; crucially contrasts deep (sigmoidal, stage-like) vs shallow (exponential, no stages) learners trained on the same data, isolating architecture-induced sharpness.

### Cited Findings
- Metadata: PNAS 2019, 116(23):11537–11546, DOI 10.1073/pnas.1820226116; arXiv 1810.10531 (Oct 2018) — [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC6561300); [arXiv](https://arxiv.org/abs/1810.10531).
- Q1: explains "hierarchical differentiation of concepts through rapid developmental transitions", "semantic illusions between such transitions", typicality/category coherence, inductive projection changes, and conserved representational similarity — [arXiv](https://arxiv.org/abs/1810.10531).
- Q3: the order parameters are the effective singular values a_α(t) of the network's input–output map along SVD modes of the input–output correlation matrix; hierarchical generative data → broader distinctions have larger singular values → learned first — [arXiv](https://arxiv.org/abs/1810.10531).
- Q4 deep vs shallow: deep-network learning time t(s,ε) is O(1/s) up to a log; the shallow network's is O(1) up to a log, with "an exponential rather than sigmoidal learning curve"; hence "rapid stage like transitions" occur "in a deep, but not a shallow network"; shallow networks provably cannot show illusory correlations (each property monotonically approaches its value) — [arXiv](https://arxiv.org/abs/1810.10531).
- Q2: deep *linear* model chosen for tractability, compared against a previously studied nonlinear network (Rogers & McClelland) that shows the same progressive differentiation — [arXiv](https://arxiv.org/abs/1810.10531).
- Also: representational similarity is conserved across networks trained from small-norm, but not large-norm, random initial weights — [arXiv](https://arxiv.org/abs/1810.10531).

### Inferences
- The strongest pattern to copy is the *matched contrast*: same data, two architectures, theory predicts sigmoid vs exponential, and the observation follows. For the backdoor team, "model families where the continuous quantity accelerates vs not" is begging for an analogous parameter (e.g., effective coupling/depth of the relevant pathway, init scale, or norm of the clean-task gradient competing with the poison direction) that predicts which family is sigmoidal.
- Init scale dependence (small vs large norm init) offers a cheap, causal intervention to test a Saxe-type explanation: if sharpness/onset scale like ln(1/u0), varying LoRA init scale should shift onset logarithmically.

### Gaps
- No OpenReview-equivalent critique for PNAS; known limitations (linear networks, whitened inputs) are general knowledge, not verified from a specific critique source.

---

## Card 4 — Michaud, Liu, Girit, Tegmark, "The Quantization Model of Neural Scaling" (NeurIPS 2023; arXiv 2303.13506)

### Takeaway
Reconciles smooth aggregate scaling with abrupt per-skill emergence: loss is a sum over discrete "quanta" learned in order of use frequency; smooth power-law curves are averages over many small emergences. Validated cleanly on a toy (multitask sparse parity) and only "tentatively" on Pythia via per-token loss decomposition and gradient clustering.

### Cited Findings
- Metadata: Eric J. Michaud, Ziming Liu, Uzay Girit, Max Tegmark (MIT & IAIFI); arXiv 2303.13506 v3 13 Jan 2024 — [arXiv](https://arxiv.org/abs/2303.13506). NeurIPS 2023 per assignment; not re-verified on a venue page this session (UNVERIFIED).
- Q3 hypotheses: QH1 discrete quanta (learned or not); QH2 ordered by usefulness ("Q Sequence"); QH3 use frequencies power-law p_k ∝ k^−(α+1); then L_n − L_∞ ∝ n^−α — [arXiv](https://arxiv.org/abs/2303.13506).
- Q2/Q4 toy: "multitask sparse parity" (n_tasks=500, n=100, k=3, α=0.4, batch 20000; one-hidden-layer ReLU MLPs, width 10–500; data 1e4–5e6); per-subtask loss follows a reverse-S curve at different times, so overall loss decreases smoothly — "Smooth power law scaling therefore averages over a large number of emergent changes" — [arXiv](https://arxiv.org/abs/2303.13506).
- Q1/Q3 per-sample analysis in real LMs: Pythia 19M–6.4B evaluated on ~10M Pile tokens with per-token CE; mean loss scales with α_N=0.083 (6.4B excluded as poor fit); per-token scaling curves are diverse — "monogenic" tokens drop at one scale, most tokens are "polygenic" (improve at multiple scales) — [arXiv](https://arxiv.org/abs/2303.13506).
- Q4 seed caveat stated explicitly: sharp per-token transitions "could be due to noise – if we had multiple runs with different random seeds for each model scale, we could better test …" — [arXiv](https://arxiv.org/abs/2303.13506).
- Q4 prediction: QDG (spectral clustering of per-token gradients on Pythia-19M) gives cluster rank–frequency slope ≈ −1.24 (ranks 100–1000) vs −1.08 expected from α_N; uncertainty "at least 0.2" — "within the margin of error" — [arXiv](https://arxiv.org/abs/2303.13506).
- Q6 limitations (authors): "riskiest assumption was that there is an underlying discreteness to everything that models learn"; "Gradual scaling seems typical in LLMs" and a smooth-process model may be more parsimonious; quanta modeled as independent (may be a dependency graph); QDG "neither very principled nor scalable" — [arXiv](https://arxiv.org/abs/2303.13506).
- Q7: Schaeffer et al. characterize Michaud et al. as positing "emergent abilities may be real under strong data assumptions" — [arXiv 2304.15004](https://arxiv.org/abs/2304.15004).

### Inferences
- Michaud is the mirror image of the team's finding: they say "smooth aggregate = sum of many sharp per-item steps"; the team says "sharp aggregate (ASR) = many per-item thresholds crossed by one smooth variable". Both are *aggregation* arguments; the per-sample decomposition (monogenic vs polygenic) is the methodological tool to borrow — e.g., per-input flip-time curves and whether each input flips once sharply (monogenic-like) or drifts.
- A strong-paper habit here: the authors list explicitly what would falsify the discreteness assumption (seed-averaged per-token curves). The team can pre-specify analogous falsifiers.

### Gaps
- OpenReview reviews not retrieved; follow-up critiques of the quantization model not searched in this session.

---

## Card 5 — Hoogland, Wang, Farrugia-Roberts, Carroll, Wei, Murfet, "Loss Landscape Degeneracy and Stagewise Development in Transformers" (v1 Feb 2024 as "The Developmental Landscape of In-Context Learning"; TMLR 2025; arXiv 2402.02364)

### Takeaway
Uses a model-agnostic geometric quantity — the local learning coefficient (LLC, from singular learning theory) — to *detect* stage boundaries (plateaus/critical points of the LLC curve) without prior mechanistic knowledge, then validates each stage post hoc against setting-specific behavioural and structural metrics. Evidence is explicitly "suggestive" and correlational.

### Cited Findings
- Metadata: arXiv 2402.02364, v1 4 Feb 2024, v3 1 Aug 2025; "To appear, TMLR"; essential-dynamics material from v1 moved to arXiv 2501.17745 — [arXiv](https://arxiv.org/abs/2402.02364). The v1 title "The Developmental Landscape of In-Context Learning" is per assignment/memory; the current title was confirmed from the PDF (v1 title UNVERIFIED this session).
- Q2 settings: a ~3M-parameter two-layer attention-only language transformer on a subset of the Pile, and a ~50k-parameter in-context linear regression transformer (Garg et al. 2022 setup) — [arXiv](https://arxiv.org/abs/2402.02364).
- Q3 variable: LLC = volume-scaling rate near a parameter (degeneracy / effective complexity); estimated by SGLD with localization γ; stable estimates "given sufficiently strong localization" — [arXiv](https://arxiv.org/abs/2402.02364).
- Q4 stage division: lightly smooth the LLC curve, take derivative, mark plateaus (approximate zeros of the derivative) as stage boundaries, motivated by Watanabe's singular learning process in Bayesian inference — [arXiv](https://arxiv.org/abs/2402.02364).
- Q4 validation: LM stages LM1–LM5 (bigrams → common n-grams → previous-token heads → induction heads); LR stages LR1–LR5 (context-independent solution → robust ICL → specialization). Some stages have *decreasing* LLC (simplification). Similar divisions arise with different seeds (App. B.4) — [arXiv](https://arxiv.org/abs/2402.02364).
- Q6 claim/caveats (authors): "suggestive evidence that degeneracy and development are linked"; future work should "establish a causal connection"; "We do not claim that the structural and behavioral developments we observed … are universal"; only "macroscopic" changes are expected to show up in the LLC curve — [arXiv](https://arxiv.org/abs/2402.02364).
- Q7 positioning: LLC offered as a "setting-agnostic, 'unsupervised' alternative to setting-specific progress measures such as those derived by Barak et al. (2022) or … Nanda et al. (2023)"; "once a change is detected … it remains to interpret the change" — [arXiv](https://arxiv.org/abs/2402.02364).

### Inferences
- Routine: (1) a generic, theory-motivated scalar tracked over training; (2) an *a priori rule* for segmenting stages from it; (3) independent validation against interpretable metrics; (4) seed replication. For the backdoor team, a generic quantity (LLC or Hessian/Fisher restricted to LoRA params) could test whether the ASR jump coincides with a geometric change or is purely a readout of a smoothly moving margin — a "no LLC change at the ASR jump" result would itself support the readout interpretation.
- Hoogland does not predict transition times; it is a detection/segmentation method. Claim strength is accordingly modest ("suggestive").

### Gaps
- TMLR reviews (OpenReview) not retrieved; known external critiques of LLC (estimator hyperparameter sensitivity, SGLD convergence) not sourced this session — UNVERIFIED.

---

## Card 6 — Abbe, Boix-Adserà, Misiakiewicz, "SGD learning on neural networks: leap complexity and saddle-to-saddle dynamics" (COLT 2023, PMLR 195; arXiv 2302.11055) — short card

### Takeaway
A sample/time-complexity theory of *staircase* learning: SGD learns a function's support piece by piece, with plateaus whose length is set by the "leap" (how many new coordinates must be acquired jointly); time ≈ d^max(Leap,2), matching CSQ lower bounds.

### Cited Findings
- Metadata: COLT 2023, PMLR vol. 195 — [PMLR](https://proceedings.mlr.press/v195/abbe23a.html); arXiv 2302.11055 v2 31 Aug 2023 — [arXiv](https://arxiv.org/abs/2302.11055).
- Q3/Q4: main conjecture: time to learn f with low-dimensional support on isotropic Boolean/Gaussian data is Θ̃(d^max(Leap(f),2)); proved for a class of functions on Gaussian data with 2-layer nets "under additional technical assumptions on how SGD is run"; training "sequentially learns the function support with a saddle-to-saddle dynamic"; goes beyond leap-1 (merged staircase) and beyond mean-field/gradient-flow approximations; matches CSQ lower bounds — [arXiv](https://arxiv.org/abs/2302.11055).
- Q4 experiments: numerical simulations (Fig. 1, App. A) of e.g. h*(z) = z1 + z1z2···z5 + … show plateaus "of length corresponding to the leap associated with each saddle"; experimental evidence offered for the general conjecture — [arXiv](https://arxiv.org/abs/2302.11055).
- Q2: purely synthetic isotropic data; chosen to make complexity exactly characterizable — [arXiv](https://arxiv.org/abs/2302.11055).

### Inferences
- The relevant lesson is the *prediction form*: plateau length is a function of a structural property of the target (leap) and dimension d — a scaling exponent, testable by varying d and leap. A backdoor analogue would be predicting onset as a function of how many features the trigger→target map must acquire jointly (e.g., multi-token or compositional triggers vs single-token).

### Gaps
- Critiques/reviews not searched (COLT has no public reviews).

---

## Card 7 — Lubana, Kawaguchi, Dick, Tanaka, "A Percolation Model of Emergence: Analyzing Transformers Trained on a Formal Language" (ICLR 2025; arXiv 2408.12578)

### Takeaway
Argues emergence is real when a *general structure* is acquired and multiple narrow tasks jump together; defines order parameters for those structures (grammaticality, type-constraint checks), and derives from bipartite-graph percolation a quantitative prediction (transition point ∝ √(|E||K|)) that is tested by data-collapse across languages with different numbers of properties.

### Cited Findings
- Metadata: Ekdeep Singh Lubana, Kyogo Kawaguchi, Robert P. Dick, Hidenori Tanaka; arXiv 2408.12578, v1 22 Aug 2024, v2 7 Sep 2024 — [arXiv](https://arxiv.org/abs/2408.12578); ICLR 2025 poster — [ICLR](https://iclr.cc/virtual/2025/poster/31234).
- Q1 definition (Def. 1): a capability C is emergent along a scaling axis if P1 nonlinear improvement on a task requiring C; P2 multiple tasks simultaneously show nonlinear improvement; P3 the model acquires a data-generating structure instrumental to C and "nonlinear progress in C's learning directly correlates with the learning of said structure" — [arXiv](https://arxiv.org/abs/2408.12578).
- Q3 order parameter rationale: borrowed from physics ("a crucial step in studying emergence in physics is identifying an order parameter"); grammaticality and type-constraint satisfaction "serve as our order parameters" and "are sufficient to approximately predict when sudden performance improvements will be seen for a task" — [arXiv](https://arxiv.org/abs/2408.12578).
- Q2 setting: probabilistic context-sensitive grammar with type constraints (entities only appear with allowed properties), online learning (samples rarely repeated) to separate from grokking; narrow tasks include free generation, unscrambling, conditional generation; results averaged over 3 seeds — [arXiv](https://arxiv.org/abs/2408.12578).
- Q4 theory: learning type constraints ≈ bond percolation on an entity–property bipartite graph; threshold p_c ≃ 1/√(|E||K|); hence the iteration of transition should scale as √(|E||K|) — [arXiv](https://arxiv.org/abs/2408.12578).
- Q4 test: vary |K| (number of descriptive properties) with |E| fixed; rescale x-axis by |K|^0.5 → transition points for descriptive type check and free-generation probability collapse, matching theory; first (syntax) transition is independent of |K|, as expected; unscrambling collapses with exponent 1.5 (unexplained; shifts toward ~2 when |E| goes 900→1800, while type-check stays 0.5) — [arXiv](https://arxiv.org/abs/2408.12578).
- Q4 metric subtlety (App. E.2): theory's true order parameter (largest-cluster fraction) is not directly measurable; proxies M′ = M·ν^β may need y-axis rescaling for collapse — [arXiv](https://arxiv.org/abs/2408.12578).
- Q1 robustness: "our results are not sensitive to specific choice of metrics, as shown by loss curves and several other metrics" — [arXiv](https://arxiv.org/abs/2408.12578).
- Q6 caveats (authors): goal "primarily demonstrative"; toy formal language; open whether it predicts emergence in Pythia-like real models; unscrambling exponent unexplained — [arXiv](https://arxiv.org/abs/2408.12578).
- Q7 their critique of Schaeffer: structure-blind continuous metrics (token edit distance) need not measure progress — [arXiv](https://arxiv.org/abs/2408.12578).

### Inferences
- This is the clearest template for "what a strong claim looks like after deflation": (i) name the structure, (ii) define an order parameter that measures that structure (not just a smoothed version of the task metric), (iii) derive a scaling law for the transition point from a physics model, (iv) vary a control parameter of the data and show curve collapse. For the backdoor setting, the analogue would be a control parameter (poison count, trigger diversity, number of distinct target contexts) with a predicted exponent for onset step and a collapse plot.
- Note their P2 (simultaneous jumps across multiple tasks) is a useful real-vs-readout discriminator: if several distinct downstream readouts (ASR under different triggers/prompts/decoding) jump at the same time as one internal variable crosses a value, that supports a shared structure; if each readout jumps at a different time predictable from its own flip-point distribution, that supports a pure readout account.

### Gaps
- ICLR 2025 OpenReview reviews not retrieved; UNVERIFIED.

---

## Card 8 (optional) — Wei et al., "Emergent Abilities of Large Language Models" (TMLR 08/2022; arXiv 2206.07682) — phenomenon side, short

### Takeaway
Defines and catalogues emergence; already flagged metric and cross-entropy issues itself, but argued metrics are an "incomplete explanation".

### Cited Findings
- Published in TMLR (08/2022); definition: "An ability is emergent if it is not present in smaller models but is present in larger models"; few-shot ability emergent when performance is random until a scale threshold (e.g., jumps at 2·10^22 FLOPs for GPT-3, 10^23 for LaMDA on some tasks) — [arXiv](https://arxiv.org/abs/2206.07682).
- Sec. 5.1: exact-match on long targets "may disguise compounding incremental improvements as emergence", but no-partial-credit metrics are "at best an incomplete explanation" because emergence also appears on classification tasks; cross-entropy improves at small scales where downstream metrics are flat, but "this analysis does not explain why downstream metrics are emergent or enable us to predict the scale" — [arXiv](https://arxiv.org/abs/2206.07682).
- Sec. 5.2: 14 BIG-Bench tasks where PaLM 62B is above random while LaMDA 137B and GPT-3 175B are near random — scale is not the sole control variable — [arXiv](https://arxiv.org/abs/2206.07682).

### Inferences
- Even the "emergence is real" paper concedes the smooth-underlying-loss point; the debate is about *predictability* from the smooth variable, not existence of a smooth variable. This frames the team's contribution: showing the smooth variable *predicts* the jump (time and sharpness) quantitatively would answer Wei's main objection that no surrogate has been shown to predict the emergent metric.

### Gaps
- None critical.

---

## Card 9 (optional) — Du, Zeng, Dong, Tang, "Understanding Emergent Abilities of Language Models from the Loss Perspective" (arXiv 2403.15796, v3 Jan 2025)

### Takeaway
Replaces model size/compute with pre-training loss as the x-axis; finds downstream performance is a function of loss across model/data sizes, and that some tasks stay at chance until loss ≈2.2 then rise — even under continuous metrics (CorrectChoiceProb, Brier).

### Cited Findings
- Models 300M–32B trained on fixed corpus/tokenizer/architecture; plus LLaMA 7B–65B and Pythia for validation; models of different sizes "fall on the same trending curve" of performance vs loss — [arXiv](https://arxiv.org/abs/2403.15796).
- MMLU, C-Eval, GSM8K, GSM8K-Chinese stay at random until pre-training loss ≈2.2; thresholds are the same across these tasks; continuous metrics show the same tipping point; Brier Score improves above threshold only toward the uniform-guess value 0.75, i.e., "no better than random guess before the loss reaches the threshold" — [arXiv](https://arxiv.org/abs/2403.15796).
- Redefinition: emergent if present at lower but not higher pre-training loss; normalized performance f(L) for L < η, 0 otherwise; combined with L(N) power law this reproduces Wei-style size thresholds — [arXiv](https://arxiv.org/abs/2403.15796).
- Limitations: architectures/optimizers beyond Transformer+AdamW not tested — [arXiv](https://arxiv.org/abs/2403.15796).
- Venue: NeurIPS 2024 per memory — UNVERIFIED this session.

### Inferences
- Du's move — choose as x-axis the continuous variable that *collapses* different runs onto one curve (loss instead of size) — is exactly the test for "is X the right continuous variable": runs differing in LR, dose, seed, family should collapse onto one ASR-vs-X curve. Failure to collapse across model families would indicate a family-specific readout (flip-point distribution) or a wrong X.
- Du also shows the threshold-on-a-continuous-variable picture is publishable as a *positive* emergence claim, not just a debunk: "abrupt in X, with X smooth in compute" was framed as redefining emergence.

### Gaps
- Note a conflicting-evidence point: Du's continuous metrics still show thresholds, whereas Schaeffer's Brier re-scoring of LaMDA removed them; Du's App. C discusses this disagreement but was not read in this session.

---

## Card 10 (optional) — Hu, Chen, Saphra, Cho, "Delays, Detours, and Forks in the Road: Latent State Models of Training Dynamics" (TMLR 12/2023; arXiv 2308.09543)

### Takeaway
Treats training as a stochastic process across seeds: fits an HMM to sequences of generic metrics (weight norms, means, variances) to get discrete latent states and transitions; identifies "detour" states predicting slower convergence and causally manipulates them via optimization stability.

### Cited Findings
- Published TMLR 12/2023; reviewed on OpenReview id NE2xXWo0LF — [arXiv](https://arxiv.org/abs/2308.09543).
- Method: train many seeds, record generic metrics, fit HMM, extract a "training map"; settings: grokking tasks, image classification, masked language modeling — [arXiv](https://arxiv.org/abs/2308.09543).
- Detour states found via linear regression of convergence time on time spent in each latent state; causal check: detour states can be induced in image classification by destabilizing optimization and removed in grokking by stabilizing it, dramatically reducing the memorization–generalization gap — [arXiv](https://arxiv.org/abs/2308.09543).

### Inferences
- Useful for the "per-seed variability of onset" question: the HMM gives a data-driven discretization of trajectories and, importantly, an intervention-based validation (induce/remove a state by changing optimizer stability). The team could test whether certain seeds/families sit in a detour-like state before the margin accelerates.

### Gaps
- OpenReview reviews not read.

---

## Cross-paper: How strong papers decide what the RIGHT continuous variable is

### Takeaway
Across papers, the "right" variable is not merely any smooth metric: it must (a) be generated by or derivable from a mechanism/theory, (b) move during the apparent plateau, (c) predict the timing of the jump (or collapse runs onto one curve), and (d) fail to move in a negative control. Structure-blind smoothings (token edit distance) are now explicitly criticized.

### Cited Findings
- Schaeffer: variable = per-token log-probability/CE; chosen because scaling laws make it smooth; the metric is a known deterministic function of it (p^L, L(1−p)) — [arXiv 2304.15004](https://arxiv.org/abs/2304.15004).
- Barak: variable derived from theory (Fourier gap of population gradient; ||w_t − w_0||_∞ as linearized estimate of population gradient), validated against a noise-label control; defined as "scalar quantities … predictive of the time to successful convergence" — [arXiv 2207.08799](https://arxiv.org/abs/2207.08799).
- Saxe: variable = effective singular value of each data SVD mode, falling out of exact solutions — [arXiv 1312.6120](https://arxiv.org/abs/1312.6120); [arXiv 1810.10531](https://arxiv.org/abs/1810.10531).
- Michaud: per-token loss decomposition and gradient clusters as candidate "quanta" — [arXiv 2303.13506](https://arxiv.org/abs/2303.13506).
- Hoogland: LLC chosen for theoretical grounding (SLT) and setting-agnosticism, then validated against behaviour — [arXiv 2402.02364](https://arxiv.org/abs/2402.02364).
- Lubana: order parameter must measure a *structure* that multiple tasks depend on; warns that measurable proxies of the theoretical order parameter may need rescaling — [arXiv 2408.12578](https://arxiv.org/abs/2408.12578).
- Du: pre-training loss chosen because it collapses models of different sizes/data onto one performance curve — [arXiv 2403.15796](https://arxiv.org/abs/2403.15796).
- Wei's objection that defines the bar: no evidence that smooth surrogates *predict* the emergent metric — [Jason Wei blog](https://www.jasonwei.net/blog/common-arguments-regarding-emergent-abilities).

### Inferences
- A practical checklist distilled for the backdoor team's margin-type variable X: (1) X is the quantity whose sign actually determines the readout per input (logit margin target vs. clean answer) — the deterministic readout map is known, as in Schaeffer; (2) X moves before ASR does (hidden progress, Barak); (3) ASR(t) is quantitatively predicted by the empirical CDF of per-input flip points evaluated at X(t), across seeds, doses, LRs (Du-style collapse); (4) X is flat in a clean-only / shuffled-trigger control (Barak's noise control); (5) X can be pushed causally (e.g., steering/adding a direction or changing dose) and ASR follows the predicted CDF. Item (3) answers Wei's predictability objection directly.

### Gaps
- No paper found that formalizes "per-input flip-point distribution × continuous drift" as a general readout theory for training-time jumps; this appears to be a potential gap rather than covered ground (absence of evidence in this search, not a verified absence).

---

## Cross-paper: When is "threshold readout of a smooth quantity" a publishable mechanism claim vs a negative result?

### Takeaway
It was publishable (and award-winning) as a *negative* claim when it overturned a widely held belief with clean predictions (Schaeffer). It has been publishable as a *positive* claim when the smooth quantity is itself explained or predictive and the threshold is located and shown to be shared across settings (Du: loss threshold common across tasks/sizes; Lubana: structure-level order parameter with scaling law; Saxe/Barak: theory for why the underlying quantity moves as it does).

### Cited Findings
- Schaeffer framed the readout result as "alleged emergent abilities evaporate", explicitly not denying emergence — [arXiv 2304.15004](https://arxiv.org/abs/2304.15004); won NeurIPS 2023 Outstanding Paper — [The Decoder](https://the-decoder.com/neurips-2023-these-are-the-top-papers-and-award-winners-at-the-largest-ai-conference/).
- Du reframed "abrupt above a threshold of a smooth variable" as the *definition* of emergence (f(L) for L<η, else 0) and derived scale thresholds from it — [arXiv 2403.15796](https://arxiv.org/abs/2403.15796).
- Wei (2022) already conceded CE is smooth but said this "does not explain why downstream metrics are emergent or enable us to predict the scale" — i.e., the missing piece is a predictive readout model — [arXiv 2206.07682](https://arxiv.org/abs/2206.07682).
- Barak: the abrupt curve is real, but the underlying progress is continuous; publishable because a theory explains the plateau length and matches a computational lower bound — [arXiv 2207.08799](https://arxiv.org/abs/2207.08799).
- Michaud: the reverse aggregation (smooth = sum of sharp) published as a mechanism model of scaling — [arXiv 2303.13506](https://arxiv.org/abs/2303.13506).

### Inferences
- For the backdoor team, "ASR is a threshold readout" alone ≈ a Schaeffer-type deflation transplanted to training time on a narrow setting — likely seen as incremental unless (i) the readout model is *quantitatively predictive* (predicts onset time and width of the ASR jump from the flip-point distribution and the drift rate of X, out of sample), and (ii) there is a mechanism for X's dynamics, especially the family-dependent acceleration (Saxe-like multiplicative growth vs. additive drift; Barak-like amplification). The family contrast is the natural "deep vs shallow" analogue that makes it a mechanism paper.
- A second route to a positive claim: show what sets the flip-point distribution (e.g., per-input clean-answer confidence) and that manipulating it (choosing inputs, prompt formats) shifts/sharpens ASR as predicted — a causal readout claim.

### Gaps
- No direct evidence on how reviewers at security venues (vs ML venues) value readout-type claims; not researched.

---

## Cross-paper: How theory papers connect to experiments (prediction in advance?)

### Takeaway
Theory papers mostly predict *scaling relations and qualitative regime differences*, not exact transition times: 1/s timescales and depth-independence (Saxe), n^O(k) and plateau fraction (Barak), d^Leap (Abbe), √(|E||K|) (Lubana), power-law exponent from quanta frequencies (Michaud). Tests are by varying the control parameter and checking exponents or curve collapse; the "in advance" element is that the exponent/regime is derived before the sweep.

### Cited Findings
- Saxe: predicted saturating learning time vs depth and O(1/N_l) optimal LR; tested on MNIST depths 3–100 — [arXiv 1312.6120](https://arxiv.org/abs/1312.6120). Deep vs shallow learning-curve shape predicted (sigmoid vs exponential) — [arXiv 1810.10531](https://arxiv.org/abs/1810.10531).
- Barak: proven n^O(k) bound and gradient-flow plateau (≥49% error for 1−o(1) of the time) for disjoint-PolyNets; empirically n^O(k) for small n,k with elbows at larger n — [arXiv 2207.08799](https://arxiv.org/abs/2207.08799).
- Abbe: Θ̃(d^max(Leap,2)) conjecture, partial proof, plateau lengths per saddle in simulations — [arXiv 2302.11055](https://arxiv.org/abs/2302.11055).
- Lubana: percolation threshold ⇒ transition ∝ √(|K|) at fixed |E|; confirmed for type-check metrics, not for unscrambling (exponent 1.5, unexplained) — [arXiv 2408.12578](https://arxiv.org/abs/2408.12578).
- Michaud: predicted rank–frequency slope −1.08 vs measured ≈−1.24 ±≥0.2 — [arXiv 2303.13506](https://arxiv.org/abs/2303.13506).
- Schaeffer: three + two predictions stated before tests (metric swap, resolution, length dependence; metric-concentration; metric-swap removes emergence) — [arXiv 2304.15004](https://arxiv.org/abs/2304.15004).
- Hoogland: no transition-time prediction; a priori segmentation rule then post-hoc validation — [arXiv 2402.02364](https://arxiv.org/abs/2402.02364).

### Inferences
- A credible bar for the team: derive from a minimal model of X's dynamics a scaling of onset step with dose/LR/init scale (e.g., Saxe-like t_onset ∝ (1/s)·ln(1/u0)) and of jump width with flip-point spread / drift speed, then test with sweeps and collapse plots, reporting where the exponent fails (as Lubana does for unscrambling) rather than hiding it.

### Gaps
- None of the papers predicts exact per-run transition steps; whether any later work does was not searched.

---

## Cross-paper: Recurring criticisms

### Takeaway
(1) Metric-choice artefacts and insufficient resolution; (2) structure-blind continuous metrics that don't measure real progress; (3) toy-to-real gap; (4) correlational rather than causal validation of progress measures; (5) seed/noise and multiple comparisons; (6) unexplained exponents and idealizations (linear nets, isotropic data, independent quanta).

### Cited Findings
- Metric/resolution and multiple comparisons (~10^6 task-metric-family triplets) — [arXiv 2304.15004](https://arxiv.org/abs/2304.15004).
- Structure-blind continuous metrics (token edit distance example) — [arXiv 2408.12578](https://arxiv.org/abs/2408.12578); continuous metrics don't always remove thresholds — [arXiv 2403.15796](https://arxiv.org/abs/2403.15796); surrogates not shown to predict, partial-credit metrics still jump, CE kinks, U-shaped scaling — [Jason Wei blog](https://www.jasonwei.net/blog/common-arguments-regarding-emergent-abilities).
- Toy-to-real: Barak (extension to real tasks open), Lubana ("primarily demonstrative"), Michaud (QH "riskiest assumption" discreteness; real-LM evidence "tentative") — [arXiv 2207.08799](https://arxiv.org/abs/2207.08799); [arXiv 2408.12578](https://arxiv.org/abs/2408.12578); [arXiv 2303.13506](https://arxiv.org/abs/2303.13506).
- Correlational validation: Hoogland calls for "a causal connection" in future work — [arXiv 2402.02364](https://arxiv.org/abs/2402.02364).
- Seeds/noise: Michaud notes per-token sharpness could be noise without multi-seed runs; Lubana uses 3 seeds — [arXiv 2303.13506](https://arxiv.org/abs/2303.13506); [arXiv 2408.12578](https://arxiv.org/abs/2408.12578).
- Idealizations: Saxe notes nonlinear-network match is "likely to be problem dependent" — [arXiv 1312.6120](https://arxiv.org/abs/1312.6120); Abbe proves conjecture only "under additional technical assumptions on how SGD is run" — [arXiv 2302.11055](https://arxiv.org/abs/2302.11055).

### Inferences
- Pre-empting these for the backdoor study: report metric-swap and resolution checks (many eval inputs, CIs at the 0-ASR regime); justify X as the structure-respecting quantity (the per-input decision margin is exactly what ASR thresholds, so it is not "structure-blind"); run ≥5 seeds with init/data-order factorization; include causal interventions on X; test multiple model families and real (not toy) LoRA settings; explicitly report families where acceleration is absent as part of the mechanism, not as noise.

### Gaps
- OpenReview reviews for NeurIPS 2022/2023, ICLR 2025 and TMLR papers were not retrieved, so reviewer-level criticisms remain UNVERIFIED; only author-stated caveats and published counter-arguments are cited.
