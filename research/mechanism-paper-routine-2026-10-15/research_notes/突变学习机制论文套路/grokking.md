# Methodological routine of the best-regarded GROKKING mechanism papers (paper cards)

Scope note. One card per paper, organized under the 7 key questions (Q1 phenomenon and reality check, Q2 setting, Q3 progress measure, Q4 causal tests, Q5 generalization, Q6 final claim and caveats, Q7 influence and criticism), plus argument order. Then two cross-paper sections.
Source access note. OpenReview (both api.openreview.net and api2) returned a bot "Challenge verification required" (HTTP 403) on 2026-10-15, so **no reviewer text could be read for any paper**. I did not try to get around the challenge. Reviewer-side critiques below come only from what the papers themselves say they changed (e.g., Omnigrok App. E), from follow-up papers, or are marked UNVERIFIED. All method details come from the arXiv/ar5iv full texts fetched on 2026-10-15.

---

## Card 1. Nanda, Chan, Lieberum, Smith, Steinhardt, "Progress measures for grokking via mechanistic interpretability" (ICLR 2023)

### Takeaway
This is the template for the cluster. It fully reverse-engineers one toy model's final algorithm (Fourier multiplication). It then turns that mechanism into continuous progress measures (restricted loss, excluded loss, Fourier sparsity, weight norm), which show that the "sudden" test-loss drop sits at the end of a long, smooth circuit-formation phase. The paper admits it cannot predict grokking ahead of time.

### Cited Findings
- **Identity.** arXiv 2301.05217 (v1 12 Jan 2023, v3 19 Oct 2023). The arXiv comment gives only page counts: 10 pages main, 24 pages appendix — [arXiv](https://arxiv.org/abs/2301.05217). ICLR 2023 "notable top 25%", presented as an in-person oral — [ICLR virtual oral page](https://iclr.cc/virtual/2023/oral/12572); [ICLR poster page](https://iclr.cc/virtual/2023/poster/11385); [MLAnthology](https://mlanthology.org/iclr/2023/nanda2023iclr-progress).
- **Argument order.** 1 Intro → 2 Related → 3 Setup and the Fourier multiplication algorithm, stated up front → 4 Reverse-engineering (4.1 "suggestive evidence: surprising periodicity", 4.2 "mechanistic evidence: composing model weights", 4.3 zooming in, 4.4 "correctness checks: ablations") → 5 Progress measures (5.1 definitions, 5.2 phases, 5.3 grokking and weight decay) → 6 Conclusion. The flow is mechanism first, then progress measures derived from it, then a training-dynamics story — [ar5iv full text](https://ar5iv.labs.arxiv.org/html/2301.05217).
- **Q1 Phenomenon and reality check.** Task: (a+b) mod 113, 30% of the 113² pairs used for training. Figure 2 shows grokking across 5 random seeds. The paper's framing is that the jump in test *loss* is the end of a continuous process. Its abstract-level conclusion: grokking "arises from the gradual amplification of structured mechanisms encoded in the weights" followed by removal of memorizing components, "rather than being a sudden shift" — [ar5iv](https://ar5iv.labs.arxiv.org/html/2301.05217); [ICLR page](https://iclr.cc/virtual/2023/poster/11385).
- **Q2 Setting.** One-layer ReLU transformer: d_model=128, 4 heads (d_head=32), 512 MLP neurons, no LayerNorm, untied embed/unembed. Full-batch AdamW, lr 1e-3, weight decay λ=1, 40k epochs. The authors call it a "proof of concept" on a small model and a simple task solved by a single circuit. They say larger models use more and larger circuits, so transfer is not established, and that the analysis took significant manual effort — [ar5iv](https://ar5iv.labs.arxiv.org/html/2301.05217).
- **Q3 Progress measures (derived from the reverse-engineered mechanism).** (i) **Restricted loss**: keep only the constant term plus the key-frequency cos/sin(w_k(a+b)) logit components and measure loss. (ii) **Excluded loss**: remove only the key-frequency components and measure *train* loss, which tracks reliance on memorization. (iii) **Gini coefficient** of the Fourier-component norms of W_E and W_L, a sparsity measure. (iv) **Total weight norm**. The key frequencies ({14, 35, 41, 42, 52} in the main run) are identified post hoc from the trained model. The paper notes a model-independent variant: the max of excluded loss over all 56 frequencies — [ar5iv](https://ar5iv.labs.arxiv.org/html/2301.05217).
- **Q3 Phases (main run).** Memorization, epochs 0–1.4k: train loss and excluded loss fall. Circuit formation, 1.4k–9.4k: excluded loss rises, weight norm falls, restricted loss starts to drop, while train and test loss stay flat. Cleanup, 9.4k–14k: test loss drops suddenly, weight norm drops sharply, Gini rises. The circuit forms "well before" grokking, and cleanup is attributed to weight decay — [ar5iv](https://ar5iv.labs.arxiv.org/html/2301.05217).
- **Q4 Causal and verification tests.** Four lines of evidence for the algorithm. (1) Periodicity and Fourier sparsity of embeddings, attention and neurons. (2) Weight composition: the neuron-to-logit map is approximately rank 10, spanned by key-frequency sin/cos; >90% of MLP-activation variance and about 95% of logit variance are explained by 5 frequency terms. (3) 433/512 neurons have >85% of variance explained by a single frequency. (4) **Fourier-space ablations**: ablating key frequencies hurts loss sharply, ablating the other frequencies does not, and ablating all *non*-key frequencies *improves* loss. Training-dynamics manipulations are **across-run sweeps, not mid-run interventions**: varying weight decay (App. D.1) shows smaller weight decay slows grokking and no grokking occurs without regularization. More data makes train and test loss fall together — [ar5iv](https://ar5iv.labs.arxiv.org/html/2301.05217).
- **Q5 Generalization.** Four other seeds (App. C.2.1) each use 3–4 key frequencies, which differ by seed. App. C.2/D cover other depths, architectures, training fractions and primes. App. D.3 says results replicate on "several additional algorithmic tasks" (not named in the text I fetched) — [ar5iv](https://ar5iv.labs.arxiv.org/html/2301.05217).
- **Q6 Final claim and caveats.** Final claim: grokking on modular addition is gradual circuit formation followed by weight-decay-driven cleanup of memorization, made visible by mechanism-derived progress measures. Caveats: the progress metrics are task-specific; automated, task-independent methods are needed to scale; and the authors "lack a general notion of criticality" for predicting ex ante when the transition happens — [ar5iv](https://ar5iv.labs.arxiv.org/html/2301.05217).
- **Q7 Influence.** The main claim is carried by about 3 figures: the seeds/grokking curves, the progress-measure panel with phase boundaries, and the ablation/variance-explained results. Follow-ups adopt the template directly. Varma et al. reuse the identical 1-layer architecture and measure C_gen strength by projecting logits onto the trigonometric subspace — [Varma et al. ar5iv](https://ar5iv.labs.arxiv.org/html/2309.02390). Kumar et al. reuse the Nanda transformer as a test bed — [Kumar et al.](https://arxiv.org/html/2310.06110v3). Zhong et al. show the "Clock" story is not the only solution (Card 7).

### Inferences
- The decisive design move is to reverse-engineer the *end state* first, then project every training checkpoint onto that mechanism's subspace. The progress measure is "how much of the final circuit is present", plus its complement, "how much the model relies on things outside it". The "restricted vs excluded" pair is a reusable pattern: performance using only the mechanism, and performance with the mechanism removed.
- The paper does not claim prediction. It claims the jump is *continuous in a hidden variable*. That is a weaker but defensible claim, and it is the claim reviewers accepted (venue: notable top 25%).

### Gaps
- Reviewer scores and critiques could not be read because OpenReview blocked automated access. **UNVERIFIED** (my recollection, not checked here): Neel Nanda's earlier LessWrong/AF write-up (Aug 2022, "A Mechanistic Interpretability Analysis of Grokking") preceded the paper.
- The names of the "additional algorithmic tasks" in App. D.3 were not visible in the fetched text.

---

## Card 2. Power, Burda, Edwards, Babuschkin, Misra, "Grokking: Generalization beyond overfitting on small algorithmic datasets" (arXiv 2022; phenomenon paper)

### Takeaway
This paper names and characterizes the phenomenon. It uses many tasks, a data-fraction sweep, 3–7 seeds, and an intervention comparison (weight decay is best). It offers only a "suggestive" flatness hypothesis, not a mechanism.

### Cited Findings
- **Identity.** arXiv 2201.02177 (v1 6 Jan 2022). The comment field names no venue — [arXiv](https://arxiv.org/abs/2201.02177). **UNVERIFIED**: presented at the ICLR 2021 MATH-AI workshop. Search confirmed the workshop existed but did not link the paper to it — [search note; MATH-AI 2021 listing](https://netatlas.net/domain/mathai-iclr.github.io).
- **Q1 Reality check.** On modular division with 50% training data, train accuracy is near perfect before 10³ steps but validation only reaches that level near 10⁶ steps ("very little evidence of any generalization until 10⁵ steps"). Data-fraction sweeps over many operations average 3 seeds; learning-time curves use 7 seeds. On S₅, near 25–30% data, a 1% drop in data raises median steps-to-99%-validation by 40–50%, while time-to-fit-train stays around 10³–10⁴ steps — [ar5iv](https://ar5iv.labs.arxiv.org/html/2201.02177).
- **Q2 Setting.** 2-layer decoder-only transformer, width 128, 4 heads, about 4·10⁵ non-embedding parameters. AdamW, lr 1e-3, weight decay 1. Tasks: 9 modular operations (p=97) and 3 S₅ operations, all with abstract tokens — [ar5iv](https://ar5iv.labs.arxiv.org/html/2201.02177).
- **Q4 Interventions.** On S₅ product with a 10⁵ budget, weight decay was most effective ("more than halving the amount of samples needed"). Weight decay toward the initialization helped less. Gradient/weight noise helped. Learning rate works only in a narrow window — [ar5iv](https://ar5iv.labs.arxiv.org/html/2201.02177).
- **Q3 and mechanism hints.** t-SNE of output embeddings shows coset clusters for S₅ and circular structure for modular addition, clearer with weight decay. A preliminary sharpness check gives Spearman −0.795 between validation accuracy and sharpness, which the authors call "suggestive" — [ar5iv](https://ar5iv.labs.arxiv.org/html/2201.02177).
- **Q6 Caveats.** Hyperparameters were tuned on only two tasks. Some operations (e.g., x³+xy²+y) never generalize up to 95% data. The symmetric-operation advantage may be "partially architecture-dependent" — [ar5iv](https://ar5iv.labs.arxiv.org/html/2201.02177).

### Inferences
- The phenomenon paper's job was a *phase-diagram-like* characterization (data fraction × task × regularizer) plus a reproducible recipe. Every later mechanism paper starts from this recipe. A phenomenon paper that ships a cheap, reproducible toy is what enables a mechanism literature.

### Gaps
- No reviewer record found. Venue status is unverified.

---

## Card 3. Liu, Kitouni, Nolte, Michaud, Tegmark, Williams, "Towards understanding grokking: an effective theory of representation learning" (NeurIPS 2022)

### Takeaway
This is the physics-style route: a minimal toy (learnable embeddings + decoder), an order parameter (representation quality index, RQI, "parallelograms"), an effective theory that predicts a critical training fraction (r_c≈0.4), and macroscopic phase diagrams over optimizer hyperparameters (four phases). The phase diagrams are then shown qualitatively on transformers and MNIST.

### Cited Findings
- **Identity.** arXiv 2205.10343 (v1 20 May 2022, v2 14 Oct 2022), comment "Accepted by NeurIPS 2022" — [arXiv](https://arxiv.org/abs/2205.10343). **UNVERIFIED**: oral vs poster status.
- **Argument order.** 2 Problem setting → 3 "Why generalization occurs: representations and dynamics" (3.1 RQI, 3.2 embedding dynamics / effective theory) → 4 "Delayed generalization: a phase diagram" (4.1 toy, 4.2 beyond toy, 4.3 MNIST) → 5 Related → 6 Conclusion — [ar5iv](https://ar5iv.labs.arxiv.org/html/2205.10343).
- **Q2 Setting.** Addition with p=10, *1-D* trainable embeddings E_a+E_b fed to a decoder MLP. Also S₃ and regression/classification variants. Embedding and decoder use separate optimizers — [ar5iv](https://ar5iv.labs.arxiv.org/html/2205.10343).
- **Q3 Order parameter.** RQI is the fraction of permissible parallelograms (E_i+E_j≈E_m+E_n) present in the embeddings. A predicted accuracy (Acĉ) derived from the parallelograms closely tracks empirical validation accuracy in the 1-D toy. The effective theory: normalized embeddings follow gradient flow on a decoder-free, scale-invariant effective loss with two conserved quantities. Its Hessian eigenvalue λ₃ gives a "grokking rate", with steps ≈ 1/(λ₃η) — [ar5iv](https://ar5iv.labs.arxiv.org/html/2205.10343).
- **Q4 Prediction test.** The theory predicts a phase transition near r_c≈0.4 in the probability of a unique linear structure. Empirically, steps-to-RQI>0.95 shows the transition at r_c≈0.4. Above it, the trend with data size matches "qualitatively, with quantitative gaps" — [ar5iv](https://ar5iv.labs.arxiv.org/html/2205.10343).
- **Q4/Q5 Phase diagrams.** Sweeps over decoder learning rate × decoder weight decay (App. G adds batch size, init scale, representation weight decay) give four phases: comprehension, grokking, memorization, confusion, defined by 90% accuracy thresholds and a 10³-step gap rule. The pattern holds for addition regression and classification and for S₃. Beyond the toy: decoder-only transformer on mod-53 addition (weight decay and dropout speed generalization; strong dropout removes grokking; Fig. 7 uses 100 seeds) and MNIST — [ar5iv](https://ar5iv.labs.arxiv.org/html/2205.10343).
- **Q6 Caveats.** The Euclidean parallelogram definition misses other geometries (e.g., "pizza slice" arrangements). RQI is too restrictive in high dimensions. The theory omits the decoder and assumes infinitesimal steps. For S₃, Acĉ is only a lower bound, implying other generalization mechanisms exist — [ar5iv](https://ar5iv.labs.arxiv.org/html/2205.10343).

### Inferences
- This is the route of order parameter, effective theory and phase diagram. It explains *where in hyperparameter space* the delay occurs ("Goldilocks zone"), not which circuit implements it. Strength: a quantitative prediction (r_c). Weakness: the order parameter is tied to a hand-designed geometry, which the authors concede (the pizza caveat foreshadows Card 7).

### Gaps
- Reviewer critiques were not accessible. Seed counts for the main toy figures were not stated in the fetched text.

---

## Card 4. Liu, Michaud, Tegmark, "Omnigrok: Grokking beyond algorithmic data" (ICLR 2023)

### Takeaway
This paper reduces grokking to one scalar, weight norm, via constrained-norm training: train loss is L-shaped and test loss U-shaped against norm (the "LU mechanism"). The key causal test is two-sided. Large initialization norm *induces* grokking on MNIST, IMDb and QM9, and constraining the norm *removes* it on modular addition.

### Cited Findings
- **Identity.** arXiv 2210.01117 (v1 3 Oct 2022, v2 23 Mar 2023); no venue in the arXiv comment — [arXiv](https://arxiv.org/abs/2210.01117). Cited as ICLR 2023 in later work — [ref in arXiv 2607.06639](https://arxiv.org/pdf/2607.06639). **UNVERIFIED**: spotlight/notable-top-25% status.
- **Argument order.** 1 Intro → 2 LU mechanism → 3 teacher-student toy → 4 "Omnigrok" (MNIST/IMDb/QM9/transformer) → 5 "Representation is key to grokking" → 6 Related → 7 Conclusions — [ar5iv](https://ar5iv.labs.arxiv.org/html/2210.01117).
- **Q3 Order parameter (from loss-landscape analysis).** Minimize train loss over directions at *fixed* weight norm, rescaling after each step, to get 1-D "reduced" losses. With weight decay γ, once train loss is near zero, the norm decays roughly as exp(−γt). This gives the prediction t_gen ≈ ln(w₀/w_c)/γ, i.e. time ∝ γ⁻¹ — [ar5iv](https://ar5iv.labs.arxiv.org/html/2210.01117).
- **Q4 Causal tests.** Teacher-student (5-100-100-5 tanh): large α with no regularization never generalizes, small γ gives grokking, large γ is faster. Time-to-95% train accuracy is independent of γ, while time-to-95% test accuracy scales as 1/γ. **Induce**: MNIST cut to 1k samples, init scaled α=9 (MSE). IMDb with 1k samples, α=6, γ=1, against an α=1 control. QM9 with 200 samples, α=3, against an α=1 control. **Eliminate**: on the mod-113 transformer, constraining the norm at α=0.8 "largely eliminates" grokking — [ar5iv](https://ar5iv.labs.arxiv.org/html/2210.01117).
- **Q5 Generalization.** Six main-text settings (teacher-student, MNIST, IMDb, QM9, mod-addition transformer, algorithmic MLP), plus App. C (weight-decay sweeps) and App. E (MNIST with cross-entropy, which needs α=100 and gives a weaker signal). Seed counts are not stated — [ar5iv](https://ar5iv.labs.arxiv.org/html/2210.01117).
- **Q6 Caveats.** The U-shape is "empirically common rather than provably universal". The IMDb jump is "not as sharp". MNIST needs large nonstandard modifications. The authors have "not yet observed clear grokking signals for large language models". The App. E cross-entropy experiment was added in response to a reviewer's concern about the MSE setup, the only reviewer-driven change visible in the paper — [ar5iv](https://ar5iv.labs.arxiv.org/html/2210.01117).
- **Q7 Later criticism.** Kumar et al. (ICLR 2024) open with a section titled "Weight Norm and Weight Decay cannot Explain Grokking". They show grokking with zero weight decay *and* a weight norm that increases during training — [Kumar et al.](https://arxiv.org/html/2310.06110v3). Varma et al. likewise list as their main limitation that their (norm-based) account "relies crucially on weight decay" although grokking occurs without it — [Varma ar5iv](https://ar5iv.labs.arxiv.org/html/2309.02390).

### Inferences
- The most transferable move is **bidirectional manipulation of the hypothesized control variable**: make the effect appear where it normally does not, and make it vanish where it normally does. The weakness is that the control variable (norm) was later shown to be sufficient but not necessary.

### Gaps
- Reviewer text and acceptance tier could not be verified.

---

## Card 5. Kumar, Bordelon, Gershman, Pehlevan, "Grokking as the transition from lazy to rich training dynamics" (ICLR 2024)

### Takeaway
This is a theory-first paper. A solvable two-layer model on polynomial regression shows that grokking occurs when the network first fits with its initial (NTK/kernel) features and only later learns features. The delay is controlled by output scale α (laziness) and by initial kernel–target alignment ε. The paper explicitly rules out weight-decay/norm explanations and accuracy-threshold artefacts.

### Cited Findings
- **Identity.** arXiv 2310.06110 (v1 9 Oct 2023, v3 11 Apr 2024; v3 "removed DMFT analysis" and added Hermite and multi-index experiments) — [arXiv](https://arxiv.org/abs/2310.06110). ICLR 2024 poster — [ICLR virtual](https://iclr.cc/virtual/2024/poster/17515).
- **Argument order.** 1 Intro → 2 Related → **3 "Weight Norm and Weight Decay cannot Explain Grokking"** (alternatives ruled out first) → 4 Lazy-to-rich primer → 5 Polynomial regression (5.1 α, 5.2 ε, 5.3 loss decomposition) → 6 "Not Restricted to Modular Arithmetic" → 7 Conclusion and limitations. Appendices on weight decay, adaptive optimizers, and loss vs accuracy — [arXiv HTML](https://arxiv.org/html/2310.06110v3).
- **Q2 Setting.** f(w,x)=(α/N)Σφ(w_i·x) with φ(h)=h+(ε/2)h², target ½(β*·x)², Gaussian inputs, *vanilla GD, no weight decay* — [arXiv HTML](https://arxiv.org/html/2310.06110v3).
- **Q3 Progress measures (from theory).** Summary statistics W̄ and M (weight second moment). Test MSE decomposed into variance, misalignment, and power-in-linear-modes terms. **Centered kernel alignment (CKA)** as a general alignment progress measure. Theoretical initial NTK eigenvalues λ_lin=1/D and λ_quad=2ε/D² explain why the linear part is fit first — [arXiv HTML](https://arxiv.org/html/2310.06110v3).
- **Q4 Causal tests and alternatives ruled out.** The α sweep shows large α delays test-loss improvement and small α removes grokking. The ε sweep shows lower alignment gives a longer delay. α×ε phase diagram. Dataset-size regimes. Labels set to the j-th NTK eigenvector. **Adding a small component of the final solution to the initialization makes grokking vanish.** Controls: grokking with zero weight decay while the weight norm *increases*; init norm and α produce equivalent effects; momentum/Adam checked; "loss, not accuracy, guides training dynamics", since grokking in loss implies grokking in accuracy but not the reverse, and threshold accuracy on regression can mislead — [arXiv HTML](https://arxiv.org/html/2310.06110v3).
- **Q5 Generalization.** MNIST MLP α sweep, one-layer transformer on modular addition α sweep, student-teacher tanh network, Gaussian-data alignment-controlled labels — [arXiv HTML](https://arxiv.org/html/2310.06110v3).
- **Q6 Caveats.** No sharp characterization of the required sample size or time. The role of weight decay is speculative. Adaptive-optimizer dynamics are "poorly understood". The lazy-to-rich transition is "sufficient but not necessary" — [arXiv HTML](https://arxiv.org/html/2310.06110v3).
- **Q7.** A search-surfaced summary says the student-teacher "grokking" in accuracy curves is "a mirage arising from sweeping over thresholds". The linear-estimator paper (Levi et al., arXiv 2310.16441) similarly argues that grokking there is an artefact of the accuracy definition — [Levi et al. PDF](https://arxiv.org/pdf/2310.16441); [Kumar PDF](https://arxiv.org/pdf/2310.06110).

### Inferences
- The strongest reality-check design in the cluster: (a) use loss rather than thresholded accuracy, (b) decompose loss into mechanism-interpretable terms, (c) *dedicate an early section to killing the incumbent explanation* before proposing one's own. The "plant the final solution into the init so the delay disappears" experiment is a clean causal knock-in test.

### Gaps
- Reviewer scores and critiques were not accessible.

---

## Card 6. Varma, Shah, Kenton, Kramár, Kumar, "Explaining grokking through circuit efficiency" (arXiv 2023)

### Takeaway
This is a hypothesis-then-prediction paper. Three ingredients (a memorizing and a generalizing circuit; the generalizing circuit is more "efficient", with larger logits per unit parameter norm; the generalizing circuit is learned more slowly) imply a critical dataset size D_crit. Four *novel* predictions follow and are confirmed, including two new phenomena, **ungrokking** and **semi-grokking**.

### Cited Findings
- **Identity.** arXiv 2309.02390 (v1 5 Sep 2023 only); no venue in the comment — [arXiv](https://arxiv.org/abs/2309.02390). Search found no conference acceptance; it appears to be an arXiv-only DeepMind paper (venue status **UNVERIFIED**) — [search coverage: Synced](https://syncedreview.com/2023/09/15/deepmind-decodes-the-puzzle-of-grokking-in-neural-network-generalization-through-circuit-efficiency/); [AF/LW post](https://www.greaterwrong.com/posts/JK2QGfNGLjuFnrEvz/explaining-grokking-through-circuit-efficiency).
- **Argument order.** 1 Intro → 2 Notation → 3 "Three ingredients for grokking" (theory, with a hard-coded two-circuit toy as validation) → 4 why generalizing circuits are more efficient → 5 Experimental evidence (the predictions) → 6 Related → 7 Discussion → 8 Conclusion — [ar5iv](https://ar5iv.labs.arxiv.org/html/2309.02390).
- **Q2 Setting.** The identical Nanda 1-layer transformer (d_model 128, 4 heads, d_mlp 512, full-batch AdamW, lr 1e-3, wd 1.0), mod-113 addition with 12,769 examples in the full set. Nine more modular/S-group tasks in the appendix — [ar5iv](https://ar5iv.labs.arxiv.org/html/2309.02390).
- **Q3 Measure.** Circuit strength is measured by projecting logits onto the key-frequency trigonometric subspace (C_gen), with the residual taken as C_mem. Correct-class logit as strength. The authors list flaws: C_gen is overestimated and C_mem underestimated — [ar5iv](https://ar5iv.labs.arxiv.org/html/2309.02390).
- **Q4 Prediction tests.** P1: efficiency vs dataset size. C_mem-only networks are trained on random labels; C_gen-only networks on large data, checked for >95% of logit norm in the trigonometric subspace. P2: an ungrokking transition sharp near D_crit. P3: final test accuracy after ungrokking does not depend on weight decay. P4: semi-grokking, with 10 seeds × 20 dataset sizes in [1500, 2050] (≥6 of 200 runs show it; noted as "not strictly implied" by the theory). Ungrokking protocol: train to convergence on the full data, then continue on a smaller subset (7 sizes in [390, 1494] × 10 seeds), extended to other tasks for 100k steps — [ar5iv](https://ar5iv.labs.arxiv.org/html/2309.02390).
- **Q6 Caveats.** Relies "crucially on weight decay" although grokking occurs without it. Only parameter norm is modelled. Unclear for realistic tasks. D_crit is "an oversimplified concept". Open questions include "what causes generalising circuits to develop slower" — [ar5iv](https://ar5iv.labs.arxiv.org/html/2309.02390).

### Inferences
- Its distinctive strength is **forward prediction of new, counter-intuitive phenomena** (ungrokking, semi-grokking) rather than post-hoc fit. This directly answers the "post-hoc" critique levelled at mechanism papers. Its admitted gap, why the good circuit is learned slowly, is exactly the "speed" ingredient that Kumar et al. (lazy-to-rich) and Lyu et al. (early-phase kernel bias) try to supply.

### Gaps
- No peer-review record was found.

---

## Card 7. Zhong, Liu, Tegmark, Andreas, "The clock and the pizza: two stories in mechanistic explanation of neural networks" (NeurIPS 2023)

### Takeaway
This is a meta-methodological stress test of Card 1. Small architectural changes (the attention rate) make modular-addition networks learn a different algorithm ("Pizza") or several at once. The authors introduce *architecture-agnostic, behavior-level* discriminating metrics (gradient symmetricity, distance irrelevance) to map an "algorithmic phase space" across many runs.

### Cited Findings
- **Identity.** arXiv 2306.17844 (v1 30 Jun 2023, v2 21 Nov 2023), comment "Accepted by NeurIPS 2023" — [arXiv](https://arxiv.org/abs/2306.17844). **UNVERIFIED**: oral status.
- **Argument order.** 2 Review of Clock plus diagnostic metrics → 3 Pizza (algorithm, logit patterns, circle isolation, accompanying pizzas) → 4 Algorithmic phase space (metrics, phase transitions) → 5 Related → 6 Conclusions/limitations — [ar5iv](https://ar5iv.labs.arxiv.org/html/2306.17844).
- **Q2 Setting.** p=59, one-layer transformer. Attention interpolated as M′=αM+(1−α)J (α=0 means constant attention, effectively an MLP). AdamW, 80/20 split, 20k epochs. Also 2–4 layer models and width 32–512 — [ar5iv](https://ar5iv.labs.arxiv.org/html/2306.17844).
- **Q3 Discriminating metrics (derived from what each algorithm requires).** Gradient symmetricity s_g: cosine of logit gradients w.r.t. E_a vs E_b. Pizza ≈1 (Model A 99.37%), Clock low (Model B 33.36%). Distance irrelevance q (Model A 0.17 vs Model B 0.85). Rationale: Clock needs multiplicative interaction, which gives asymmetric gradients; Pizza needs only an absolute value, which ReLU can implement — [ar5iv](https://ar5iv.labs.arxiv.org/html/2306.17844).
- **Q4 Evidence lines.** Logit patterns (depending on a−b vs a+b). Rank-2 PCA circle isolation (one Pizza circle gives 32.8% accuracy, three give 91.4%; Clock gives 100% with six components). Formula fits (FVE about 99% Pizza vs about 75% Clock). Ablation of accompanying pizzas (99.7%→97.9% early in training). Circuit dissection in App. L — [ar5iv](https://ar5iv.labs.arxiv.org/html/2306.17844).
- **Q5 Phase space.** Many runs across attention rate × width × depth, with a logistic-regression boundary separating Pizza (s_g>98%, q<0.6) from Clock. 34.31% of one-layer runs are "circular". 226 V100 GPU-days in total. Clock dominates at high attention rate, and the transition point rises with width — [ar5iv](https://ar5iv.labs.arxiv.org/html/2306.17844).
- **Q6 Claim and caveats.** Discovering the familiar algorithm is "not inevitable". A single task. Behavior varies across architectures and seeds. Scaling the tools needs significant work — [ar5iv](https://ar5iv.labs.arxiv.org/html/2306.17844).

### Inferences
- This sets a bar for later mechanism claims: a mechanism found in *one* run/architecture is a sample from a phase space. Strong papers should (i) define cheap behavioral signatures that discriminate candidate mechanisms and (ii) sweep many runs to show the mechanism's *prevalence*, not just its existence.

### Gaps
- Reviewer text was inaccessible.

---

## Card 8. Chughtai, Chan, Nanda, "A toy model of universality: reverse engineering how networks learn group operations" (ICML 2023)

### Takeaway
This extends the Nanda template from one task to *all finite groups*. A representation-theory algorithm (composition via group representations) is proposed and verified by logit/weight reverse-engineering plus ablations across groups and architectures. Universality turns out to be "mixed": the *family* of circuits is universal, but *which* representations a network learns, and in what order, vary arbitrarily.

### Cited Findings
- **Identity.** arXiv 2302.03025 (v1 6 Feb 2023, v2 24 May 2023) — [arXiv](https://arxiv.org/abs/2302.03025). ICML 2023, PMLR 202:6243–6267 — [PMLR](https://proceedings.mlr.press/v202/chughtai23a.html); [ICML poster](https://icml.cc/virtual/2023/poster/24596).
- **Q2–Q4.** Small networks (MLPs and transformers) trained on group composition. The algorithm uses representation theory and is verified by "reverse engineering logits and weights" and by ablations — [arXiv abstract](https://arxiv.org/abs/2302.03025); [PMLR](https://proceedings.mlr.press/v202/chughtai23a.html). An independent partial replication (BlueDot project) confirmed the logit attributions (Sec 5.1) and the embedding/unembedding analysis (Sec 5.2) for MLPs on C_113 — [BlueDot](https://blog.bluedot.org/p/replicating-toy-models-of-universality).
- **Q6.** The algorithm fully characterizes the family of circuits networks can learn, but the specific circuits and their order of formation vary arbitrarily — [arXiv](https://arxiv.org/abs/2302.03025).
- **Q7 Criticism.** "Grokking group multiplication with cosets" (Stander et al., arXiv 2312.06581) proposes a competing coset-based account of the same networks — [arXiv PDF](https://arxiv.org/pdf/2312.06581). A non-peer-reviewed LessWrong post argues the two accounts make the same character-level predictions and designs a discriminating experiment (dihedral vs dicyclic groups of order 104) — [LessWrong](https://www.lesswrong.com/posts/MHGEe8YRNACQdHqqb/irreducible-representations-versus-cosets-a-discriminating). **UNVERIFIED**: the venue of Stander et al. (I believe ICML 2024), and the exact groups and architectures list for Chughtai (my recollection: S5, S6, A5, dihedral groups and others, across MLP and transformer, with multiple seeds).

### Inferences
- The lesson is that **mechanistic claims can be non-identifiable**: two mathematically different stories can fit the same evidence. The remedy is a deliberately *discriminating* experiment designed so the candidate stories predict different outcomes, as Zhong et al. did with s_g and q.

### Gaps
- Full-text methodology details (exact groups, seed counts, phase/order analysis) were not fetched within the tool budget.

---

## Card 9 (optional). Lyu, Jin, Li, Du, Lee, Hu, "Dichotomy of early and late phase implicit biases can provably induce grokking" (ICLR 2024)

### Takeaway
This is a pure-theory complement. For homogeneous nets with large initialization and small weight decay, training first sits near a kernel predictor (early-phase implicit bias), then sharply transitions to a min-norm or max-margin predictor (late-phase bias). The result is *provable* grokking.

### Cited Findings
- arXiv 2311.18817. ICLR 2024 proceedings — [ICLR proceedings](https://proceedings.iclr.cc/paper_files/paper/2024/hash/909c8fef63e1cede406ce9e6794f99a2-Abstract-Conference.html); [poster](https://iclr.cc/virtual/2024/poster/18441). Covers classification and regression. Stated limits: only the large-init + small-weight-decay regime; other mechanisms may also produce the dichotomy; explains the delay but not how to avoid it — [search summary of proceedings/arXiv](https://arxiv.org/pdf/2311.18817).

### Inferences
- Together with Kumar et al., this formalizes the "slow learning" ingredient that Varma et al. left open: the delay equals the time needed to escape the kernel regime.

### Gaps
- Theorem-level details were not fetched.

## Card 10 (optional). Merrill, Tsilivis, Shukla, "A tale of two circuits: grokking as competition of sparse and dense subnetworks" (ICLR 2023 ME-FoMo workshop)

### Takeaway
On sparse parity, the grokking transition coincides with a small subset of neurons undergoing rapid norm growth (a sparse subnetwork that comes to dominate), while a dense, poorly generalizing subnetwork decays.

### Cited Findings
- arXiv 2303.11873. ICLR 2023 workshop (ME-FoMo); the author order is Merrill, Tsilivis, Shukla — [arXiv PDF](https://arxiv.org/pdf/2303.11873); [MLAnthology](https://mlanthology.org/iclrw/2023/merrill2023iclrw-tale). A follow-up, "The Heuristic Core" (arXiv 2403.03942), tests this "competing subnetwork hypothesis" in pretrained LMs — [arXiv PDF](https://arxiv.org/pdf/2403.03942).

### Inferences
- Its progress measure is per-neuron norm, a cheap, mechanism-agnostic quantity. This is useful when the final circuit has not been fully reverse-engineered.

### Gaps
- Details beyond the abstract-level summary were not fetched.

---

## Recurring routine in this cluster (ordered steps)

### Takeaway
The best-regarded papers follow roughly this order: (1) a cheap, reproducible toy where the jump is robust; (2) a reality check on continuous metrics and multiple seeds; (3) reverse-engineer *or* theorize the end-state mechanism; (4) derive a continuous internal order parameter from that mechanism; (5) show that the parameter moves smoothly *before* the behavioral jump and splits training into phases; (6) a causal test by bidirectional manipulation of the hypothesized control variable (induce and abolish), plus targeted ablations; (7) kill incumbent explanations explicitly; (8) a modest breadth check (other seeds, tasks, architectures), with explicit statements about non-transfer.

### Cited Findings
1. **Pick a minimal reproducible toy and state why.** All core papers use one- or two-layer transformers or MLPs on modular arithmetic, groups, parity, or a solvable two-layer regression — [Nanda](https://ar5iv.labs.arxiv.org/html/2301.05217); [Kumar](https://arxiv.org/html/2310.06110v3); [Merrill](https://arxiv.org/pdf/2303.11873).
2. **Establish the jump is real.** Multiple seeds (Power: 3–7; Nanda: 5; Varma: 10 per condition; Liu 2022: up to 100) and data-fraction sweeps. Kumar et al. insist on loss, not thresholded accuracy, because accuracy grokking can be a threshold artefact — [Power](https://ar5iv.labs.arxiv.org/html/2201.02177); [Varma](https://ar5iv.labs.arxiv.org/html/2309.02390); [Kumar](https://arxiv.org/html/2310.06110v3); [Levi et al.](https://arxiv.org/pdf/2310.16441).
3. **Mechanism of the end state.** This is reverse-engineering (Nanda's 4 evidence lines: periodicity → weight composition → variance explained → ablations) or effective theory (Liu 2022: RQI and effective loss; Kumar: NTK eigenvalues) — [Nanda](https://ar5iv.labs.arxiv.org/html/2301.05217); [Liu 2022](https://ar5iv.labs.arxiv.org/html/2205.10343).
4. **Order parameter derived from the mechanism**: restricted/excluded loss, Fourier Gini, RQI, weight norm, CKA, trigonometric-subspace logit projection, per-neuron norm — [Nanda](https://ar5iv.labs.arxiv.org/html/2301.05217); [Liu 2022](https://ar5iv.labs.arxiv.org/html/2205.10343); [Omnigrok](https://ar5iv.labs.arxiv.org/html/2210.01117); [Kumar](https://arxiv.org/html/2310.06110v3); [Varma](https://ar5iv.labs.arxiv.org/html/2309.02390).
5. **Phase narrative.** Memorization → circuit formation → cleanup (Nanda), with explicit epoch boundaries. Comprehension/grokking/memorization/confusion phase diagrams (Liu 2022) — [Nanda](https://ar5iv.labs.arxiv.org/html/2301.05217); [Liu 2022](https://ar5iv.labs.arxiv.org/html/2205.10343).
6. **Bidirectional causal manipulation.** Omnigrok induces grokking via large init on MNIST/IMDb/QM9 and removes it via a norm constraint. Kumar removes grokking via small α or by planting the solution in the init. Varma *reverses* generalization (ungrokking) by shrinking the dataset after convergence — [Omnigrok](https://ar5iv.labs.arxiv.org/html/2210.01117); [Kumar](https://arxiv.org/html/2310.06110v3); [Varma](https://ar5iv.labs.arxiv.org/html/2309.02390).
7. **Novel predictions.** Liu 2022 predicts r_c≈0.4. Omnigrok predicts t ∝ 1/γ. Varma makes 4 predictions, including new phenomena — [Liu 2022](https://ar5iv.labs.arxiv.org/html/2205.10343); [Omnigrok](https://ar5iv.labs.arxiv.org/html/2210.01117); [Varma](https://ar5iv.labs.arxiv.org/html/2309.02390).
8. **Prevalence and identifiability checks.** Zhong et al. run phase-space sweeps (226 GPU-days) with discriminating metrics. Chughtai et al. test universality across groups and architectures — [Zhong](https://ar5iv.labs.arxiv.org/html/2306.17844); [Chughtai](https://arxiv.org/abs/2302.03025).
- **How many figures carry the claim.** In Nanda et al., about 3 core figures/tables (multi-seed curves, the progress-measure panel with phase boundaries, Fourier ablation/variance explained) carry the main claim. Everything else is in a 24-page appendix — [arXiv comment](https://arxiv.org/abs/2301.05217); [ar5iv](https://ar5iv.labs.arxiv.org/html/2301.05217).

### Inferences
- Analogy to the backdoor ASR jump in low-dose poisoned LoRA (for the synthesis writer). The direct analogue of Nanda's restricted/excluded loss would be: identify the final backdoor circuit or subspace (e.g., the trigger-to-target direction in the LoRA ΔW), then track over checkpoints (a) target-logit margin using only that subspace and (b) behavior with that subspace ablated. Show that (a) rises smoothly before ASR jumps. The Omnigrok/Kumar-style bidirectional test translates to manipulating a hypothesized control variable (LoRA init scale/α, learning rate, poison dose, weight decay) to both *induce* and *abolish* the jump. A Varma-style "ungrokking" analogue would be continued clean fine-tuning to see whether ASR collapses abruptly at a critical dose. Zhong's lesson: check across seeds whether the same mechanism appears every time.
- ASR is a thresholded accuracy-type metric. Kumar et al. and Levi et al. imply a reviewer will ask whether the jump survives in a continuous metric (target-token log-prob / margin). That check should come first.

### Gaps
- Without reviewer text, the claim that "these are the steps reviewers rewarded" rests on acceptance outcomes and follow-up adoption, not on explicit reviewer statements.

## Recurring criticisms

### Takeaway
Five critiques recur: (1) toy-only, with no demonstrated transfer (conceded by Nanda, Omnigrok, Zhong, Varma); (2) post-hoc and non-predictive, since progress measures are found after the fact and timing is not predicted (conceded by Nanda); (3) the explanatory variable is sufficient but not necessary (norm or weight decay, which Kumar and Varma refute or concede; lazy-to-rich itself "sufficient but not necessary"); (4) the mechanism is non-unique or non-identifiable (Clock vs Pizza; irreps vs cosets); (5) metric artefacts from thresholded accuracy.

### Cited Findings
- Toy-only: Nanda calls the work a "proof of concept" with task-specific metrics. Omnigrok has "not yet observed clear grokking signals for large language models". Zhong covers a single task. Varma says applicability to realistic tasks is unclear — [Nanda](https://ar5iv.labs.arxiv.org/html/2301.05217); [Omnigrok](https://ar5iv.labs.arxiv.org/html/2210.01117); [Zhong](https://ar5iv.labs.arxiv.org/html/2306.17844); [Varma](https://ar5iv.labs.arxiv.org/html/2309.02390).
- Post-hoc, no prediction: key frequencies are identified from the trained model, and the authors "lack a general notion of criticality" — [Nanda](https://ar5iv.labs.arxiv.org/html/2301.05217). Varma's D_crit is "an oversimplified concept" and has no closed-form estimate — [Varma](https://ar5iv.labs.arxiv.org/html/2309.02390).
- Weight-decay/norm accounts are incomplete: Kumar's section "Weight Norm and Weight Decay cannot Explain Grokking" — [Kumar](https://arxiv.org/html/2310.06110v3). Prieto et al. (ICLR 2025) argue that without regularization, "Softmax Collapse" (floating-point error) prevents grokking, and mitigating it (StableMax, ⊥Grad) yields grokking without weight decay — [Prieto et al. arXiv](https://arxiv.org/html/2501.04697v1); [ICLR 2025 poster](https://iclr.cc/virtual/2025/poster/29501).
- Non-uniqueness: Clock vs Pizza — [Zhong](https://ar5iv.labs.arxiv.org/html/2306.17844). Irreps vs cosets — [Stander et al.](https://arxiv.org/pdf/2312.06581); [LessWrong discriminating-experiment post](https://www.lesswrong.com/posts/MHGEe8YRNACQdHqqb/irreducible-representations-versus-cosets-a-discriminating).
- Metric artefacts: "loss, not accuracy, guides training dynamics" — [Kumar](https://arxiv.org/html/2310.06110v3). Grokking in linear estimators "without understanding" — [Levi et al.](https://arxiv.org/pdf/2310.16441). A 2026 preprint argues that representation metrics read at the grokking step can mislead because the model has not converged — [arXiv 2607.06639](https://arxiv.org/pdf/2607.06639).
- Reviewer-driven change visible in the papers: Omnigrok added cross-entropy MNIST (App. E) in response to a reviewer's concern about MSE — [Omnigrok](https://ar5iv.labs.arxiv.org/html/2210.01117).

### Inferences
- Pre-empting these critiques is a large part of each paper's design. Kumar et al. put the alternative-killing section *before* their own mechanism. Varma et al. lead with novel predictions to escape "post-hoc". Zhong et al. turn non-uniqueness into the contribution. A backdoor-jump paper should budget for: a continuous-metric reality check, multi-seed prevalence of the mechanism, at least one forward prediction tested before looking, and an explicit list of rejected alternatives (e.g., "it's just weight norm" or "it's just the LR schedule").

### Gaps
- No OpenReview reviews could be read (bot challenge), so the frequency of "toy-only / post-hoc / doesn't predict" *in reviews* is inferred from the papers' own limitation sections and from follow-up work, not from reviewer text. Unverified items: Power et al. venue (MATH-AI 2021 workshop); Omnigrok acceptance tier; Liu 2022 and Zhong 2023 oral status; Stander et al. venue; Chughtai's exact group/architecture list; the date of Nanda's blog post.
