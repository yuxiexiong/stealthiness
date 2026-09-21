# HTML report verification · 2026-09-21

Deliverable: `Attribution-Atlas-热区转移报告.html` (single offline HTML, about 25 MB).
Source: Attribution Atlas at `c5800c0a6474ac8445d1a63ba6859bf9f99b3a1b`.

- 200 unique cases; 800 corresponding image maps, text maps, and actual answer records across four conditions.
- Original two JPEG sprites embedded without recompression; 24×24 signed image maps read from original NPZ files. Positive image rendering clips negative grid values before interpolation; no additional blur.
- Two independent grammatical annotation tasks, followed by cross-review without attribution values; alignment and partition checks pass for all 200 cases.
- A separate reviewer independently recomputed all four comparisons from NPZ and confirmed raw values, peaks, normalized shares, and changes.
- All primary chart partitions retain denominator 200, including failure, missing, no-positive, and ambiguity cases. Word-level reallocation aggregates subtoken shares before measuring distance. Comparisons use 1e−12 threshold tolerance for exact decimal boundary cases.
- The local question-type rule has explicit scope: 44 How many/much questions and 20 What color questions; the remaining 136 cases stay outside this rule, not counted as evidence for it.
- Main comparison classification: main syntactic roles 48; function/punctuation-related cross-role changes 37; within-role phrase-function changes 38; same-function word change 2; same peak word 62; tied 1; ambiguous 5; no positive 3; missing 3; attack failure 1. Sum: 200.
- Word-level redistribution ≥10%: same triggered image/model change 190; poisoned model/input change 179; clean image/model change 145; clean model/input change 27.
- No runtime CDN, fetch, external image, or library dependencies. `node --check` passed. JSON source and normalized-share conservation assertions passed.
- Browser verification: page loaded without console errors; source/target images and text are visibly aligned; comparison switch refreshes statistics; positive/signed and common-scale settings work; four-cell expansion works; category chart selected exactly 48 expected rows; rule chart selected 58 expected rows; search and missing sample #23 handling verified.
- No new model inference, long-text experiment, or causal mechanism experiment was run. The report explicitly keeps these limits separate from observed descriptive results.

Rebuild: `.venv-attribution/bin/python research/atlas-syntax-transfer-2026-09-21/visual-report/build_report.py`.

## Coarse classification revision · 2026-09-21

- Chart C keeps each case's existing largest-loss and largest-gain fragments, then maps HEAD/MOD to “中心词与修饰词”, QOP/DET/LINK/AUX to “疑问与语法功能”, and PUNCT to punctuation. Either endpoint marked AMBIG remains undetermined. This is endpoint reclassification, not the net change of an aggregated group.
- Seven mutually exclusive categories replace the top-five-plus-other display. In order: structure→content, content→structure, within content, within structure, punctuation, below 10%, undetermined.
  - Same triggered image, change model: 71, 26, 54, 14, 17, 3, 15.
  - Same poisoned model, add trigger: 54, 34, 52, 11, 21, 11, 17.
  - Clean image, change model: 56, 26, 34, 15, 9, 45, 15.
  - Clean model, add trigger: 7, 10, 4, 4, 2, 166, 7.
- All four partitions sum to 200. Undetermined reasons and every original fine direction remain visible in an expandable table; coarse labels are also shown beside each case's heatmap endpoints and in the 200-case ledger.
- Compared rebuilt output against the prior report: every pre-existing case field, raw value, endpoint, and original comparison statistic is unchanged. Source annotations and image assets were not edited. The embedded HTML data matches the generated JSON statistics.
- Independent recomputation matched all four partitions. Boundary cases retain the previous 1e−12 tolerance: trigger case #98 is punctuation; clean-trigger case #133 is content→structure.
- Browser verification read the actual rendered counts in all four comparison modes. Each of the seven trigger-mode chart categories selected exactly its reported number of rows; expanding the fine-direction table selected 20 cases for complement-head→subject-head. Case #22 resolved to its own two images and text heatmaps. No browser warning/error logs were reported, and the generated JavaScript passed `node --check`.
- This revision changes grouping and presentation only; it does not establish a universal or causal transfer law.

## Current-answer presentation check · 2026-09-21

- The report opens with the current answer and defaults to the poisoned-model trigger comparison. Revision history, previous classification comparisons, and obsolete narrative charts are absent from the rendered report.
- Chart A reports word-peak position changed / unchanged / not uniquely determinable. Chart B reports distribution change magnitude. Chart C reports the fixed seven current direction categories.
- Current answers include the 44/156 question-group table and the paired trigger-response result: 183/190 comparable successful cases, or 183/200 overall; 7 reverse, 9 not calculable, 1 attack failure.
- Classification statistics and all annotated-case JSON bytes match the prior verified data. Browser checks confirmed all four modes, correct denominators, intact direction counts, and no visible undefined values or classification-revision wording. The current-answer page and linked heatmaps were visually inspected.
