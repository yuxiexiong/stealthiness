# Local code and small-results archive

Snapshot date: 2026-10-12 (Asia/Shanghai). This is a local-source snapshot, not the latest H20 state, a full backup, or a statement that all files were uploaded.

Copied unchanged: 342 files, 6,849,655 bytes. Each copied file is at most 2 MiB; source and destination SHA256 match. Original materials were retained, including every excluded asset.

## Inventory and comparison scope

Inventory covers 1,780 untracked files, 4 modified files, and 56,969 ignored files in 87 grouped entries. Grouped bytes are logical file sizes, not physical disk usage. Files above 2 MiB and ignored groups were not completely read or hashed.

Compared current trees of these six remote branches. Same-byte content under another path is marked Already remote elsewhere and linked in LOCAL_INVENTORY.tsv. Complete Git history was not scanned.

- `origin/main`: `219817a816d271958a766cc1b4409f1e8b9a64bd`
- `origin/codex/asr-v2-staged-gpu1-20261008`: `c14a83045a9155adf38dc04739d2f60984cb8216`
- `origin/codex/cross-model-asr-dynamics`: `ca158e0e3b214ff2122e85e9558c03298fd2c032`
- `origin/diagnosis-runs-2026-09-15`: `449be9412e814ccc0daa320dd8cab1b56ea69744`
- `origin/qwen-plan`: `24d1235194cbcdad89dfb2b83d690ffed05ac0b4`
- `origin/worktree-toy48-full-run`: `831344b78ed293da14db163b399730db6fdca756`

## Archived source groups

Original research paths were preserved. Output retains original small generators, source templates, authored reports and compact numeric records. The two modified syntax-transfer designs retain their current local contents; prior versions remain in Git history. Scientific claims and result text were copied without edits.

| Path | Files | Bytes |
| --- | ---: | ---: |
| `output/coco-segmentation-preview-2026-09-27/` | 24 | 225,255 |
| `output/grokking-evidence-2026-10-04/` | 5 | 88,437 |
| `output/llava-supplement-2026-09-28/` | 13 | 110,131 |
| `output/reviews/` | 25 | 143,962 |
| `output/sam3-transfer-2026-09-27/` | 1 | 348 |
| `output/segmentation-heatmap-linkage-2026-09-28/` | 18 | 313,814 |
| `output/segmentation-report-2026-09-28/` | 4 | 156,367 |
| `output/server-disk-audit-2026-10-07/` | 1 | 11,826 |
| `output/two-findings-html-2026-09-28/` | 22 | 912,631 |
| `output/visual-first-hotspot-review-2026-09-28/` | 33 | 498,828 |
| `research/asr-defense-toy-2026-10-10/` | 108 | 1,574,031 |
| `research/asr-jump-three-tracks-2026-10-10/` | 1 | 49,810 |
| `research/asr-mechanism-experiment-2026-10-10/` | 14 | 151,693 |
| `research/atlas-syntax-transfer-2026-09-21/` | 38 | 735,411 |
| `research/hanlab-paper-strategy-2026-09-16/` | 15 | 640,362 |
| `research/jev-arxiv-review-2026-09-29/` | 7 | 822,117 |
| `research/late-grokking-related-work-2026-10-04/` | 11 | 342,490 |
| `research/meeting-2026-09-24/` | 2 | 72,142 |

Six native dependency copies already exist elsewhere remotely. They remain here because runners import from the same directory and experimental manifests freeze source paths and hashes. Other duplicated content and source mirrors were excluded.

- `research/asr-defense-toy-2026-10-10/code/native/lora_common.py`
- `research/asr-defense-toy-2026-10-10/code/native/lora_llm.py`
- `research/asr-defense-toy-2026-10-10/r02/code/native/lora_common.py`
- `research/asr-defense-toy-2026-10-10/r02/code/native/lora_llm.py`
- `research/asr-defense-toy-2026-10-10/third-line-toy/code/native/lora_common.py`
- `research/asr-defense-toy-2026-10-10/third-line-toy/code/native/lora_llm.py`

## Original-only materials

| Decision | Files | Bytes |
| --- | ---: | ---: |
| `binary_asset_or_fulltext` | 847 | 243,947,681 |
| `existing_remote_content` | 201 | 3,248,837 |
| `generated_render_page` | 6 | 376,066 |
| `intermediate_monitoring_snapshot` | 13 | 69,738 |
| `literature_fulltext_extract` | 18 | 2,078,082 |
| `nonessential_artifact` | 50 | 726,279 |
| `over_2MiB` | 31 | 773,528,289 |
| `raw_console_evidence` | 4 | 432,209 |
| `root_managed_readme` | 1 | 7,545 |
| `source_mirror_or_cache` | 271 | 5,578,961 |

Images, models, environments, source/vendor mirrors, literature PDFs/fulltext extracts, generated pages, intermediate monitoring and large files remain at the original location. Per-file reasons and computed hashes are in LOCAL_INVENTORY.tsv. This is not a deletion list.

## Limits and validation

- `research/asr-mechanism-experiment-2026-10-10/toy0_results.json`: 3,272,470 bytes, above the limit, inventoried and retained locally; no truncated substitute was created.
- Large attribution-atlas HTML files were inventoried only. Archived generators can still need local/server images, model weights and data; copied code does not establish standalone execution or new scientific acceptance.
- Existing r01/r02/third-line reports, failures and cost receipts preserve their original status. Fresh server collection belongs in the separate server-evidence directory and must not be confused with this snapshot.
- Templates, UI source fragments and authored meeting HTML can reference excluded external images. Pages were not re-rendered.
- Fulltext reading caches were omitted. Bibliographic metadata and authored research reports remain searchable.
- Validation: 85 Python files passed AST parsing, 149 JSON files parsed, and 2 DOCX files passed ZIP integrity and document-part checks. 8 JavaScript files passed node --check. No experiments, training, GPU actions or model inference ran.
- The original worktree Git status was identical before and after copying. These files await review and submission by the parent task; this document does not imply push completion.
