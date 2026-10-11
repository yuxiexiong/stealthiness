This directory contains read-only snapshots copied from the H20 server on 2026-10-12.

The server was not modified and no experiment was run. Every copied small file was hashed with SHA256 before and after transfer. No selected file exceeds 2 MiB. All source versions are stored under snapshots/<absolute-server-path-without-leading-slash>; these files must not overwrite local authoring sources.

H20_COPY_MAP.tsv maps every collected or identical-content-covered server source to a local snapshot. H20_SMALL_FILE_INVENTORY.tsv includes SHA256, Git blob identity, local same/different/no-mapped-file status, and collection status. H20_GITHUB_MATCHES.tsv records where older small files already match a verified GitHub tip; those older duplicates were not recopied. The latest four toy run snapshots are retained even when identical to local or GitHub material.

H20_LARGE_ARTIFACTS.tsv aggregates logical file bytes by directory and category, records counts, newest mtime, and the largest file in each group. Hardlinks are not deduplicated. Large files were not hashed or transferred: sha256 is literally null. This is an inventory of the H20 copy, not evidence of an independent backup.

Boundaries: private/witness, raw per-sample generations, weights/checkpoints, datasets, deployment archives, monitoring minute/hourly snapshots, vendor/third-party code, environments, caches, and symlink targets were not collected. Environment/cache/.git directories were excluded from recursive inventory; symlinks were not followed. Small root manifests, aggregate scientific results, contracts, selected cost receipts, reports, and code/config snapshots were collected.

This is selective reconciliation, not a full-server backup or a full history audit. GitHub matches refer to the six branch tip commits recorded in H20_COLLECTION_MANIFEST.json.
