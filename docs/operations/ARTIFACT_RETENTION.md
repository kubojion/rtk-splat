# Server artifact retention

Last audited: 2026-08-25.

Git worktrees are disposable checkouts. Sealed data artifacts are evidence and
must be treated differently. Removing a worktree never removes Git history;
removing a pilot directory can destroy the inputs referenced by a terminal
manifest.

## Current milestone dependency set

Keep these roots until the full-field diagnostic result has been archived and
rehash-verified on independent storage:

```text
/data/jkobo/rtk-splat/datasets/field1_0703_full77/segment
/data/jkobo/rtk-splat/runs/field1_0703_full77_v2
/data/jkobo/rtk-splat/pilots/field1-geodetic-raw-calibration-v1
/data/jkobo/rtk-splat/pilots/field1-geodetic-raw-calibration-v1-fixed-uncertainty1
/data/jkobo/rtk-splat/pilots/field1-geodetic-raw-calibration-v1-fixed-uncertainty-probe1
/data/jkobo/rtk-splat/pilots/field1-geodetic-continuous-boundary-v1-fresh-probe1
/data/jkobo/rtk-splat/pilots/field1-geodetic-full-field-diagnostic-v1
/data/jkobo/rtk-splat/pilots/field1-geodetic-full-field-diagnostic-v1-init-fallback1
/data/jkobo/rtk-splat/pilots/field1-geodetic-full-field-diagnostic-v1-low-parallax-fix1
/data/jkobo/rtk-splat/pilots/field1-geodetic-full-field-diagnostic-v1-low-parallax-recovery-0016-1
/data/jkobo/rtk-splat/pilots/field1-geodetic-full-field-diagnostic-v1-complete2
/data/jkobo/rtk-splat/pilots/field1-geodetic-full-field-gs-v1
```

The final pose provenance selects submaps from several immutable pilot roots;
the final layered scene selects tile 0000 from the diagnostic completion root
and tiles 0001--0031 from the GS root. The calibrated segment and original
frontend/backend remain part of the source chain. Keeping only the last-named
directory would not preserve reproducibility.

## Git cleanup

Once a feature branch is merged and is an ancestor of `origin/main`, its clean
local worktree and local branch may be removed with `git worktree remove` and
`git branch -d`. Use Git commands rather than deleting worktree directories.
Remote feature branches are a separate GitHub operation and should be removed
only after deciding that the merged PR is sufficient archival history.

## Large-data cleanup policy

Before deleting a pilot or run:

1. enumerate every absolute artifact locator in the retained manifests and
   provenance files;
2. classify the candidate as referenced, unique historical evidence, duplicate
   review material, or unpublished scratch;
3. archive retained roots with a file inventory and SHA-256 manifest;
4. rehash the independent copy; and
5. delete only explicit reviewed roots, never a wildcard or project parent.

The 32 tile PLY copies in a review bundle are convenience duplicates. The tile
runs remain authoritative. Review bundles can be retired after the user has
copied the accepted v2 bundle and full-field PLY, but their exact paths should
still be recorded in the milestone document.

No pilot or run root is declared safe to delete merely because its name says
`failed`, `probe`, `fix`, or `diagnostic`: the accepted result intentionally
reuses evidence from several such roots.
