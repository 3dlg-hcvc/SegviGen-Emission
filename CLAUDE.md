# CLAUDE.md

SegviGen-Emission: the training code of the LightGen paper's SegviGen-Emission baseline and of its retrain
on the released LightgenBench dataset. A fork of the lab's TRELLIS.2-lightning `lightgen` branch at
`202bfb0` (Kevin's `segvigen-lightning`), being cut down to the code those two runs use.

The refactor is in progress: the plan and each step's record are in `REFACTOR_TODO.md`, the checks in
`script/internal/gate/`. Step 15 writes this repo's doc of record. The `lightgen` branch's TRELLIS.2-Emission
guide that this file held before is in git history (tag `pre-refactor`); those lanes are not part of this repo.

Parent repo context: `../AGENTS.md`, `../docs/baselines/segvigen-live-training.md`.
