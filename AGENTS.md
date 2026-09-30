# DrosoMem research scope and working agreement

DrosoMem is a **research-only project** studying sequence learning and autonomous
recall in computational models constrained by fruit-fly brain connectivity.
Pi is a controlled sequence-memory benchmark. Distinguish computational model
results from biological claims without repeatedly explaining this to the user.

- Latest scope decision (2026-09-27): keep this repository focused on research.
  Work on models, experiments, analysis, reproducibility and scientific reports.
- State precisely which parameters are learned. Current rate baselines freeze
  recurrent connectivity and train a readout; constrained recurrent learning is
  a separate research line.
- Preserve negative results, matched controls, reproducible seeds and honest
  scoring. Do not fabricate generated digits, generate by reference lookup,
  select the best seed after evaluation, or equate teacher-forced accuracy with
  autonomous recall.
- Whole-brain trajectory feasibility is established within the Phase 6 scope.
  ACT I also executed matched rate-model pi training/recall without establishing
  a whole-brain advantage. Broader physiological validation is not established.
- Plan the next study around one explicit question with a fixed budget and
  acceptance criteria before observing outcomes. Keep proposals distinct from
  executed experiments and successful hypotheses.
- Keep archived research outputs and their checksum manifests unchanged.
  Historical replay uses the recorded code/protocol revision; do not rewrite
  old hashes to accommodate current source or editorial changes.
- The user explicitly authorized committing and pushing completed, verified work
  directly to main. Do not open routine PRs or ask again to merge normal work.
  Follow branch protection if the server requires it.
- Read docs/research-direction.md, docs/next-work.md and the latest results
  before selecting the next task. Use docs/phase-status.md for completion scope.
