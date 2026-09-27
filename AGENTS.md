# Flying project intent and working agreement

Flying is research toward a **human-versus-fly-connectome-model pi memorization
game**. The opponent learns using a computational model based on fruit-fly brain
connectivity. The user already understands that an actual fly is not memorizing
pi; keep that scientific distinction accurate without repeatedly lecturing about it.

- Research should produce reliable, playable opponents and explain which parts
  of the model are trained. Current best baselines freeze recurrent connectivity
  and train a readout; constrained recurrent learning is a separate research line.
- Preserve negative results, matched controls, reproducible seeds and honest
  scoring. Do not fabricate opponent digits, use a pi lookup to generate its moves,
  choose a best seed after evaluation, or present teacher-forced accuracy as recall.
- A useful playable baseline can ship before whole-brain simulation or perfect
  long-prefix recall. Research and the game should advance together.
- Latest user priority (2026-09-27): finish the original research stages before
  further game screen work. Preserve the console prototype, defer graphical UI,
  and use `docs/phase-status.md` to distinguish completed and unfinished phases.
- Default human study time is ten minutes, as proposed by the user. Label the
  current opponent pretrained. Later live-training displays must reflect actual
  updates/completion; do not present a waiting timer as model learning.
- The user explicitly authorized committing and pushing completed, verified work
  directly to `main` (2026-09-27). Do not open routine draft PRs or ask again to merge
  this project's normal work. Follow branch protection if the server requires it.
- Read `docs/game-direction.md`, `docs/next-work.md` and the latest results before
  selecting the next task. Keep protocol changes separate from observed outcomes.
