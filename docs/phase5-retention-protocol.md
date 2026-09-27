# Locked first-prefix-share retention study — 2026-09-27

Editorial update: research terminology only; the registered numerical design is
unchanged. Original protocol bytes are preserved in Git history; see
[historical reproduction](historical-reproduction.md).

Base: 41210771509ad4d81994a2d9082b1bf58b269ebc. Written before outcomes.
This is a scoped discovery study for the default `314` sequence prefix, offset 0.
Other offsets require a separate confirmation study; do not claim them tested.

## One intervention and matched controls

Use fixed 32-target weighting, the previous 32→64→128 curriculum, and an anchored
curriculum. Every arm gets 6,000 Adam updates with boundaries at 2,000/4,000/6,000,
learning rate .03, L2=1e-5, 482 head parameters and one continuous optimizer state.
Only the final model is primary. All statistics, frozen states and initialization
are paired, and each head begins from scratch. Do not select the best checkpoint.

The ordinary curriculum gives weight 4 to the first W generated targets and 1
elsewhere. For the anchored arm keep the ordinary weights outside the first 32,
but set each of those 32 weights to `4*(71+3W)/167` for W=32/64/128. More generally,
anchor weight = 4 * (non-anchor weight sum) / 167. Normalize by total weight as
before. This preserves first-32 normalized loss share `128/295` (43.39%) at all
stages. Anchor weights are 4, about 6.2994 and 10.8982. Initial two prompt-related
targets remain weight 1; all other targets are retained. This does not guarantee
preservation of learned predictions. Do not add replay buffers, label lookup,
teacher-state matching losses, gradient projection or outcome-dependent gates.

## Fixed sample and research decision

Use existing s701/s702 subsets, real/role-degree-shuffled graphs, new seeds
3142/3143/3144, all initializations 0/1/2, incoming-L1, mbon_after_kc, 200 pi digits,
prompt 3, horizon 197 and only offset 0. Twelve state conditions × three arms ×
three initializations = 108 fits and 324 saved stage-head evaluations.

Primary success requires BOTH:
1. On real graphs, anchored final mean recall exceeds both controls and at least
   4/6 initialization-averaged conditions beat each control.
2. Anchored loses no more initially complete 32-digit models than fixed weighting
   and has at least as many final 32-digit completions as fixed weighting.

The stronger joint criterion additionally requires final later-165 accuracy at
least that of fixed weighting. Report each criterion separately, including failure.
Report all arms, seeds, initializations, 32/64/128/197 completion counts, early
retention at both boundaries, fixed position-band accuracies, full accuracy and
unweighted cross entropy. Shuffled controls must be included. Criteria are
descriptive, not statistical significance tests; overlapping subsets and model
initializations are not independent animals.

A pass justifies a fresh cross-segment confirmation, not automatic replacement of
the existing fixed-32 research baseline. No hyperparameter retuning in this run.

## Verification

Use the existing atomic condition checkpoint and strict resume engine. Rebuild
all 12 state conditions and replay all 324 heads and full generated strings.
Independently refit all nine heads in the first real subset/seed, matching all 27
stage heads and training histories. Check all 200 pi digits against Decimal.
Test exact initial-stage equivalence, normalized loss-share invariance, weight
boundaries, matched optimizer budget, conservative criteria, stop/resume/replay,
changed context and corrupted checkpoint rejection. Full pytest must pass.
