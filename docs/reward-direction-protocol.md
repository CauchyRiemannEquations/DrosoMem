# ACT IV — Local update/reward directional diagnostic

Locked before any new graph smoke/main outcome. Question: at INITIAL weights,
does the current local reward rule propose a direction that raises fixed-code
correctness on independent inputs more consistently than random directions?
This tests one finite directional response, NOT a full gradient, training success,
autonomous recall, biological learning or formal memory capacity.

## Baseline, rule, data and interface

Reproduce prior local-reward real_contingent_c701_s241142 (expected38.1%) by an
independent exact training/trajectory replay. Preserve all historical artifacts.
Use real partial circuits c701/c702 (686neurons,3309/3241edges,threshold5), same
48MBON observation and0-trained-parameter nearest-code decoder. Codebook consists
of four seeded disjoint12-slot groups with amplitude.25. Input fraction.1,
amplitude.5, incoming-L1 gain.9, leak.6, synchronous tanh, K4 iid train/test,
warmup100,train2000,test1000,target lag2. Symbols remain externally supplied;
there is no autonomous rollout and no Pi Memory Score in this diagnostic.

Use EXACTLY the preceding local rate-covariance rule: lr.05,floor1e-4,trace.8,
activity/reward EMA rates.05,initial reward baseline.25,training MBON-drive noise
SD.02. Ten independent-noise passes over the same training sequence. Reset state,
mean,trace and reward baseline each epoch, same warmup handling. At each reward
opportunity compute the existing rule's full one-step magnitude change including
floor and incoming-mass normalization, accumulate that proposed change, then
restore ORIGINAL weights before the next forward step. Thus input states always
follow the frozen initial graph; no proposed weight change affects later states.
Average20000 proposed updates. The rule receives no target vector, code error,
test sequence or classifier gradient. This is an empirical mean proposal, not an
analytic expectation; lr stays identical and is not swept.

## Directions and symmetric matched perturbations

Let m be positive initial magnitudes of existing KC→MBON edges, B_j their sum at
postsynaptic cell j. Remove numerical row-sum residuals from average proposal v:

    tangent(v)_ji = v_ji - (m_ji/B_j) * sum_i(v_ji)

Five random controls start with independent standard Gaussian edge draws TIMES
initial magnitude m, and undergo the same tangent projection. They are NOT an
isotropic global direction ensemble; this weighting avoids huge relative changes
to tiny edges. Normalize each local/random tangent to unit global L2 norm.
All signed weights outside the plastic mask remain exactly unchanged.

Nominal radius=.01*||m||_2. Choose ONE shared radius for the local and all five
random directions and both orientations, using only these directions/initial graph:

    radius = min(nominal, .5 * min_all_nonzero_entries((1-floor)*m/abs(unit)))
    m_plus/minus = m +/- radius*unit

This preserves edge signs, existing topology and each MBON's absolute plastic
incoming mass, while matching the TOTAL actual L2 perturbation across directions.
No post-perturbation projection/renormalization changes norms. Report actual radius
and any boundary reduction. Radius<nominal/10 is flagged resolution-limited:
failure in that case cannot establish lack of directional utility. If local mean
has exactly zero norm, record zero directions/radius and no improvement; do not
replace the rule/dataset. Control orientation is determined by RNG, never outcomes.
There are13 graphs per block: initial plus six directions times two orientations.

## Evaluation, controls and prespecified decisions

For each graph, freeze all weights and evaluate the held-out1000 symbols after
warmup. PRIMARY: mean correctness over32 independent noise streams, SD.02 as in
training. The same32 streams are reused across all13 graphs (common random numbers).
SeedSequence([evaluation_seed,replica_index]) generates each stream separately.
SECONDARY: one noise-free trajectory per graph. No readout fitting, test-driven
direction selection, calibration, task/lr/noise/radius sweep or extra epochs.
Replicates share the input dataset and are Monte Carlo noise samples, NOT32
independent biological/model-seed samples. Raw fixed scores are uncalibrated
negative squared distances, not probabilities.

Per circuit define local directional response = accuracy(local+)−accuracy(local−),
random-axis sensitivity = mean over5 of abs(accuracy(random+)−accuracy(random−)),
above-random = local response − random sensitivity, forward gain = local+ − initial.
Taking absolute control responses intentionally gives each RANDOM axis its favorable
sign before averaging; it is prespecified, not best-control or best-seed selection.
Average both circuits within each input/code seed, n=3 per cohort.

Each cohort's primary must satisfy ALL:
- mean local directional response>=.25pp, positive for every paired seed;
- mean above-random>=.10pp, positive for every seed;
- mean forward gain>=.10pp, positive for every seed.
Confirmed alignment requires discovery AND fresh confirmation to pass. The small
diagnostic thresholds reflect a1% perturbation question, not a replacement for the
previous study's5pp learning criterion. If any gate fails, report unconfirmed under
this probe, not universal uselessness of local learning. Clean results are secondary
and cannot rescue a failed noisy primary. Report every circuit/seed/replicate,
mean,median,sample variance,paired dz and10000-draw bootstrap95; no p-value decision.

## Seeds, finite budget and preservation

Smoke261001,c701,train200/test100,one proposal epoch,two noisy replicates.
Discovery261142–261144; confirmation271142–271144, two circuits each.
Train seed=input+1000,test+2000,code+3000,proposal noise+4000,random directions+5000,
evaluation noise+6000. Bootstrap278399. Prior numeric-token seed audit saved.
13 proposal blocks including smoke,169 probe graphs,5018 noisy +169 clean =5187
evaluation trajectories,338 aggregate graph/mode rows. Smoke excluded from main
statistics. Confirmation runs whether discovery passes or fails. One finite plan,
max7200s/3GiB RSS per runner/verification. No adaptive retries for bad performance.

Checkpoint stores mean proposal, unit directions, input/code/symbol identities,
all proposal actions/rewards/advantages/norms, all evaluation scores/replica accuracies,
clean48MBON features and all13 sparse probe graphs. Histories,actual radius,
config/Git/source hashes, graph IDs, timing/RSS/environment and old-result hashes
are saved. Stop and preserve failures on nonfinite values, constraints/replay/budget
failure. Independent verifier uses scalar trajectories and a separate local-update
implementation; require exact proposals,graphs,scores/actions and statistics.

Preflight engineering note: a synthetic batch-versus-scalar test initially exposed
~1e-17 distance-reduction rounding from noncontiguous feature layout. Making the
observed batch contiguous before the unchanged distance calculation restores exact
agreement. This repair preceded all new graph outcomes; no score rounding or relaxed
tolerance was introduced. Full tests and source/protocol commit precede execution.
