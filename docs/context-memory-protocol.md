# Fixed-context ambiguity and autonomous recall — locked protocol

## Scope and provenance

ACT II task/decoding diagnostic, using the already published alphabet study at
`a88c609a92089f98320ca6e12a23fd0b1b43d28f`. Its neural outcomes are known.
This preregistration fixes the NEW context analysis before computing its outcomes;
it is not a blinded replication of the known K curve. Commit this document and
config before execution. No neural fitting, graph change or hyperparameter search.

## Cohort and training

Use all 80 archived tasks in results/alphabet_main: K=2,4,10,16;
model seeds21142–21146 paired with data seeds22142–22146; strata701,702;
legacy5 and brain1. These contain only **20 distinct symbol sequences**.
Fit each context model once per distinct sequence: 20 x 6 =120 fits.
Neither graph nor input stratum is an independent n-gram replicate.
N128, prompt[0,1,0], 125 generated symbols. Training targets are positions1..127
(zero based), including two within-prompt targets, as in the neural study.
Target positions3..34 have weight4; all others weight1. No validation tuning.

## Model definition (order means number of preceding symbols)

Orders1,2,3,4,5,8, all reported. Primary diagnostic order=3, selected before
outcomes; order2 is the primary ambiguity measure. For maximum order m, count
weighted next-symbol occurrences at every suffix length0..min(m,t) at target
position t. No wraparound, padding, absolute position or unique start token.
Contexts shorter than m at sequence start remain shorter tuple keys. Empty
context is the weighted global distribution. Each training position contributes
once to each available suffix length. Store every count table.

Prediction uses the longest suffix present in the fitted tables, backing off
one symbol at a time to the empty context. No smoothing. Ties choose the smallest
integer symbol. Probabilities are normalized weighted counts for the selected
context. Rollout API accepts only fitted tables, prompt and requested length;
append each prediction to generated history, never targets. The initial history
is the three-symbol prompt even when m>3. Teacher-forced metrics separately use
true preceding symbols. This is training-instance recall, not unseen prediction.

## Diagnostics and metrics

For target positions3..127, group by the exact preceding min(m,t) symbols,
without backoff. Record distinct contexts, singleton contexts, conflicting
contexts (more than one distinct next symbol), fraction of occurrences in
conflicting contexts, and unweighted ambiguity error floor:
`(125 - sum_context max_symbol count(context,symbol))/125`.
This is the best deterministic same-context training classification error, not
entropy or memory capacity. Also save every context's unweighted symbol counts.

For every fit: exact-prefix symbols/bits, first error (1-based generated position,
null for complete), 125 integer predictions, correctness and conditional
probabilities, teacher accuracy over127 transitions and125 evaluation positions,
accuracy regions1–32/33–64/65–125, context length used at each step, and fallback
rate (relative to min(m,history length)). Report table contexts, nonzero
context-symbol cells, dense count scalars K x contexts, and serialized bytes.
These storage measures are not equivalent to neural trainable parameters.
For each of80 neural runs compare its saved prefix to each n-gram order; report
all480 differences. Average the two neural strata within each seed before
paired summaries. Do not pick a best order from these results.

## Hypotheses and decision rules

H1: order2 ambiguity error floor falls from K2 toK16 by at least0.10 on average,
with a strictly negative difference in at least4/5 paired data seeds.
H2: order3 autonomous prefix grows from K2 toK16 by at least25 symbols on average,
with a strictly positive difference in at least4/5 seeds.
Both passing is evidence consistent with reduced finite-context ambiguity as
one explanation of the known K curve. It cannot establish a causal mechanism
inside the neural model. Failure of either withholds that joint interpretation;
report the failed endpoint without changing order, threshold or training rules.
Neural-minus-ngram differences are descriptive, without a superiority gate.

Report every seed, mean, median, sample variance, paired mean/SD effect size
(null when SD=0), wins/ties/losses and percentile bootstrap95% intervals using
10,000 paired block draws, RNG26399. Only five seed blocks: intervals are
descriptive; no p-value-based claims or multiple-order cherry-picking.

## Budget, checks and stopping

Exactly120 table fits plus independent reference verification of all120; stop
after these and reporting. No new cohort triggered by a positive diagnostic:
it remains exploratory on known tasks. Total local budget15 minutes,1GiB RSS;
abort and preserve partial files if exceeded. No retries to select outcomes.
Verify all archived main artifact checksums and duplicate sequences. Verify
order1 reproduces the old weighted Markov control on all20 unique tasks.
Use synthetic tests for starts, ties, backoff, conflicts, and autonomous feedback.
Independently implement a scan-based reference that recounts matching training
positions on demand, rather than using the fitted-table code. Check all saved
predictions, probabilities, teacher outputs and ambiguity counts. Preserve old
results and numerical source bytes. Save protocol/config/code/input hashes,
git revision, environment, timestamps, runtime and measured peak process RSS.

## Limits and next-step rule

An n-gram table memorizes observed finite contexts; it is neither a random
connectome nor a matched-parameter neural control. Strong recall does not prove
biological memory; poor neural-relative recall does not prove unique topology.
This diagnostic does not separate graph scale and edge completeness. If both
hypotheses pass, the next single proposed experiment is a preregistered matched
short-context-conflict sequence intervention, keeping K and N fixed. Otherwise
prioritize one fixed-K input-code diagnostic. Neither follow-up is executed here.
