# M1–M11 historical reproduction index

Date: 2026-10-13; evidence cutoff `c48e0c43836e640ec1a15b1ac1152f7e3ec4e2a4`.

The synthesis audits **static replay entry points, recorded revisions, environment/parent references and archived checksums**. It does not execute these commands, install packages, fetch data or freshly replay old neural outcomes. Static consistency is not a guarantee of current-machine or cross-platform bitwise replay.

Use separate disposable clones/checkouts at the producer and verifier pins below; keep the full repository history and required parent artifacts. Leave the primary research checkout and all sealed result directories intact. A producer pin can predate its final result commit, so supply the recorded input/parent archives rather than assuming its own completed result is present at that older pin.

Use each recorded Python/package environment. These studies document Python3.12 and `requirements-act1-lock.txt`, but the archived environment record is authoritative. Set `PYTHONPATH=src;scripts` and numerical threads to1. Restore only hash-matched source/cache inputs at their configured paths; the JSON external-input inventory reports local availability at audit time. Whole-brain runners/verifiers can need ignored `outputs/tdc_v2/whole_cache` and pinned raw inputs; M10 and partial-only M11 differ. No data download was performed by this synthesis.

Use fresh destinations. If producer and verifier revisions differ, keep the newly produced result available in both disposable checkouts (copying new replay output is acceptable; do not relocate/overwrite old evidence). M1 uses the memory-repaired verifier pin; P1 also has a later verifier pin. Never update an old checksum or lower a validation tolerance to accommodate a changed checkout.

| Study | Producer source | Successful verifier source | Parent/protocol lineage |
| --- | --- | --- | --- |
| P1 | `03c12e9e383738161f9bb468e45143c5b3670d83` | `4edfee637a90fcca3290990ba60220afab4ff086` | v1/source-cache anchor |
| P2 | `8cb3aef4fa75910e8e253fcfd61dfa83c672d9eb` | `8cb3aef4fa75910e8e253fcfd61dfa83c672d9eb` | P1 |
| M1 | `4c2619df99a87c4072ecc8bd7481c67932963ab2` | `2c3d096fd9cf00087e8e02c0b4a46da6662cd755` | P1 |
| M2 | `7a70b1d5301115e60831721c21c0bd78abf81e6b` | `7a70b1d5301115e60831721c21c0bd78abf81e6b` | M1 |
| M3 | `a3ff879b00b69f5538210e4cc05e2a07cba54e3e` | `a3ff879b00b69f5538210e4cc05e2a07cba54e3e` | M1, M2 |
| M4 | `c9e96e936280c5e622ac45432bd2557150d9f8ff` | `c9e96e936280c5e622ac45432bd2557150d9f8ff` | M3 |
| M5 | `c541993484f6778a031d71ad4eb1798c748f6ce7` | `c541993484f6778a031d71ad4eb1798c748f6ce7` | M4 |
| M6 | `341aefa6353a81484001b86b72141d48ccfbf348` | `341aefa6353a81484001b86b72141d48ccfbf348` | M5 |
| M7 | `cfb05b0431a432271a8a93fb84e99e9df6aca168` | `cfb05b0431a432271a8a93fb84e99e9df6aca168` | M5, M6 |
| M8 | `d9dd96dcb8be10bd2c31d7a3213edc08ba6b1800` | `d9dd96dcb8be10bd2c31d7a3213edc08ba6b1800` | P2, M3, M4 |
| M9 | `e7100ea3066cca4126114a9d963fc97c260771e3` | `e7100ea3066cca4126114a9d963fc97c260771e3` | M8 |
| M10 | `7805fda8652dee3a54bc372b35c891e1613afbe3` | `11f52f314719346291603d493f126fe959e7de5a` | M9 |
| M11 | `87b00311bc30d5044d6ea65dc58a1ac5a3fb1b81` | `e106de6a9d1b0af4f09cbe2675b856fd0c5c9a90` | M8, M10 |

## Statically checked full-run entry points

Commands below are templates, **not executed by this synthesis**. Complete each study’s recorded safeguards/smoke first and obey its unchanged resource/stopping rules. The result reports supply the detailed stage order; these two command lines alone are not a complete protocol. For P2, the committed retry bank is reused explicitly to avoid its old failed default bank paths.

### P1

[Study’s detailed reproduction/evidence](../docs/temporal-memory-curve-results.md) · [Verifier pin record](../results/tdc_v2/p1_main_validation/source-validation.json)

```powershell
python scripts/temporal_memory_curve.py --out outputs/synthesis_replay/P1/main
python scripts/verify_temporal_memory_curve.py outputs/synthesis_replay/P1/main --out outputs/synthesis_replay/P1/validation
```

Producer CLI arguments at its pin: `--out, --smoke`.
Verifier CLI arguments at its pin: `--out, result`.

Archived manifest to compare: `d9fcc47fd5204d6432760f15fce9b89597bcfda8a643a9547e1e8606af53b955`. A fresh replay has new timestamps/resources/manifests; compare the registered scientific arrays/counts and source/parent validation rules, not an expectation that every new metadata file hashes identically.

### P2

[Study’s detailed reproduction/evidence](../docs/temporal-null-ensemble-results.md) · [Verifier pin record](../results/tdc_v2/p2_main_validation/checks.json)

```powershell
python scripts/temporal_null_ensemble.py run --graphs results/tdc_v2/p2_graphs_retry --graph-validation results/tdc_v2/p2_graph_validation_retry --out outputs/synthesis_replay/P2/main
python scripts/verify_temporal_null_ensemble.py run outputs/synthesis_replay/P2/main --out outputs/synthesis_replay/P2/validation
```

Producer CLI arguments at its pin: `--graph-validation, --graphs, --out, --smoke, command`.
Verifier CLI arguments at its pin: `--out, command, target`.

Archived manifest to compare: `8b6f5ecd57dc621a5bdd7fcda98d6bfd6ec14c57c7329a9db0d6ca30cb8f180b`. A fresh replay has new timestamps/resources/manifests; compare the registered scientific arrays/counts and source/parent validation rules, not an expectation that every new metadata file hashes identically.

### M1

[Study’s detailed reproduction/evidence](../docs/temporal-mechanism-results.md) · [Verifier pin record](../results/tdc_mechanism_v1/main_validation_retry1/checks.json)

```powershell
python scripts/temporal_mechanism.py --out outputs/synthesis_replay/M1/main
python scripts/verify_temporal_mechanism.py outputs/synthesis_replay/M1/main --out outputs/synthesis_replay/M1/validation
```

Producer CLI arguments at its pin: `--out, --smoke`.
Verifier CLI arguments at its pin: `--out, result`.

Archived manifest to compare: `96b4f5afe156daa9b025c980c60586ec6fa637df8aaa6e5443c45a1a50e89bfc`. A fresh replay has new timestamps/resources/manifests; compare the registered scientific arrays/counts and source/parent validation rules, not an expectation that every new metadata file hashes identically.

### M2

[Study’s detailed reproduction/evidence](../docs/temporal-tradeoff-results.md) · [Verifier pin record](../results/tdc_tradeoff_v1/main_validation/checks.json)

```powershell
python scripts/temporal_tradeoff.py --out outputs/synthesis_replay/M2/main
python scripts/verify_temporal_tradeoff.py outputs/synthesis_replay/M2/main --out outputs/synthesis_replay/M2/validation
```

Producer CLI arguments at its pin: `--out, --smoke`.
Verifier CLI arguments at its pin: `--out, result`.

Archived manifest to compare: `d0c112119f47ebc80727d40e864e3da402321af44a6493c32e2febe4aec90321`. A fresh replay has new timestamps/resources/manifests; compare the registered scientific arrays/counts and source/parent validation rules, not an expectation that every new metadata file hashes identically.

### M3

[Study’s detailed reproduction/evidence](../docs/temporal-cycles-results.md) · [Verifier pin record](../results/tdc_cycles_v1/main_validation/checks.json)

```powershell
python scripts/temporal_cycles.py --out outputs/synthesis_replay/M3/main
python scripts/verify_temporal_cycles.py outputs/synthesis_replay/M3/main --out outputs/synthesis_replay/M3/validation
```

Producer CLI arguments at its pin: `--out, --smoke`.
Verifier CLI arguments at its pin: `--out, result`.

Archived manifest to compare: `8d5de4917d2fc1fc3aa65e78ca4283ff5d64b6729b9db2cf346487f3739a9939`. A fresh replay has new timestamps/resources/manifests; compare the registered scientific arrays/counts and source/parent validation rules, not an expectation that every new metadata file hashes identically.

### M4

[Study’s detailed reproduction/evidence](../docs/cycle-attribution-feasibility-results.md) · [Verifier pin record](../results/cycle_attribution_v1/main_validation/checks.json)

```powershell
python scripts/cycle_attribution_feasibility.py --out outputs/synthesis_replay/M4/main
python scripts/verify_cycle_attribution_feasibility.py --result outputs/synthesis_replay/M4/main --out outputs/synthesis_replay/M4/validation
```

Producer CLI arguments at its pin: `--out, --smoke`.
Verifier CLI arguments at its pin: `--out, --result`.

Archived manifest to compare: `2e24e3f10867dc5945100908ec77c50fc8e7dce043144251fad468b114b49898`. A fresh replay has new timestamps/resources/manifests; compare the registered scientific arrays/counts and source/parent validation rules, not an expectation that every new metadata file hashes identically.

### M5

[Study’s detailed reproduction/evidence](../docs/temporal-functional-sensitivity-results.md) · [Verifier pin record](../results/functional_sensitivity_v1/main_validation/source.json)

```powershell
python scripts/temporal_functional_sensitivity.py --out outputs/synthesis_replay/M5/main
python scripts/verify_temporal_functional_sensitivity.py --result outputs/synthesis_replay/M5/main --out outputs/synthesis_replay/M5/validation
```

Producer CLI arguments at its pin: `--out, --smoke`.
Verifier CLI arguments at its pin: `--out, --result`.

Archived manifest to compare: `98eb3b4c3c49934365caa30608fd50b677dd883b7e3ed722271db4aceeaf73af`. A fresh replay has new timestamps/resources/manifests; compare the registered scientific arrays/counts and source/parent validation rules, not an expectation that every new metadata file hashes identically.

### M6

[Study’s detailed reproduction/evidence](../docs/counterfactual-tracking-results.md) · [Verifier pin record](../results/counterfactual_tracking_v1/main_validation/source.json)

```powershell
python scripts/counterfactual_tracking.py --out outputs/synthesis_replay/M6/main
python scripts/verify_counterfactual_tracking.py --result outputs/synthesis_replay/M6/main --out outputs/synthesis_replay/M6/validation
```

Producer CLI arguments at its pin: `--out, --smoke`.
Verifier CLI arguments at its pin: `--out, --result`.

Archived manifest to compare: `d7aa7b1ce4f951d0bfdeabe0eddad39fd71554d9c7d47049181410a989850b39`. A fresh replay has new timestamps/resources/manifests; compare the registered scientific arrays/counts and source/parent validation rules, not an expectation that every new metadata file hashes identically.

### M7

[Study’s detailed reproduction/evidence](../docs/counterfactual-noise-results.md) · [Verifier pin record](../results/counterfactual_noise_v1/main_validation/source.json)

```powershell
python scripts/counterfactual_noise.py --out outputs/synthesis_replay/M7/main
python scripts/verify_counterfactual_noise.py --result outputs/synthesis_replay/M7/main --out outputs/synthesis_replay/M7/validation
```

Producer CLI arguments at its pin: `--out, --smoke`.
Verifier CLI arguments at its pin: `--out, --result`.

Archived manifest to compare: `0a374377bda7e6743dfe0f303235df286bcbe38d4e5d7ca4b33bfa4f6eafc67f`. A fresh replay has new timestamps/resources/manifests; compare the registered scientific arrays/counts and source/parent validation rules, not an expectation that every new metadata file hashes identically.

### M8

[Study’s detailed reproduction/evidence](../docs/structural-specificity-design-results.md) · [Verifier pin record](../results/structural_specificity_design_v1/main_validation/source.json)

```powershell
python scripts/structural_specificity_design.py --out outputs/synthesis_replay/M8/main
python scripts/verify_structural_specificity_design.py --result outputs/synthesis_replay/M8/main --out outputs/synthesis_replay/M8/validation
```

Producer CLI arguments at its pin: `--out, --smoke`.
Verifier CLI arguments at its pin: `--out, --result`.

Archived manifest to compare: `528d420d16b55ccc99d8c180fdd07886cab1235d14b6ce1043bb381f2a0780eb`. A fresh replay has new timestamps/resources/manifests; compare the registered scientific arrays/counts and source/parent validation rules, not an expectation that every new metadata file hashes identically.

### M9

[Study’s detailed reproduction/evidence](../docs/structural-tracking-results.md) · [Verifier pin record](../results/structural_tracking_v1/main_validation/source.json)

```powershell
python scripts/structural_tracking.py --out outputs/synthesis_replay/M9/main
python scripts/verify_structural_tracking.py --result outputs/synthesis_replay/M9/main --out outputs/synthesis_replay/M9/validation
```

Producer CLI arguments at its pin: `--out, --smoke`.
Verifier CLI arguments at its pin: `--out, --result`.

Archived manifest to compare: `5f720f4b4e456810ac7b866e37f9fcb8b59f518d29aec962fd322132f3d2777f`. A fresh replay has new timestamps/resources/manifests; compare the registered scientific arrays/counts and source/parent validation rules, not an expectation that every new metadata file hashes identically.

### M10

[Study’s detailed reproduction/evidence](../docs/rule-transfer-design-results.md) · [Verifier pin record](../results/rule_transfer_design_v1/independent_validation/source.json)

```powershell
python scripts/rule_transfer_design.py --out outputs/synthesis_replay/M10/main
python scripts/verify_rule_transfer_design.py --result outputs/synthesis_replay/M10/main --out outputs/synthesis_replay/M10/validation
```

Producer CLI arguments at its pin: `--out`.
Verifier CLI arguments at its pin: `--out, --result`.

Archived manifest to compare: `40b857dc78b20a8e6965903e309cbbadf16bdea19b4df4b7343c5b17d897772d`. A fresh replay has new timestamps/resources/manifests; compare the registered scientific arrays/counts and source/parent validation rules, not an expectation that every new metadata file hashes identically.

### M11

[Study’s detailed reproduction/evidence](../docs/rule-transfer-results.md) · [Verifier pin record](../results/rule_transfer_v1/main_validation/source.json)

```powershell
python scripts/rule_transfer.py --out outputs/synthesis_replay/M11/main
python scripts/verify_rule_transfer.py --result outputs/synthesis_replay/M11/main --out outputs/synthesis_replay/M11/validation
```

Producer CLI arguments at its pin: `--out, --smoke`.
Verifier CLI arguments at its pin: `--out, --result, --smoke`.

Archived manifest to compare: `7d15d388d1ebe5b667e1f0304bd3f3e12afe214283311c51676e372c6905edf1`. A fresh replay has new timestamps/resources/manifests; compare the registered scientific arrays/counts and source/parent validation rules, not an expectation that every new metadata file hashes identically.

## Preserved technical records and availability

The new inventory lists incomplete manifests and complete-but-scientifically-invalid snapshots separately. Active successful validations never use those failed stages. In particular P2 uses the retry graph bank, M1 uses `main_validation_retry1`, M6/M7 use their final-audit retries, and M4 retains its captured source mismatch. M8 retains the interrupted root-ID smoke and successful retry. None is removed or relabeled by this index.

The baseline Git inventory protects sparse-excluded paths. Fresh byte/hash checks concern materialized files, and external-input availability is stated separately. Existing recorded independent replay counts remain historical evidence; they are not new replay counts for this synthesis.

[General historical replay discipline](../docs/historical-reproduction.md) · [Static inventory checks](../results/research_synthesis_v1/inventory/checks.json) · [External inputs and local availability](../results/research_synthesis_v1/inventory/external-inputs.json) · [Retained technical records](../results/research_synthesis_v1/inventory/retained-technical-records.json)
