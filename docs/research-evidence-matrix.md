# DrosoMem M1–M11 claim, evidence and limitation matrix

Date: 2026-10-13. Evidence cutoff `c48e0c43836e640ec1a15b1ac1152f7e3ec4e2a4`. This is an archived-evidence synthesis with **zero new model runs/fits**.

P1/P2 are anchors for M1–M11; the earlier [v1 closeout](../docs/drosomem-v1-closeout.md) remains sealed. P3 physiology/spiking/plasticity is outside this programme.

**Outcome legend:** PASS = the named registered gate; FAIL = a valid completed negative hypothesis test; assay-invalid = an undefined conditional endpoint; INFEASIBLE = a strict design obstruction; READY = tools/task preparation. Verification PASS authenticates the recorded result, including FAIL. These heterogeneous rows must not be combined into a success percentage.

| Study / question | Learned budget in that study | Final registered status | Supported observation | Interpretation limit |
| --- | --- | --- | --- | --- |
| P1: Can a fitted readout decode held-out iid past symbols at lags1–5? | External ridge heads:490 scalars/lag;252 lag heads; train-only scaler. Recurrent weights/encoder fixed. | **PASS** | All30 required partial lag/block cells pass >=10pp baseline excess. | Operational decoding, not sequence prediction, autonomous recall, physiological time or formal capacity. |
| P2: Does real wiring exceed every conditional null and a fixed material gap? | Same490-scalar/lag external ridge budget;3339 lag heads incl repeated real fits. Recurrent weights fixed. | **FAIL** | Both material gates fail; partial raw gap−2.285pp; whole+0.985pp below2.7pp. | Conditional finite bank, graph-dependent normalization; secondary positive direction is not registered superiority. |
| M1: Does synaptic history add decoding while every arm preserves current access? | 490 scalars/lag external ridge;756 heads; fixed recurrent/encoder coefficients. | **assay-invalid** | Execution/replay complete; carry-only current decoding fails90% eligibility, leaving primary null. | Positive descriptive contrasts cannot adjudicate the ineligible endpoint; no feedback-cycle-only attribution. |
| M2: Does carry-only improve short-history access while impairing current decoding? | 490 scalars/lag external ridge;756 heads; no recurrent learning. | **PASS** | Fresh-stream conjunction passes: partial history+6.562pp/current−15.025pp. | A particular intervention/readout, not a conserved memory budget or universal cost; M1 stays undefined. |
| M3: Can an acyclic zero-carry source subset retain lag2 iid access? | 490 scalars/lag external ridge;567 heads; masked normalized weights fixed. | **PASS** | DAG lag2 partial11998/12000 (99.983%); all registered cells pass; finite dependency certificates verify. | Mask also changes degree/weight mass/paths; does not isolate cycle benefit or establish cycle irrelevance. |
| M4: Can a same-node DAG preserve exact degree/roles/current paths and all-lag walk support? | 0 decoder fits;0 neural runs; frozen graph/support/count proofs. | **INFEASIBLE** | Partial degree/role obstruction and90 all-lag cycle witnesses; strict matching infeasible in both levels. | Whole degree-only question remains UNRESOLVED; not a functional memory comparison or rejection of approximate controls. |
| M5: Does the local response to a selected past-symbol contrast attenuate by lag5? | 490 scalars/lag external diagnostic ridge;189 heads; derivative and valid-replacement probes do not train recurrence. | **PASS** | Every partial R5<=.10 cell passes; mean local relative gain0.661% while lag2 decoding100%. | Amplitude attenuation is not information loss, storage localization or unique feedback effect; selected directions/anchors only. |
| M6: Does the old lag2 head follow a replaced past symbol with prefix/future fixed? | 0 new fits/parameters; reuse nine M5 heads of490 scalars and their old scalers. | **PASS** | Unconditioned joint tracking384/384 partial and192/192 paired whole; unchanged controls joint0. | One lag/substitution rule; unchanged-control joint0 is mathematical, not10% joint chance; no robustness or biological inference. |
| M7: Does frozen lag2 joint tracking meet90% at one registered noise dose? | 0 new fits; exact old M5 head/scaler bytes; no noise refit or feedback. | **FAIL** | Noisy joint255/384 (66.406%) partial and109/192 (56.771%) whole; every block fails. | Synthetic sigma.1 standardized observation noise at endpoints, not a dose curve, physiology or autonomous noisy recall. |
| M8: Can distinct controls preserve both raw weight multisets, degrees/roles/current coefficients? | 0 decoder fits; analytic source-derived prototypes and rank certificates; no source neural trajectories. | **READY** | 10 partial/3 whole graphs independently regenerate; delayed locations change22.34–23.76%/85.94–85.99%. | Finite conditional bank with shared paths; engineered delay has source front end; construction alone is not performance. |
| M9: After matched fits, does real delayed wiring exceed every matched null by3pp? | 174 lag heads of490 scalars;85260 affine scalars+8352 scaler moments; fixed recurrence/encoder/templates. | **FAIL** | All87 real/null/delay cases track100%; valid current access; real−null0pp fails strict order/3pp. | Saturated finite task does not establish wiring equivalence or universal absence of effects; whole bank is small. |
| M10: Can finite splits separate seen-rule starts from held-out coefficients and trivial lookup? | 0 neural fits; frequency/suffix count tables learned from2200 transitions/family; supplied-form oracle fits0. | **READY** | 990 structured episodes;2970 across families/63 evaluations independently reconstruct; formula oracle440/440 completes. | Oracle is privileged; seen-rule cycles overlap; period2 shortcut remains; task READY is not neural rule learning. |
| M11: Does the fixed source/affine learner acquire rules and materially outperform shift/nulls on held-out a7–9? | 216 heads of539 affine scalars;116424 scalars+20736 moments;2200 rows/head; frozen recurrence/encoder. | **FAIL** | All216 fits/independent SVD refits and427680 trajectory digests match; first13.23/13.94%, completion0.51/1.01%. | Train teacher19.35/18.86% fails50% acquisition; cannot isolate transfer failure after successful learning; shared tasks/shift repeats. |

## Canonical evidence and provenance

All links below address the immutable study records. The synthesis verifies saved decisions and checksum/source links; it does not freshly execute the archived model validations. Detailed source versions, cache/parent dependencies and static CLI checks are in the [reproduction index](../docs/research-reproduction-index.md).

### P1 — Independent historical-input access

Endpoint kind: **historical access**. Final status: **PASS**.

[Protocol](../docs/temporal-memory-curve-protocol.md) · [Config](../configs/temporal_memory_curve.json) · [Result report](../docs/temporal-memory-curve-results.md) · [Raw archive manifest](../results/tdc_v2/p1_main/manifest.json) · [Independent checks](../results/tdc_v2/p1_main_validation/checks.json) · [Canonical final audit](../results/tdc_v2/final_audit/audit.json)

Protocol `981441f5d3479271e6784e3f31c880d302f6f788`; producer `03c12e9e383738161f9bb468e45143c5b3670d83`; verifier `4edfee637a90fcca3290990ba60220afab4ff086`.

Main manifest SHA256: `d9fcc47fd5204d6432760f15fce9b89597bcfda8a643a9547e1e8606af53b955`.

### P2 — Fine-wiring TDC advantage

Endpoint kind: **wiring advantage**. Final status: **FAIL**.

[Protocol](../docs/temporal-null-ensemble-protocol.md) · [Config](../configs/temporal_null_ensemble.json) · [Result report](../docs/temporal-null-ensemble-results.md) · [Raw archive manifest](../results/tdc_v2/p2_main/manifest.json) · [Independent checks](../results/tdc_v2/p2_main_validation/checks.json) · [Canonical final audit](../results/tdc_v2/final_audit/audit.json)

Protocol `68eb46985466d946c7f76f32c45bc6bac85853a6`; producer `8cb3aef4fa75910e8e253fcfd61dfa83c672d9eb`; verifier `8cb3aef4fa75910e8e253fcfd61dfa83c672d9eb`.

Main manifest SHA256: `8b6f5ecd57dc621a5bdd7fcda98d6bfd6ec14c57c7329a9db0d6ca30cb8f180b`.

### M1 — Carry versus synaptic-history contribution

Endpoint kind: **undefined mechanism endpoint**. Final status: **assay-invalid**.

[Protocol](../docs/temporal-mechanism-protocol.md) · [Config](../configs/temporal_mechanism.json) · [Result report](../docs/temporal-mechanism-results.md) · [Raw archive manifest](../results/tdc_mechanism_v1/main/manifest.json) · [Independent checks](../results/tdc_mechanism_v1/main_validation_retry1/checks.json) · [Canonical final audit](../results/tdc_mechanism_v1/final_audit/audit.json)

Protocol `6c6c2966788a658c1d8cab5f640e67bd14eaf8e6`; producer `4c2619df99a87c4072ecc8bd7481c67932963ab2`; verifier `2c3d096fd9cf00087e8e02c0b4a46da6662cd755`.

Main manifest SHA256: `96b4f5afe156daa9b025c980c60586ec6fa637df8aaa6e5443c45a1a50e89bfc`.

The original primary flag remains JSON `null`; the positive descriptive contrasts are not a PASS or a valid mechanism FAIL. Use `main_validation_retry1`, retaining the original memory/test failures.

### M2 — Specific current/history tradeoff

Endpoint kind: **specific decoder tradeoff**. Final status: **PASS**.

[Protocol](../docs/temporal-tradeoff-protocol.md) · [Config](../configs/temporal_tradeoff.json) · [Result report](../docs/temporal-tradeoff-results.md) · [Raw archive manifest](../results/tdc_tradeoff_v1/main/manifest.json) · [Independent checks](../results/tdc_tradeoff_v1/main_validation/checks.json) · [Canonical final audit](../results/tdc_tradeoff_v1/final_audit/audit.json)

Protocol `7552c234cc5822bdafce1238a9050ea7b38fafaa`; producer `7a70b1d5301115e60831721c21c0bd78abf81e6b`; verifier `7a70b1d5301115e60831721c21c0bd78abf81e6b`.

Main manifest SHA256: `d0c112119f47ebc80727d40e864e3da402321af44a6493c32e2febe4aec90321`.

### M3 — Acyclic delayed-path sufficiency

Endpoint kind: **feedforward sufficiency**. Final status: **PASS**.

[Protocol](../docs/temporal-cycles-protocol.md) · [Config](../configs/temporal_cycles.json) · [Result report](../docs/temporal-cycles-results.md) · [Raw archive manifest](../results/tdc_cycles_v1/main/manifest.json) · [Independent checks](../results/tdc_cycles_v1/main_validation/checks.json) · [Canonical final audit](../results/tdc_cycles_v1/final_audit/audit.json)

Protocol `438cae437e110dbeb9cb2a02d0a58cfe61e6d233`; producer `a3ff879b00b69f5538210e4cc05e2a07cba54e3e`; verifier `a3ff879b00b69f5538210e4cc05e2a07cba54e3e`.

Main manifest SHA256: `8d5de4917d2fc1fc3aa65e78ca4283ff5d64b6729b9db2cf346487f3739a9939`.

### M4 — Strict cycle-control feasibility

Endpoint kind: **structural design feasibility**. Final status: **INFEASIBLE**.

[Protocol](../docs/cycle-attribution-feasibility-protocol.md) · [Config](../configs/cycle_attribution_feasibility.json) · [Result report](../docs/cycle-attribution-feasibility-results.md) · [Raw archive manifest](../results/cycle_attribution_v1/main/manifest.json) · [Independent checks](../results/cycle_attribution_v1/main_validation/checks.json) · [Canonical final audit](../results/cycle_attribution_v1/final_audit/audit.json)

Protocol `2c7a5f593c22909652a1374c9338571656411855`; producer `c9e96e936280c5e622ac45432bd2557150d9f8ff`; verifier `c9e96e936280c5e622ac45432bd2557150d9f8ff`.

Main manifest SHA256: `2e24e3f10867dc5945100908ec77c50fc8e7dce043144251fad468b114b49898`.

The whole degree-only subquestion remains UNRESOLVED. The archived source-validation failure snapshot is complete as an archive but scientifically invalid; it is retained separately from successful M4 checks.

### M5 — Functional contrast attenuation

Endpoint kind: **local functional sensitivity**. Final status: **PASS**.

[Protocol](../docs/temporal-functional-sensitivity-protocol.md) · [Config](../configs/temporal_functional_sensitivity.json) · [Result report](../docs/temporal-functional-sensitivity-results.md) · [Raw archive manifest](../results/functional_sensitivity_v1/main/manifest.json) · [Independent checks](../results/functional_sensitivity_v1/main_validation/checks.json) · [Canonical final audit](../results/functional_sensitivity_v1/final_audit/audit.json)

Protocol `37212abeb6cfc1a4907e08348ff41030f3edfb8b`; producer `c541993484f6778a031d71ad4eb1798c748f6ce7`; verifier `c541993484f6778a031d71ad4eb1798c748f6ce7`.

Main manifest SHA256: `98eb3b4c3c49934365caa30608fd50b677dd883b7e3ed722271db4aceeaf73af`.

### M6 — Frozen-readout past-value intervention

Endpoint kind: **conditional historical tracking**. Final status: **PASS**.

[Protocol](../docs/counterfactual-tracking-protocol.md) · [Config](../configs/counterfactual_tracking.json) · [Result report](../docs/counterfactual-tracking-results.md) · [Raw archive manifest](../results/counterfactual_tracking_v1/main/manifest.json) · [Independent checks](../results/counterfactual_tracking_v1/main_validation/checks.json) · [Canonical final audit](../results/counterfactual_tracking_v1/final_audit_retry1/audit.json)

Protocol `c2263270503c1d106dce7069a496c30c924e04f6`; producer `341aefa6353a81484001b86b72141d48ccfbf348`; verifier `341aefa6353a81484001b86b72141d48ccfbf348`.

Main manifest SHA256: `d7aa7b1ce4f951d0bfdeabe0eddad39fd71554d9c7d47049181410a989850b39`.

### M7 — Frozen tracking with artificial observation noise

Endpoint kind: **one-dose robustness**. Final status: **FAIL**.

[Protocol](../docs/counterfactual-noise-protocol.md) · [Config](../configs/counterfactual_noise.json) · [Result report](../docs/counterfactual-noise-results.md) · [Raw archive manifest](../results/counterfactual_noise_v1/main/manifest.json) · [Independent checks](../results/counterfactual_noise_v1/main_validation/checks.json) · [Canonical final audit](../results/counterfactual_noise_v1/final_audit_retry1/audit.json)

Protocol `2d07a9d303bd700ab69fa29797f8d6a712624d08`; producer `cfb05b0431a432271a8a93fb84e99e9df6aca168`; verifier `cfb05b0431a432271a8a93fb84e99e9df6aca168`.

Main manifest SHA256: `0a374377bda7e6743dfe0f303235df286bcbe38d4e5d7ca4b33bfa4f6eafc67f`.

### M8 — Matched structural-bank construction

Endpoint kind: **control feasibility**. Final status: **READY**.

[Protocol](../docs/structural-specificity-design-protocol.md) · [Config](../configs/structural_specificity_design.json) · [Result report](../docs/structural-specificity-design-results.md) · [Raw archive manifest](../results/structural_specificity_design_v1/main/manifest.json) · [Independent checks](../results/structural_specificity_design_v1/main_validation/checks.json) · [Canonical final audit](../results/structural_specificity_design_v1/final_audit/audit.json)

Protocol `8f36a891f3b572aabc5e6865e8683cc5cbcdf1a4`; producer `d9dd96dcb8be10bd2c31d7a3213edc08ba6b1800`; verifier `d9dd96dcb8be10bd2c31d7a3213edc08ba6b1800`.

Main manifest SHA256: `528d420d16b55ccc99d8c180fdd07886cab1235d14b6ce1043bb381f2a0780eb`.

### M9 — Material fine-wiring tracking advantage

Endpoint kind: **wiring advantage at ceiling**. Final status: **FAIL**.

[Protocol](../docs/structural-tracking-protocol.md) · [Config](../configs/structural_tracking.json) · [Result report](../docs/structural-tracking-results.md) · [Raw archive manifest](../results/structural_tracking_v1/main/manifest.json) · [Independent checks](../results/structural_tracking_v1/main_validation/checks.json) · [Canonical final audit](../results/structural_tracking_v1/final_audit/audit.json)

Protocol `6bb55fb7c534fe4d29ab366bf9a04eec3ac4b81e`; producer `e7100ea3066cca4126114a9d963fc97c260771e3`; verifier `e7100ea3066cca4126114a9d963fc97c260771e3`.

Main manifest SHA256: `5f720f4b4e456810ac7b866e37f9fcb8b59f518d29aec962fd322132f3d2777f`.

### M10 — Held-out affine-rule task design

Endpoint kind: **task feasibility**. Final status: **READY**.

[Protocol](../docs/rule-transfer-design-protocol.md) · [Config](../configs/rule_transfer_design.json) · [Result report](../docs/rule-transfer-design-results.md) · [Raw archive manifest](../results/rule_transfer_design_v1/main/manifest.json) · [Independent checks](../results/rule_transfer_design_v1/independent_validation/checks.json) · [Canonical final audit](../results/rule_transfer_design_v1/final_audit/audit.json)

Protocol `61b6139028720b708463e442c149cd74f8d2bd2b`; producer `7805fda8652dee3a54bc372b35c891e1613afbe3`; verifier `11f52f314719346291603d493f126fe959e7de5a`.

Main manifest SHA256: `40b857dc78b20a8e6965903e309cbbadf16bdea19b4df4b7343c5b17d897772d`.

The producer summary intentionally remains `READY_PENDING_INDEPENDENT_VERIFICATION`. Final READY is supplied by the matching independent checks/final audit; the old producer file is not edited to change that label.

### M11 — Matched rule-continuation performance

Endpoint kind: **acquisition and autonomous continuation**. Final status: **FAIL**.

[Protocol](../docs/rule-transfer-protocol.md) · [Config](../configs/rule_transfer.json) · [Result report](../docs/rule-transfer-results.md) · [Raw archive manifest](../results/rule_transfer_v1/main/manifest.json) · [Independent checks](../results/rule_transfer_v1/main_validation/checks.json) · [Canonical final audit](../results/rule_transfer_v1/final_audit/audit.json)

Protocol `68b49d37e79a9a00c95f00baaf44c73d61147c6f`; producer `87b00311bc30d5044d6ea65dc58a1ac5a3fb1b81`; verifier `e106de6a9d1b0af4f09cbe2675b856fd0c5c9a90`.

Main manifest SHA256: `7d15d388d1ebe5b667e1f0304bd3f3e12afe214283311c51676e372c6905edf1`.

## Cross-study claim boundaries

- Historical-input access: supported by P1 and bounded M3/M6 observations; it is neither next-symbol prediction nor autonomous string recall.
- Pure feedback-cycle attribution: unresolved. M3 is sufficiency under a broader mask; M4 blocks one exact ideal control.
- Fine source-wiring superiority: no registered material support in P2, M9 or M11; finite constraints and saturation limit wider inference.
- Robust frozen tracking: M6 clean success does not rescind M7 one-dose failure.
- Rule learning/transfer: M10 task READY and its privileged formula oracle do not establish neural learning; M11 acquisition itself misses its gate.
- Internal synaptic learning, physiology, biological storage sites, living-fly arithmetic and formal capacity remain unestablished in this programme.

See [research synthesis](../docs/research-synthesis.md) and [internal manuscript outline](../docs/manuscript-outline.md).
