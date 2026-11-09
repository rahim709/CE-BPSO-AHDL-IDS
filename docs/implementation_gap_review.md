# Implementation review of the research gap

The later [false-alarm correction study](false_alarm_optimization.md) reduces mean normal false alarms from 42.70% to 1.90% using validation-selected square-root weighting. This is a conventional loss adjustment, with a severe NMRI-recall trade-off; it does not establish a search-allocation contribution. The 42.70% result discussed below is the original balanced-loss reference.

The research question is whether short-budget validation fitness rejects architectures that become useful with longer training, and whether resource allocation can improve selection under a limited budget. The implementation supports investigating that question. The existing four-preset experiment does not establish the proposed problem: its epoch-10 winner also wins at epoch 30 in all three seeds. A new allocation algorithm cannot be justified by the current evidence alone.

## Corrections in this review

1. **Promotion diagnostics now follow the search reduction factor.** Previously, ranking diagnostics and the sampling pool for rejected candidates always assumed one-third survival. With a reduction factor of two, this described a different selection rule from the implemented scheduler. Diagnostics now use the configured factor for promotion-boundary ties, elimination rates and rejected-candidate sampling, and record it in ranking results. Existing factor-three studies retain their rule; factor-two diagnostic outputs must be regenerated before interpretation.
2. **Incomplete selections are explicit.** A search can exhaust its cap after low-fidelity observations and still return its best observed candidate. Search summaries now record the required selection fidelity, whether it was reached, and the selection status. Reduced BPSO intentionally selects at its configured low budget; that protocol remains valid and is labelled accordingly. Reaching selection fidelity does not imply completion of every planned search generation.
3. **Automatic comparisons no longer test incomplete selections.** Previously, comparison execution trained any available candidate on the final protocol even if it had not reached the method's required selection fidelity. Such a result could appear beside completed selections. Comparisons now skip that final run and persist the reason in their status record. Partial architectures remain available for explicit diagnostic use.

Regression coverage checks the factor-two survivor count, incomplete halving versus intentional reduced-budget selection, and a real comparison command with the test array removed: an incomplete selection must finish without opening it. The complete suite and the seven-method artificial integration are rerun for this revision. Exact results are saved with the gap-audit evidence.

## Reviewed behavior and scientific limits

| Area | Implementation assessment | Remaining limitation |
|---|---|---|
| Data preprocessing | Transform parameters and class weights use training data | Random splits contain repeated non-time states; independent capture data remain unavailable |
| Candidate evaluation | Fitness uses validation; tests cover execution without test arrays | Existing test outcomes are development knowledge |
| BPSO best updates | Strict maximization; halving exposes final-rung scores to swarm updates | No novel update rule or demonstrated allocation advantage |
| Continuation | Model, optimizer and random states are restored; incremental epochs are charged | Hard kills between separate artifact writes can leave accounting gaps |
| Fidelity comparisons | Best selection uses the highest observed fidelity; incomplete selections are now labelled and excluded from automatic final testing | Equal epoch limits do not ensure equal computation or full search completion |
| Ranking evidence | Common candidates, repeated seeds, explicit tie ambiguity and selected-subset scope | Four presets do not represent the complete architecture space |
| Baselines | Seven established search controls share the evaluator | Full matched-resource comparisons remain unexecuted; native replay tests do not prove upstream DEHB resume equivalence |
| Final detector | Fixed reference trained and tested at three dedicated seeds | Normal false-positive rate remains 42.70%; this is a model-quality limitation, not fixed by scheduler corrections |

## What would establish resolution

First test a sufficiently diverse architecture cohort under an adequate training and preprocessing protocol, retaining every candidate at the common reference endpoint. Determine whether premature elimination persists across training seeds. If it does, compare a clearly specified allocation intervention against random search, halving, Hyperband, DEHB and the BPSO controls using measured resource consumption and a common final-training protocol. Report failures, incomplete selections and underused budgets. Use independent or capture-separated evaluation data for confirmatory detector claims.

If the larger study also finds reliable early selection, the supported contribution should remain the framework and empirical evaluation. The current code corrections improve experimental validity; they do not prove that the research problem is solved or that the detector is deployment-ready. No further model settings were chosen using the known test errors in this review.

## Executed verification

The revised CPU suite passed **77 tests in 6.85 seconds**, including three new regression tests. A fresh artificial integration completed all seven search controls, diagnostics, final fixture evaluation and report generation. Evidence: `docs/latex/evidence/gap_audit_tests_cpu.xml` and `gap_audit_integration.json`. The subsequent [full GPU verification](latest_gpu_verification.md) passed 77 tests on both CPU and CUDA, all 38 commands and 80 CUDA training trajectories, including these corrections. The neural training implementation and recorded 200-epoch detector measurements were unchanged.
