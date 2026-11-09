# CE-BPSO-AHDL-IDS: Compute-Efficient Binary Particle Swarm Optimization-Based Architecture Search for Hybrid Deep Learning Intrusion Detection in the Internet of Things

**Authors:** Abdur Rahim, Arpit Kumar Singh, Gosala Buela

**Affiliation:** Indian Institute of Information Technology, Design and Manufacturing Kancheepuram, Chennai, India

**Experimental revision:** 8 November 2026

## Abstract

Automated intrusion-detection model design requires training budgets that distinguish useful architectures without excessive search cost. This study implements a binary particle swarm optimization framework for hybrid convolutional and recurrent networks and examines the reliability of its experimental evaluation. The framework provides training-fitted preprocessing, a 73-bit architecture representation, resumable candidate evaluation, seven search controls and validation-only fitness. Experiments use 97,019 Gas Pipeline records with 26 predictors and eight classes. A four-preset cohort is evaluated at 3, 10 and 30 epochs across three training seeds, alongside normalization, neural-component and predictor-state grouping controls. At 30 epochs, reference validation macro-F1 is 0.4267 with column L2 and 0.7831 with Min–Max scaling. Three separately initialized reference models achieve test macro-F1 of 0.8454 ± 0.0012 at a fixed 200-epoch endpoint. The existing test split had been examined previously, so this is not an independent blinded evaluation. A subsequent validation-selected loss correction reduces normal false alarms from 42.70% to 1.90%, with 95.44% accuracy and 0.8551 macro-F1; NMRI recall falls to 0.54%, exposing a substantial sensitivity trade-off. A further validation-selected NMRI specialist raises NMRI recall to 16.82% at 2.60% false alarms, with accuracy 95.46% and macro-F1 0.8854; NMRI sensitivity remains limited. The findings characterize training-budget sensitivity and preprocessing effects. Matched-resource search comparisons and independent evaluation remain necessary before claiming search-efficiency gains or deployment performance.

**Keywords:** intrusion detection; Internet of Things; binary particle swarm optimization; neural architecture search; class imbalance; reproducibility.

## 1. Introduction

An intrusion detector must recognize uncommon attacks while controlling false alarms. Designing a hybrid neural detector adds choices about convolutional depth, recurrent units, optimization and training duration. Architecture search explores these choices, but training every candidate at a large budget is expensive. Smaller budgets are useful only when their validation scores remain informative about later performance.

Lu et al. [1] proposed BPSO-AHDL-IDS to automate hybrid network design. This project independently implements its reported search mechanics and evaluates the experimental conditions under which candidate fitness is meaningful. It asks whether early fitness identifies useful later learners after ordinary training and preprocessing choices are examined. Single-class prediction collapse or tied early scores alone cannot establish a need for a new optimizer.

The contributions are an executable budget-constrained search framework; a repeated-seed, four-preset budget study; normalization, neural-component and repeated-state controls; and a fixed-endpoint reference test study across three dedicated training seeds. These are implementation and empirical contributions. No new particle update, search speedup or verified reproduction of the source paper's numerical results is claimed.

## 2. Related Work

BPSO-AHDL-IDS [1] combines a binary architecture representation with particle swarm optimization and hybrid convolutional/recurrent networks. Its reported Gas accuracy of 0.9897 and F1 of 0.9744 are external reference measurements. Unknown partition identities, feature ordering, seeds and multiclass averaging prevent a controlled numerical comparison with this project.

Random search [2] provides an unguided architecture-selection control. Successive halving [3] promotes promising candidates, Hyperband [4] combines halving brackets, and DEHB [5] combines differential evolution with Hyperband. BOHB [6] provides another established model-based allocation approach; it was not implemented here. These methods motivate explicit resource accounting and comparisons before claiming an optimization contribution. The Gas source documentation [7, 8] also motivates label verification and controls for repeated operating states.

## 3. Methodology

### 3.1 Data flow and model

The workflow prepares fixed training, validation and test partitions; fits preprocessing on training only; builds or searches a hybrid model; trains candidates using training/validation data; and evaluates a separately specified final model after a fixed endpoint. Test predictions do not enter candidate fitness, transform fitting or class-weight estimation.

Each tabular record is reshaped into a one-channel feature sequence. Convolution, activation and max pooling extract local feature-axis representations. The tensor is transposed for sequence-preserving recurrent blocks, and a linear classifier reads the last sequence position. Recurrence therefore models positions within a feature vector, not successive network events. The implementation supports RNN, LSTM, GRU and linear recurrent alternatives; native recurrent gates are retained, with the encoded activation applied after a block.

The fixed reference uses 16 convolutional filters, kernel 3, same padding, ReLU and pool size 2, followed by a 32-unit GRU, tanh post-activation, dropout 0.1 and an eight-class output. It contains 5,128 trainable parameters. The reference is a conventional preset and is not presented as the winner of a completed full-budget BPSO search.

### 3.2 Binary representation and search

The 73-bit representation allocates eight bits to batch size, optimizer and learning rate; two to convolution count; four ten-bit convolution slots; one to recurrent count; and two eleven-bit recurrent slots. Inactive slots are removed from phenotype cache keys. Pooling, output aggregation, count mappings and feature ordering are documented implementation choices because the source description does not fully determine them.

BPSO updates velocity using inertia, personal-best attraction and global-best attraction. A sigmoid converts velocity to the probability of a binary position. The selected source-equation interpretation replaces out-of-bound velocities with independent uniform draws on [0, 1]. Personal and global bests update only for strictly greater validation macro-F1, and only evaluated candidates can be returned.

Implemented controls are BPSO, reduced-budget BPSO, BPSO with successive halving, random search, Hyperband, DEHB and a conventional random-forest proposal control. Search settings include explicit epoch or elapsed-time caps. Equal epoch caps do not equalize FLOPs or actual resource consumption; model size, batch size, caching and incomplete promotions affect cost. Full paper settings of 20 particles, 30 populations and 30 candidate epochs were configured but not executed at full scale in this study.

### 3.3 Training and continuation

The evaluated neural controls use Adam at learning rate 0.001, batch size 256, balanced cross entropy and gradient clipping at norm 1. Class weights are inverse training frequencies normalized to unit mean. Validation and test frequencies do not determine these weights. Model, optimizer, data-loader generator and random-number states are preserved for cumulative-budget training. Cached candidate evaluations are reused observations rather than independent replicates.

## 4. Data and Experimental Protocol

### 4.1 Dataset and preprocessing

The original author-linked Gas release contains 97,019 records, 26 numeric predictors and eight verified classes. A stratified split using seeds 1729/1730 produces 58,211 training, 19,404 validation and 19,404 test records. These are project-defined partitions rather than recovered source-paper split identities. Alternate target columns are excluded. Numeric imputation and scaling parameters use training rows exclusively.

| Code | Class | Train | Validation | Test |
| --- | --- | --- | --- | --- |
| 0 | Normal | 36,694 | 12,231 | 12,231 |
| 1 | NMRI | 1,657 | 553 | 553 |
| 2 | CMRI | 9,280 | 3,093 | 3,093 |
| 3 | MSCI | 469 | 157 | 156 |
| 4 | MPCI | 4,582 | 1,527 | 1,528 |
| 5 | MFCI | 344 | 114 | 115 |
| 6 | DoS | 1,102 | 368 | 367 |
| 7 | Recon | 4,083 | 1,361 | 1,361 |

NMRI denotes naive malicious response injection; CMRI, complex malicious response injection; MSCI, malicious state command injection; MPCI, malicious parameter command injection; MFCI, malicious function command injection; DoS, denial of service; and Recon, reconnaissance [7, 8]. The feature named time is a query–response interval, not an acquisition timestamp, and cannot establish chronological ordering.

Column L2 divides predictors by training-column norms. Row L2 normalizes each record by its own norm. Min–Max fits feature extrema using training only. The normalization study retains identical partition rows and training-fitted imputation across all views. Original ToN-IoT, Bot-IoT and SWaT data are unavailable; synthetic profiles are used solely for software integration and are excluded from research performance tables.

### 4.2 Validation experiments

Four fixed hybrid presets use CNN/GRU widths 8/16, 16/32 and 32/64, with a stacked alternative adding a second CNN32 block to the reference. All four reach 3, 10 and 30 epochs under column L2 for seeds 42, 43 and 44. The reference also uses Min–Max and row L2. Min–Max component controls comprise an MLP with widths 64/32, a CNN16 with a flattened 32-unit head and a GRU32. Their capacities and readouts differ, so these are practical baselines rather than capacity-matched causal ablations.

The extension includes 27 neural trajectories through 30 epochs and four prespecified reference continuations to 200 epochs: column L2 at all three seeds and Min–Max at seed 42. Continuations preserve earlier weights and are not independent final restarts. Validation-study data directories omit test arrays.

Predictor-only hashes of the 25 non-time features define a separate state-grouped split. Its 7,675 groups yield 62,198 training, 14,462 validation and 20,359 reserved rows, with zero exact group overlap between partitions. Training-fitted imputation and Min–Max are refitted for this split. Random forests, logistic regression and the reference hybrid are evaluated across three training seeds. The reserved partition is not evaluated. This is a repeated-state sensitivity analysis, not a capture-level or prospective temporal experiment.

### 4.3 Final reference testing and metrics

After validation, the fixed Min–Max reference is independently initialized at seeds 10042, 10043 and 10044 and trained for exactly 200 epochs per seed. The protocol is recorded before generating new test predictions. Intermediate validation peaks do not select checkpoints or stop training. Fresh preparation verifies the original source checksum and all partition row identities. The test partition has been used previously and is not newly blinded.

Macro-F1 averages class-level F1 over all eight classes, with zero assigned to undefined cases. Accuracy, macro precision/recall, weighted F1, per-class recall, confusion matrices and normal-class false-positive rate are retained. A majority baseline selects its class from training counts. Means and sample standard deviations describe training variability on fixed partitions; they are not independent-dataset uncertainty estimates.

Random-split neural runs use an NVIDIA RTX 3050 Laptop GPU with 4 GB, Python 3.14.4, PyTorch 2.15.0 and CUDA 13.2. Grouped neural controls use CPU. Recorded GPU seconds describe synchronized elapsed occupancy, not energy use. Hardware and row-distribution differences prevent treating cross-split performance differences as a controlled causal estimate of leakage.

## 5. Results and Discussion

### 5.1 Training-budget sensitivity

| Preset | Epoch 3 macro-F1 | Epoch 10 macro-F1 | Epoch 30 macro-F1 |
| --- | --- | --- | --- |
| Small | 0.0460 ± 0.0460 | 0.0794 ± 0.0632 | 0.1724 ± 0.0478 |
| Reference | 0.0243 ± 0.0094 | 0.1309 ± 0.1186 | 0.4267 ± 0.2074 |
| Wide | 0.0705 ± 0.0550 | 0.4912 ± 0.0715 | 0.7348 ± 0.0868 |
| Stacked | 0.0194 ± 0.0142 | 0.2903 ± 0.2629 | 0.4735 ± 0.1835 |

Values show mean ± sample SD across three training seeds. At epoch 10, the winning preset is also the epoch-30 winner for all three seeds, with zero selection regret and Spearman correlation of 0.8 in each seed. At epoch 3, only seed 43 incurs positive regret, 0.0284. Thus this four-preset cohort does not demonstrate a severe late-winner selection problem. Constant rankings and tied selection boundaries remain explicitly non-identifiable rather than being resolved into artificial certainty. Four presets cannot establish ranking reliability across the full binary search space.

![Figure 1. Validation budget sensitivity in the four-preset cohort.](../latex/figures/extended_fidelity.png)

### 5.2 Normalization and component controls

| Normalization | Reference validation macro-F1 | Mean accuracy |
| --- | --- | --- |
| Column L2 | 0.4267 ± 0.2074 | 0.3136 |
| Row L2 | 0.7190 ± 0.0390 | 0.6710 |
| Min–Max | 0.7831 ± 0.0768 | 0.6550 |

The scaling choice materially changes bounded validation fitness. This supports examining preprocessing before attributing weak early scores to the optimizer. It does not establish universal Min–Max superiority across budgets or data releases.

![Figure 2. Reference validation learning curves; shading denotes sample SD across training seeds.](../latex/figures/extended_normalization.png)

| Model | Min–Max validation macro-F1 at epoch 30 | Parameters |
| --- | --- | --- |
| MLP | 0.8325 ± 0.0012 | 4,072 |
| CNN | 0.8275 ± 0.0032 | 7,016 |
| GRU | 0.7510 ± 0.0709 | 3,624 |
| CNN–GRU | 0.7831 ± 0.0768 | 5,128 |

MLP and CNN-only controls have higher mean validation macro-F1 than the hybrid at this budget. The reference falsely flags 53.24% of normal validation records on average, while the MLP falsely flags 42.17%. These rates prevent a deployment-readiness claim despite useful aggregate scores. Different capacities and readouts prevent attributing the differences solely to convolution or recurrence.

### 5.3 Longer training and state grouping

At 200 epochs, column-L2 reference validation macro-F1 reaches 0.8390 ± 0.0084; the single Min–Max continuation reaches 0.8407. The normalization comparison is budget-dependent and has unequal seed counts at this endpoint. These are validation continuations, not the final test runs below.

| Grouped validation method | Macro-F1 | Accuracy |
| --- | --- | --- |
| Random forest | 0.8372 ± 0.0010 | 0.8480 ± 0.0005 |
| Logistic regression | 0.8018 ± 0.0000 | 0.6806 ± 0.0000 |
| CNN–GRU | 0.7559 ± 0.0804 | 0.6077 ± 0.1273 |

The original random split has 2,033 shared training–validation non-time predictor hashes and 2,087 shared training–test hashes; the corpus has 105 non-time predictor hashes associated with conflicting labels. Grouping removes exact shared states but cannot remove all near duplicates or shared capture conditions. Its different class proportions and row counts prevent interpreting the score change as a measured causal leakage effect.

### 5.4 Fixed 200-epoch test results

| Final seed | Validation macro-F1 | Test accuracy | Test macro-F1 | Normal FPR |
| --- | --- | --- | --- | --- |
| 10042 | 0.8416 | 0.7242 | 0.8456 | 0.4268 |
| 10043 | 0.8423 | 0.7236 | 0.8441 | 0.4271 |
| 10044 | 0.8436 | 0.7243 | 0.8465 | 0.4271 |

Mean test macro-F1 is 0.8454 ± 0.0012, mean accuracy is 0.7240, and mean normal-class false-positive rate is 0.4270. These scores describe three independent initializations of a fixed reference on the same historical split. They do not represent a BPSO-selected winner or a blind independent-release result. Saved predictions reproduce every reported classification metric and confusion matrix.

| Class | Mean precision | Mean recall | Mean F1 | Test support |
| --- | --- | --- | --- | --- |
| Normal | 0.9847 | 0.5730 | 0.7244 | 12231 |
| NMRI | 0.0944 | 0.9439 | 0.1717 | 553 |
| CMRI | 0.9374 | 0.9845 | 0.9604 | 3093 |
| MSCI | 0.9677 | 0.9615 | 0.9646 | 156 |
| MPCI | 0.9810 | 0.9804 | 0.9807 | 1528 |
| MFCI | 1.0000 | 0.9652 | 0.9823 | 115 |
| DoS | 0.9944 | 0.9646 | 0.9792 | 367 |
| Recon | 1.0000 | 1.0000 | 1.0000 | 1361 |

Normal recall is approximately 0.5730 and NMRI precision is approximately 0.0944, despite NMRI recall of 0.9439. About 42.70% of normal test records are falsely flagged, predominantly as NMRI. This explains why strong attack recall and macro-F1 coexist with substantially lower accuracy. The fixed protocol is retained without post-test tuning; these false alarms remain a practical limitation.

![Figure 3. Mean final test recall and sample SD across three dedicated training seeds.](../latex/figures/final_class_recall.png)

![Figure 4. Row-normalized test confusion matrices averaged across final seeds.](../latex/figures/final_confusion.png)

The earlier ten-epoch column-L2 reference achieved test macro-F1 0.299371 and accuracy 0.707792, missing CMRI and MSCI completely. That historical result is preserved. Both normalization and training duration change in the final study, so the score difference cannot isolate one factor or establish an architecture-search gain.

### 5.5 Software and evidence verification

The current CPU and CUDA-configured suites each pass 74 tests. Device-aware tests exercise CUDA; component-continuation checks explicitly use CPU. Tests cover training-only transforms, binary encoding, tensor shapes, maximization, checkpoint continuation, search without test arrays, resource caps, native scheduler replay and tie-aware rankings.

All 30 bounded integration commands pass across real Gas and three synthetic profiles, exercising preparation, training/testing, search, diagnostics, seven-method comparison, reports and search resume. Earlier validation evidence is rechecked against 94 saved prediction records and hashes. Dependency consistency and Python compilation pass. Integration budgets verify execution and cannot establish fair-compute superiority. Synthetic scores are excluded from the research findings.

### 5.6 Validation-selected correction of false alarms

The balanced-loss reference falsely flagged 42.70% of normal test records. An exploratory follow-up compared unweighted and square-root inverse-frequency weighting using the same hybrid architecture, Min–Max preprocessing, Adam settings and a fixed 200-epoch endpoint. Candidate training used seed 42 and directories containing only training and validation arrays. Unweighted and square-root-weighted loss produced validation false-positive rates of 2.0685% and 2.0930%, with attack detection of 90.9104% and 91.3007%, respectively. Square-root weighting had the higher validation macro-F1, 0.8552 versus 0.8480.

Neither candidate satisfied the initial target of at most 1% false alarms and at least 90% attack detection. This failed target was retained in the evidence. An explicitly revised exploratory limit of 3% false alarms, with the same attack-detection constraint, was recorded before new test predictions. Square-root weighting was then selected by validation macro-F1. Three final runs used paired seeds 10042, 10043 and 10044 and exactly 200 epochs each on GPU. No model parameters, thresholds or endpoints were changed after reading those new test results.

| Test measure | Earlier balanced loss | Selected square-root weighting |
|---|---:|---:|
| Normal false-positive rate | 42.70% | 1.90% |
| Attack detection recall | 98.48% | 90.96% |
| Accuracy | 72.40% | 95.44% |
| Macro-F1 | 0.8454 | 0.8551 |

Values are means across three training seeds on the same historical test rows. Normal false alarms decreased by 40.80 percentage points. Attack detection also decreased, so the correction has a measurable sensitivity trade-off. A lower false-alarm rate does not imply every attack class improved.

| Class | Earlier mean recall | Corrected mean recall | Corrected mean precision |
|---|---:|---:|---:|
| Normal | 0.5730 | 0.9810 | 0.9487 |
| NMRI | 0.9439 | 0.0054 | 1.0000 |
| CMRI | 0.9845 | 0.9845 | 0.9383 |
| MSCI | 0.9615 | 0.9615 | 0.9677 |
| MPCI | 0.9804 | 0.9804 | 0.9810 |
| MFCI | 0.9652 | 0.9652 | 1.0000 |
| DoS | 0.9646 | 0.9600 | 0.9944 |
| Recon | 1.0000 | 1.0000 | 1.0000 |

This correction was motivated by known historical errors and validated on the existing development split. It does not constitute blind independent confirmation, a new search algorithm or achievement of the original 1% false-alarm target. All three final prediction archives independently reproduce their reported metrics.

### 5.7 Dedicated NMRI specialist and remaining ambiguity

Further validation experiments compared the existing hybrid score with Extra Trees, histogram gradient boosting and a neural specialist. The specialists were fitted only on normal and NMRI training records. The neural specialist used GPU training for 100 epochs; the tree methods used CPU. Only predictions assigned to normal or NMRI by the fixed hybrid were reconsidered; other attack decisions were preserved. Threshold calibration included false alarms already assigned to other classes, and threshold ties remained normal.

No tested method reached the initial target of at least 50% NMRI recall with at most 5% false alarms and at least 90% overall attack detection. An exploratory follow-up was therefore explicitly recorded: select the highest validation macro-F1 among settings with at most 3% false alarms, at least 10% NMRI recall and at least 90% attack detection. Histogram gradient boosting was selected from validation data; its seed-42 validation NMRI recall was 18.44% at 2.95% false alarms. Before new test predictions, thresholds were calibrated separately for each of the three existing hybrid models using validation data only.

A subsequent validation-selected normal/NMRI specialist increased mean NMRI test recall from 0.54% to 16.82%, while normal false alarms rose from 1.90% to 2.60%. Accuracy was 95.46%, macro-F1 0.8854, and overall attack detection 92.23%. This improves the observed trade-off, but most NMRI attacks are still missed. The same training-fitted specialist was shared across three hybrid seeds; these are not three independent specialist fits or independent-dataset replicates.

| Test operating point | Normal false alarms | NMRI recall | Accuracy | Macro-F1 |
|---|---:|---:|---:|---:|
| Original balanced hybrid | 42.70% | 94.39% | 72.40% | 0.8454 |
| Square-root-weighted hybrid | 1.90% | 0.54% | 95.44% | 0.8551 |
| Hybrid with NMRI specialist | 2.60% | 16.82% | 95.46% | 0.8854 |

The training-only ambiguity audit found that 1,374 of 1,657 NMRI records shared their processed non-time predictor state with normal records; those states also contained 26,843 normal training records. Duration was excluded from that audit. This demonstrates substantial overlap, but it is not a proof that the full feature set cannot distinguish the classes. The small candidate comparison also cannot prove that no better model exists.

All three saved test prediction archives reproduce the reported metrics. A saved inference pipeline reproduces all 19,404 test decisions for the prespecified first hybrid seed. The revised CPU and CUDA-configured regression suites each pass 80 tests. The same previously examined split was reused, so all follow-up findings remain exploratory and require independent confirmation.

![Figure 5. Validation NMRI recall versus measured normal false-alarm rate for the tested routing policies.](../latex/figures/nmri_tradeoff.png)

## 6. Limitations and Future Work

The four-preset cohort does not replace a diverse 25-architecture repeated-seed study or a complete 200-epoch architecture cohort. Full-budget BPSO, random search, Hyperband and DEHB comparisons under matched actual resources remain outstanding. Equal epoch caps alone would not ensure equivalent compute. A fixed reference test score cannot identify a search-efficiency benefit.

The existing test split is development-known and retains repeated operating states. No independent capture, acquisition chronology or independent release is available. Feature-axis recurrence does not establish temporal-event modeling. Unknown source-paper splits, averaging definitions and layer conventions prevent exact reproduction. Original ToN-IoT, Bot-IoT and SWaT experiments remain unexecuted. No new optimizer, statistical superiority, energy saving, multi-GPU equivalence or deployment readiness is claimed.

Future work should freeze a contribution-focused protocol, acquire capture-aware or independent data, complete matched-resource search controls and quantify uncertainty across independent experiments. Mechanism-specific ablations are needed before attributing gains to one architectural or allocation choice.

## 7. Reproducibility

Experimental repeatability is supported by preserving the source checksum, predictor order, partition identities, preprocessing parameters and training settings. Imputation, scaling and class weights are estimated exclusively from training data. Validation studies use common candidate identities across seeds 42, 43 and 44, while final reference models restart at seeds 10042, 10043 and 10044 and are tested after exactly 200 epochs. Checkpoints preserve model parameters, optimizer state and random-number states for cumulative training. Saved predictions are independently re-scored to verify classification metrics and confusion matrices. Means and sample standard deviations describe training variability on fixed partitions; independent releases remain necessary for assessing broader generalization.

## 8. Conclusion

This study developed and evaluated a budget-constrained framework for automated hybrid intrusion detection. Training duration and feature normalization materially influence candidate fitness: at 30 epochs, Min–Max yields reference validation macro-F1 0.7831, compared with 0.4267 under column L2. Within the four-preset cohort, the epoch-10 winner remains the winner at epoch 30 for all three seeds. These findings support examining ordinary training choices before attributing weak early fitness to an architecture-search limitation.

Three independently initialized reference models achieve test macro-F1 0.8454 ± 0.0012 at the fixed 200-epoch endpoint. This characterizes a fixed hybrid reference on the existing split rather than a BPSO-optimized winner. Class-level errors, false alarms and repeated predictor states remain central limitations. The work contributes an executable experimental framework and measured evidence about training-budget sensitivity; establishing search-efficiency gains and deployment performance requires matched-resource comparisons and independent evaluation data.

The exploratory validation-selected square-root weighting correction reduced mean normal test false alarms from 42.70% to 1.90%, with accuracy 95.44%, macro-F1 0.8551 and attack detection 90.96%. However, NMRI recall fell from 94.39% to 0.54%. This is a false-alarm versus attack-sensitivity trade-off, not a complete detector-quality solution; the original 1% false-alarm target remains unmet. Independent evaluation is still required.

A subsequent validation-selected normal/NMRI specialist increased mean NMRI test recall from 0.54% to 16.82%, while normal false alarms rose from 1.90% to 2.60%. Accuracy was 95.46%, macro-F1 0.8854, and overall attack detection 92.23%. This improves the observed trade-off, but most NMRI attacks are still missed. The same training-fitted specialist was shared across three hybrid seeds; these are not three independent specialist fits or independent-dataset replicates.

## References

1. K.-D. Lu et al., “BPSO-AHDL-IDS: Binary Particle Swarm Optimization-Based Automated Hybrid Deep Learning Model for Intrusion Detection of Internet of Things,” IEEE Transactions on Automation Science and Engineering, vol. 22, pp. 15859–15877, 2025. DOI: 10.1109/TASE.2025.3572510.
2. J. Bergstra and Y. Bengio, “Random Search for Hyper-Parameter Optimization,” Journal of Machine Learning Research, vol. 13, pp. 281–305, 2012.
3. K. Jamieson and A. Talwalkar, “Non-stochastic Best Arm Identification and Hyperparameter Optimization,” AISTATS, 2016.
4. L. Li et al., “Hyperband: A Novel Bandit-Based Approach to Hyperparameter Optimization,” Journal of Machine Learning Research, vol. 18, no. 185, pp. 1–52, 2018.
5. N. Awad, N. Mallik and F. Hutter, “DEHB: Evolutionary Hyberband for Scalable, Robust and Efficient Hyperparameter Optimization,” IJCAI, 2021.
6. S. Falkner, A. Klein and F. Hutter, “BOHB: Robust and Efficient Hyperparameter Optimization at Scale,” ICML, 2018.
7. T. Morris and W. Gao, “Industrial Control System Traffic Data Sets for Intrusion Detection Research,” Critical Infrastructure Protection VIII, pp. 65–78, 2014. DOI: 10.1007/978-3-662-45355-1_5.
8. J. Hsu, D. Mudd and Z. Thornton, “Project Report – SCADA Anomaly Detection,” 2014. Dataset-owner-linked report.
