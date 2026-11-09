# Paper specification and reproduction boundary

Source: supplied 19-page IEEE PDF, Lu et al., IEEE TASE 22 (2025), pp. 15859–15877, DOI 10.1109/TASE.2025.3572510. Read text and rendered Algorithms 1–4, Fig. 5, Tables II–III and final architecture figures. Page references below use printed page numbers. This project is an independently implemented, paper-informed baseline, not an exact reproduction.

## Architecture and objective (explicit, pp. 15863–15867)

Fig.1 explicitly describes a one-dimensional CNN. Samples flow through a variable CNN followed by a variable sequence of simple RNN, LSTM, GRU or linear layers and a classification output. Training minimizes cross entropy on W_train; the search maximizes F1 on W_vali; W_test evaluates the selected model. The mathematical formulation does not specify the tensor reshape, pooling operator, temporal window, output aggregation, classifier head, recurrent gate activations, initialization or loss regularization coefficients.

## Complete reported search fields

Fig. 5 specifies the following ordered mappings (big-endian binary indices):

| Field | Bits | Values in encoding order |
|---|---:|---|
| Batch size | 2 | 256, 512, 1024, 2048 |
| Optimizer | 3 | SGD, momentum, Nesterov, Adagrad, Adamax, Adam, RMSprop, Adadelta |
| Learning rate | 3 | 1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2 |
| CNN layer count | 2 | Mapping not given |
| CNN padding | 1 | same, valid |
| CNN filters | 3 | 8, 16, 32, 64, 128, 256, 384, 512 |
| CNN kernel | 2 | 2, 3, 4, 5 |
| CNN pool size | 2 | 1, 2, 3, 4 |
| CNN activation | 2 | sigmoid, tanh, relu, none |
| Recurrent layer count | 1 | Mapping not given |
| Recurrent type | 2 | simrnn, lstm, gru, linear |
| Recurrent units | 3 | 8, 16, 32, 64, 128, 256, 384, 512 |
| Recurrent activation | 2 | sigmoid, tanh, relu, none |
| Recurrent regularization | 2 | none, L1, L2, L1&L2 |
| Recurrent dropout | 2 | 0, 0.1, 0.2, 0.3 |

CNN blocks use 10 bits, recurrent blocks 11 bits. L=sum_j l_j (Eq.16). Fig.5(a) appears variable length; it does not explain crossing layer counts while updating fixed-length BPSO vectors. In Fig.5(d), the drawn string has an apparent extra digit relative to the stated 11 bits; field widths and ordered mappings are used here. Eq.14 calls pt a pooling type but defines it as padding; Fig.5 calls it pa. Eq.15 lists regularizers in a different order from Fig.5; encoding follows Fig.5.

## BPSO (explicit, Eqs.17–21 and Algorithms 1–4)

Initialize binary particles; initialize velocities (distribution unspecified). Evaluate all particles, update personal and global best, update inertia, velocities, then positions, for Gmax evaluations of each population. Strictly better fitness replaces personal best. For bit d of particle i:

v = w*v + c1*r1*(pbest-x) + c2*r2*(gbest-x)

w = wmax - (wmax-wmin)*t/Gmax

If v is outside [vmin,vmax], Eq.19 replaces it with Uniform[0,1], rather than clipping it. The surrounding prose also mentions resampling in [vmin,vmax]; this implementation follows Eq.19 and Algorithm 4. Position is 1 when Uniform[0,1] <= sigmoid(v), else 0. Random values are independent by particle/bit/update.

Algorithm 4 line 5 says personal-best fitness < global-best fitness before replacing global best. This contradicts maximization and is treated as a printed inequality error: the implementation uses >. Algorithm 1 selects from the last, potentially unevaluated updated population; this implementation returns the best evaluated personal best.

## Explicit experimental settings (Table III, pp.15868,15871–15872)

| Setting | Bot-IoT / ToN-IoT / Gas | SWaT |
|---|---:|---:|
| Population | 20 | 30 |
| Generations | 30 | 20 |
| Search training TE1 | 30 epochs | 30 epochs |
| Final training TE2 | 200 epochs | 200 epochs |
| Inertia max/min | 0.8 / 0.4 | same |
| Velocity max/min | 10 / -10 | same |
| c1 / c2 | 1.5 / 1.5 | same |

Hardware: four GTX1080Ti GPUs. Software: Python3.8.10, PyTorch1.8.2, NumPy1.19.3, Pandas1.3.1. N and Gmax chosen by trial and error. No explicit data balancing. No seeds, repeated-run count, optimizer secondary settings, train/validation/test fractions, sample IDs, or exact feature list are supplied. The cited frameworks are historical and not the environment installed by this project.

## Datasets and preprocessing (Table II, pp.15869–15870)

| Dataset | Samples | Features | Classes | Normalization |
|---|---:|---:|---:|---|
| Bot-IoT | 6,706,455 | 10 | 5 | Min-Max |
| ToN-IoT | 461,043 | 44 | 10 | Min-Max |
| Gas Pipeline | 97,019 | 26 | 8 | L2 |
| SWaT | 944,919 | 51 | 2 | Min-Max |

Bot classes: normal, DoS, DDoS, reconnaissance, theft. Fig.14 ToN classes: backdoor, DDoS, DoS, injection, Mitm, normal, password, ransomware, scanning, Xss. Fig.16 Gas: CMRI, DoS, MFCI, MPCI, MSCI, NMRI, normal, reconnaissance. SWaT: normal/attack; seven normal-operation days and four attack-operation days, 36 attack scenarios. No definitive feature-selection procedure or raw-file revision is given. Bot-IoT and ToN download links are interchanged in the paper. Eq.27 states n is sample count, suggesting feature-column L2 normalization; axis remains ambiguous. No treatment of categorical data, missing data, duplicates or label-derived feature removal is stated. Exact reproduction splits cannot be recovered from these descriptions. Figure confusion-matrix counts could inform split hypotheses but do not establish sample identities or a splitting algorithm.

## Final models inferred from figures

Fig.15 Gas: batch1024, Adamax, lr0.01; one CNN (same,384,kernel2,pool4,relu); GRU256,relu,L1&L2,dropout0.2. Fig.13 ToN: batch512, Adamax, lr0.01; two active CNNs (same,384,2,3,relu), (same,512,3,1,tanh); GRU16,relu,none,0 then GRU32,relu,L1&L2,0.2. The figure includes inactive CNN entries despite a count of two, supporting fixed maximum slots as an inference. These are provided as fixed configurations. Final training restart versus continuation is not specified. No selection of test epoch by test performance is valid here.

## Reported results (paper values, not project measurements)

| Dataset | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| Bot-IoT | .9832 | .9700 | .9724 | .9712 |
| ToN-IoT | .9959 | .9788 | .9822 | .9804 |
| Gas Pipeline | .9897 | .9843 | .9650 | .9744 |
| SWaT | .9992 (Table IX) | .9962 | .9969 | .9965 |

Metric equations are binary TP/TN expressions; multiclass averaging is not established. Paper notes weaker Theft and Mitm detection, expensive offline search, and no model complexity objective. It does not experimentally establish early-versus-late ranking reliability or prove that existing multi-fidelity methods cannot address cost. Friedman/Quade tests on reported competitors do not establish seed-level reproducibility of this implementation.

## Configurable implementation decisions (not paper facts)

Use fixed four CNN slots and two recurrent slots; counts decode as 1–4 and 1–2, giving 8+2+40+1+22=73 bits. Inactive slots are ignored in phenotype cache keys. Uniform binary positions, Uniform[-10,10] velocities. Initial generation t=0; Gmax populations evaluated. Strict ties keep previous best. Use one-dimensional feature-axis CNN (N,1,F), max pooling, transpose to (N,L,C), sequence-preserving recurrent/linear layers, activation after each layer and last-position classifier. Recurrent modules use PyTorch gate defaults; encoded activation is post-layer, not custom LSTM/GRU gates. Pad short sequences only as needed before valid convolution/pooling so all codes remain constructible; record repair count. This changes receptive fields compared with rejecting invalid candidates and requires ablation.

Use train-only median numeric imputation, constant categorical imputation and one-hot encoding with ignored unknown categories. Train-only Min-Max; Gas column L2 fitted on training (configurable row L2). Default unprovided splits are seeded stratified 60/20/20, explicitly not a recovered paper split. Official train/test files override random test splitting; validation comes only from training. Chronological and grouped splits are available and preferred for dependence-sensitive studies. Explicit feature allowlists are required before paper-comparison claims; defaults remove known alternate targets/IDs/timestamps. No balancing by default. Macro-F1 over the entire training class vocabulary is the search objective, with weighted-F1 also reported. L1/L2 coefficients default 1e-5 as an implementation choice. Final selected model restarts with a dedicated seed, trains on training partition for exactly 200 epochs, and evaluates test once. Validation is not merged into training. Cache scope includes data hash, full config, phenotype, seed and cumulative budget; promotions resume the same training trajectory.

## Additional figure inconsistencies (visual audit)

Fig.11 Bot winner uses batch **4096**, outside the Fig.5 batch-size range (maximum2048). Its four CNNs are (same,64,3,3,sigmoid), (same,64,2,2,relu), (valid,512,5,4,sigmoid), (same,384,2,4,relu); two LSTM256 with tanh, regularization none/dropout.2 then L1&L2/dropout.1. A separate fixed model uses this figure value; the search space is not silently expanded.

Fig.17 SWaT winner lists batch512, Adamax, lr.001, four CNNs with filters8,128,128,**1024** (1024 is outside the encoding range). It also disagrees internally about the first CNN activation (text sigmoid, diagram relu), the active recurrent count (one but two slots shown), and recurrent activation (text tanh, diagram sigmoid). The generic SWaT configuration intentionally uses a clearly labelled Gas sanity architecture pending a chosen figure interpretation. These inconsistencies prevent an exact encoding-level reproduction of all final winners.

The paper's overall motivation and comparisons support investigating compute cost; they do not quantify energy savings, provide repeated independent search variance, or resolve the low-fidelity ranking hypothesis.


## Additional reported optimizer comparison (Table IX, p.15875)

BGA-AHDL-IDS is reported at (accuracy, precision, recall, F1): Bot-IoT(.9883,.9859,.9320,.9581); ToN-IoT(.9873,.9677,.9789,.9736); Gas(.9863,.9783,.9576,.9678); SWaT(.9990,.9964,.9945,.9954). BPSO values match the main result table above; SWaT accuracy is explicitly.9992 here despite omission from the abstract/TableVIII. The paper supplies no full BGA operator configuration supporting an exact BGA reproduction.

Supplied PDF SHA256: `f16df10af0b345218bc1dd09a1117f4f859d53bcbbdb77a191965d3597aaa057`. Fig.13 also uses an inactive CNN `linear` activation, whereas the searched activation table says `none`; active blocks avoid this naming ambiguity.

## Subsequent data-source verification

The author-linked `gas_final.arff` was retrieved on2026-11-05 from the official university URL recorded in gas_pipeline_source.json. Inspection found97,019 rows,26 numerical predictors and target `result` with codes0–7, matching TableII dimensions. gas_pipeline.yaml now explicitly lists these raw predictors. This resolves the local Gas file/schema placeholder, not the unknown paper partitions, preprocessing details, label mapping provenance or feature-order interpretation. The old Gas data release also carries an author warning about spurious attack-identifying patterns; those limits must be retained when interpreting future runs.

Optional training-only class weighting and a conventional larger reference were later added as documented ablations after inspecting collapsed low-budget models. Neither changes the default unweighted paper-informed protocol. Diagnostic loss/architecture improvements do not constitute a new NAS method; their measurements and remaining limits are in results_review.md.
