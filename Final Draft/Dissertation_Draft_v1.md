# ARES PdM: A Digital-Twin Framework for Predictive Maintenance of Tactical Unmanned Aerial Vehicles using Deep Learning and Explainable Artificial Intelligence

**Author:** Swarnim Mandal
**Programme:** MSc (Computer Science / Data Science)
**Supervisor:** *[to be confirmed]*
**Institution:** University of West London
**Submission:** *[Month] 2026*

---

## Declaration

I declare that this dissertation is the result of my own work. Where the work of others has been used or referenced, this is acknowledged in accordance with the University's regulations on academic conduct. No part of this submission has previously been submitted for any other academic award.

Signed: ____________________     Date: ____________________

---

## Acknowledgements

I would like to thank my supervisor for their patient direction throughout this project, the staff of the School of Computing for the hardware and software infrastructure made available during the practical phase, and my family for their continued support during the writing of this dissertation.

---

## Abstract

Tactical unmanned aerial vehicles (UAVs) are increasingly deployed in time-critical surveillance, logistics, and defence operations, yet their maintenance regimes remain dominated by fixed-interval inspections that neither reflect the airframe's actual condition nor exploit the dense telemetry such platforms produce. This dissertation presents *ARES PdM* (Aerial Reliability and Endurance System for Predictive Maintenance), a digital-twin framework that couples a deep-learning prognostics pipeline to a real-time tactical heads-up display, with the objective of providing operators with auditable, condition-based maintenance guidance for a heterogeneous UAV fleet.

Three deep-learning architectures — a stacked Long Short-Term Memory (LSTM) network, a four-block one-dimensional Convolutional Neural Network (1D-CNN), and a Transformer encoder — are trained for Remaining Useful Life (RUL) regression on two complementary datasets: the publicly available NASA Commercial Modular Aero-Propulsion System Simulation (C-MAPSS) turbofan benchmark, used as an external-validity anchor, and a procedurally generated multirotor UAV fleet of twenty airframes flown across ten flights each under four health classes (healthy, bearing wear, ESC thermal, battery degradation). On the UAV branch each model is trained jointly for RUL regression and four-way fault classification. Predictions are made interpretable through Transformer self-attention visualisation and Integrated Gradients sensor-importance attribution, both surfaced as first-class elements of a Next.js operator dashboard. A FastAPI inference service streams synthetic UAV telemetry from a server-side flight generator over WebSockets, completing the closed digital-twin loop from synthetic physics through trained model to operator decision.

On the C-MAPSS benchmark (combined test set of 707 engines across FD001–FD004) the Transformer attains an RMSE of 16.40 cycles and a NASA Score of 4 074, with the LSTM close behind at 16.47 / 4 224 and the 1D-CNN at 18.49 / 6 516. On the synthesised UAV fleet (85 292 validation windows) the Transformer attains an RMSE of 3.45 flight-hours and a fault-classification accuracy of 95.5 %; the 1D-CNN attains 4.08 RMSE but only 61.5 % fault accuracy, and the LSTM 4.75 RMSE at 88.0 % fault accuracy. The contribution of this work is threefold: an open, reproducible pipeline that unifies a turbofan benchmark and a UAV synthetic fleet under a single training driver; an explainability layer that exposes attention maps and Integrated Gradients attributions directly inside an operational HUD; and an architectural pattern in which a single static-artifact contract reconciles offline reproducibility with live inference.

**Keywords:** predictive maintenance, remaining useful life, digital twin, unmanned aerial vehicle, Transformer, explainable AI, Integrated Gradients, condition-based maintenance.

---

## Table of Contents

1. Introduction
2. Background
3. Literature Review
4. Methodology
5. System Design and Implementation
6. Results and Evaluation
7. Discussion
8. Conclusions and Future Work
9. References
10. Appendices

---

# 1. Introduction

## 1.1 Context and Motivation

Maintenance is, in operational terms, the largest single contributor to the through-life cost of an aerial platform. For tactical unmanned aerial vehicles, this proportion is amplified by two compounding factors. First, the airframes themselves are often relatively inexpensive compared with the payloads, the operator training, and the loss-of-mission cost incurred when a unit is grounded. Second, the duty cycles imposed on tactical UAVs — short turnaround, high-tempo deployment, frequent reconfiguration — invalidate the assumptions on which fixed-interval inspection regimes were originally calibrated. A maintenance schedule designed for a benign training environment does not transfer cleanly to a contested operational one.

Condition-Based Maintenance Plus (CBM+) and the broader discipline of Prognostics and Health Management (PHM) propose that sensor data generated during normal operation can substitute for, or at least augment, time-based inspection. Where reactive maintenance addresses faults after they manifest and preventive maintenance addresses them on a calendar, predictive maintenance estimates a *Remaining Useful Life* (RUL) for each component or system and schedules intervention only when that estimate crosses an actionable threshold. The promise is significant — fewer unscheduled removals, longer effective service life per airframe, and earlier warning of catastrophic failure — but its realisation depends on the availability of run-to-failure data, on models that can extract degradation signatures from that data, and on a delivery mechanism that places those predictions in front of an operator in a form they can act upon.

The maturity of deep learning over the last decade, in particular the demonstrated ability of recurrent and attention-based architectures to learn long-range temporal dependencies in multivariate signals, has shifted the prognostics community decisively toward data-driven RUL estimation. Yet two persistent gaps remain. The first is *credibility*: a model that predicts a fleet should be grounded in three days will not be trusted unless the operator can interrogate the basis of that prediction. The second is *operational realism*: a model whose only output is an offline RMSE on a benchmark dataset has not yet demonstrated that it can serve a live decision in a streaming context.

This dissertation addresses both gaps in the context of a tactical UAV digital twin.

## 1.2 Aim and Research Question

**Aim.** To design, implement, and evaluate a digital-twin framework that delivers explainable, condition-based predictive-maintenance guidance for a tactical UAV fleet, using deep learning on heterogeneous run-to-failure data and serving predictions to an operator dashboard.

**Research question.** *Can a single, reproducible deep-learning pipeline simultaneously deliver competitive Remaining Useful Life estimates on an established turbofan benchmark and a synthetic UAV fleet, while exposing the basis of those estimates through attention and gradient-based attribution at a fidelity sufficient for operator-level decision-making?*

## 1.3 Objectives

1. To construct a unified data pipeline ingesting the NASA C-MAPSS turbofan benchmark and a procedurally generated multirotor UAV fleet, with consistent windowing, normalisation, and RUL-target conventions.
2. To implement and train three deep-learning architectures — LSTM, 1D-CNN, and Transformer encoder — under a single configuration-driven training driver, yielding six trained models in total.
3. To extract two complementary explainability artifacts — self-attention maps and Integrated Gradients sensor attributions — and to bake them into a per-asset payload consumed by the dashboard.
4. To implement a FastAPI inference service that streams synthetic UAV telemetry and live RUL/fault predictions over a WebSocket protocol.
5. To implement a Next.js operator dashboard with four functional views (Armory, Mission, Diagnostics, Lab) reconciling offline replay with live inference under a single state model.
6. To evaluate the framework quantitatively (RMSE, NASA Score, classification accuracy, training time) and qualitatively (interpretability of XAI outputs).

## 1.4 Scope and Delimitations

The work is scoped deliberately. The UAV fleet is synthesised, not flown — public run-to-failure data for tactical-class multirotors at a fidelity sufficient for supervised RUL training does not exist at the time of writing, and procurement of such data is outside the resources of an MSc project. The C-MAPSS dataset is therefore retained as the external-validity anchor; if the pipeline can match published baselines on a benchmark known to the community, it strengthens the credibility of the same pipeline's results on the synthesised fleet. The dashboard is curated to twelve assets — eight turbofans and four quadrotors — so that the three-dimensional carousel and the per-asset diagnostic views remain legible. Production-grade concerns such as authentication, multi-tenant deployment, and over-the-wire encryption are explicitly out of scope.

## 1.5 Contributions

The principal contributions of this dissertation are:

- A reproducible training pipeline that unifies C-MAPSS and a procedural UAV fleet under a single driver, exposing six trained models through a stable artifact contract.
- A digital-twin operator interface in which Transformer attention and Integrated Gradients attributions are surfaced as primary visual elements rather than auxiliary diagnostics.
- A two-tier serving architecture that decouples reproducible offline replay from live streaming inference through a single per-asset JSON schema, allowing the same dashboard to operate in both modes without code branching at the component level.
- A quantitative comparison of LSTM, 1D-CNN, and Transformer architectures on both a turbofan benchmark and a UAV-class joint regression-and-classification task.

## 1.6 Dissertation Structure

Chapter 2 establishes the technical background in PHM, deep sequence models, digital twins, and explainable AI. Chapter 3 reviews the directly relevant literature, organised by theme, and identifies the gap that ARES PdM addresses. Chapter 4 specifies the methodology — datasets, preprocessing, model architectures, training protocol, and explainability extraction. Chapter 5 documents the system design and implementation, including the inference backend and the operator dashboard. Chapter 6 reports quantitative and qualitative evaluation results. Chapter 7 discusses the implications, threats to validity, and operational reading of the findings. Chapter 8 concludes and identifies avenues for further work.

---

# 2. Background

This chapter establishes the conceptual and technical foundations on which the remainder of the dissertation rests. It is intended for a reader familiar with software engineering and introductory machine learning but not necessarily with the prognostics literature.

## 2.1 From Reactive to Predictive Maintenance

Maintenance regimes are conventionally classified along an axis of increasing temporal foresight. *Reactive* (or corrective) maintenance acts only once a fault has manifested; it minimises planned downtime but maximises unplanned downtime and accepts the worst-case cost of in-flight failure. *Preventive* maintenance acts on a fixed calendar or duty-cycle schedule derived from population statistics; it reduces the rate of unplanned failure but is intrinsically conservative, because the schedule must be safe for the worst-degraded unit in the population, which means most units are serviced earlier than necessary.

*Predictive* maintenance, sometimes denoted PdM or — in the U.S. military context — Condition-Based Maintenance Plus (CBM+), conditions the maintenance decision on the *individual asset's* current and projected health. It rests on three premises: (i) that operationally-relevant degradation produces observable signatures in available sensor channels; (ii) that those signatures can be modelled with sufficient fidelity to support a decision; and (iii) that the decision can be delivered to the operator within the time horizon over which it remains actionable.

The output of a predictive-maintenance system is typically expressed either as a *health index* (a scalar in some bounded range whose decline tracks degradation) or as a *Remaining Useful Life* (RUL) estimate (a regression of remaining operating time, cycles, or flights to a defined failure threshold). RUL is the more operationally useful of the two because it is dimensioned in the same units the operator already plans against, but it is also harder to estimate, because it is forward-looking rather than instantaneous.

## 2.2 Run-to-Failure Data and the C-MAPSS Benchmark

Supervised RUL estimation requires run-to-failure data: trajectories that begin in a healthy regime, traverse one or more degradation modes, and terminate at a defined end-of-life condition. Such data is rare in the open domain because operators of high-value aerial platforms do not, as a rule, run their assets to destruction in a controlled fashion. The principal exception is the NASA Commercial Modular Aero-Propulsion System Simulation (C-MAPSS) dataset, which provides four subsets — FD001 through FD004 — of simulated turbofan engine degradation. The subsets vary in the number of operational conditions and fault modes, ranging from a single condition with a single fault mode (FD001) to six operational conditions with two interacting fault modes (FD004). Each engine instance contributes a multivariate time-series of twenty-one sensor channels and three operational settings, terminated at the cycle at which a defined health threshold is crossed.

C-MAPSS is widely regarded as the de facto benchmark for data-driven RUL estimation. Its stylised simplicity — discrete cycles, no missing data, well-bounded sensor counts — has the dual effect of making the benchmark accessible and of inviting models to overfit to its peculiarities. A piecewise-linear RUL target, clipped at a maximum of typically 125 cycles (Heimes 2008), is the conventional convention, justified by the observation that very early in an engine's life the true RUL is unknown and the model should not be penalised for failing to predict it accurately.

The conventional evaluation metrics are root mean square error (RMSE), which is symmetric, and the *NASA Score*, which penalises late predictions (over-estimating remaining life) more heavily than early predictions (under-estimating). The asymmetry encodes the safety preference: an early warning permits a planned removal, whereas a late warning may permit a failure in service.

## 2.3 Tactical UAV Health Monitoring

Tactical UAVs differ from turbofans in almost every respect that matters for prognostics. Their sensor suites are dense but oriented toward flight control rather than condition monitoring; the dominant degradation mechanisms are battery capacity fade, motor and ESC wear, propeller and bearing imbalance, and thermal excursions; their operational profiles are short and heterogeneous, with no clean analogue to the turbofan's "cycle"; and the safety case is governed less by a single end-of-life threshold than by a constellation of fault thresholds any one of which is mission-disqualifying.

Public run-to-failure datasets in this regime are scarce. The ALFA dataset of fixed-wing fault flights and the NASA-style multirotor testbeds reported in the recent literature are the closest analogues, but their fault taxonomies and instrumentation differ from the tactical-multirotor case of interest in this dissertation. The methodological response — and the one adopted here — is to *synthesise* a fleet under a documented procedural model that captures the relevant degradation phenomenology, while retaining C-MAPSS as the external-validity anchor for the modelling pipeline.

## 2.4 Deep Sequence Models for RUL Estimation

Three families of deep-learning architectures dominate the RUL literature, and all three are evaluated in this dissertation.

The *Long Short-Term Memory* (LSTM) network is a recurrent architecture whose gated cell state preserves information across long input sequences, mitigating the vanishing-gradient pathology of vanilla recurrent networks. LSTMs were the dominant architecture for sequence modelling throughout the mid-2010s and remain a strong baseline for RUL because the underlying task — extracting a slow degradation trend from a long, noisy multivariate signal — aligns naturally with the recurrent inductive bias.

The *one-dimensional Convolutional Neural Network* (1D-CNN) treats the multivariate time-series as a one-dimensional signal with one channel per sensor and applies stacked convolutions to extract local temporal features. CNNs are parameter-efficient, parallelisable on a GPU, and well-suited to the case in which the relevant degradation signature is a localised pattern (a vibration spike, a thermal anomaly) rather than a long-range dependency.

The *Transformer encoder*, introduced in the natural-language processing literature and subsequently adopted across modalities, dispenses with recurrence entirely. Each timestep attends to every other timestep through a learned multi-head self-attention mechanism. This makes long-range dependencies natively first-class — there is no information bottleneck to traverse — and, critically for this dissertation, it produces an attention matrix that is itself an interpretable artifact of the model's reasoning. The cost is quadratic memory in the sequence length, which constrains the practical lookback window.

## 2.5 Digital Twins

A *digital twin* is, in the broadest contemporary usage, a computational replica of a physical asset, kept in continuous correspondence with the asset by a stream of sensor data and capable of returning predictive or prescriptive outputs to the operator. The concept originated in product-lifecycle management and has since been adopted in aerospace, energy, and manufacturing. For aviation predictive maintenance, the digital twin is increasingly framed as the integrating layer that mediates between heterogeneous data sources (telemetry, maintenance logs, design data) and the AI models that consume them. Recent reviews in the aviation digital-twin literature emphasise that real-time twins for prognostics cannot, in general, be served from a centralised cloud — the data volumes (a modern twin-engine commercial aircraft can generate of the order of twenty terabytes per engine per hour) and the latency budgets favour an *edge–fog–cloud* architecture in which inference occurs as close to the asset as practicable. This consideration directly informs the FastAPI-on-localhost serving design adopted in this dissertation.

## 2.6 Explainable Artificial Intelligence in PHM

A predictive-maintenance system whose outputs cannot be interrogated will not be trusted, and, in regulated domains, may not be permitted into service. Explainable Artificial Intelligence (XAI) provides a family of techniques for surfacing the internal basis of a model's prediction. Two classes of technique are used in this dissertation.

*Attention visualisation* exploits the fact that the Transformer encoder's self-attention mechanism is itself a soft alignment over input timesteps. The attention matrix can be extracted directly from the trained model and rendered as a heatmap, revealing which past timesteps the model weighted most heavily when producing the prediction at a given point in time. The interpretability claim attaches to attention weights only with care — they are a necessary but not sufficient indicator of feature importance — but in the present application, where the alternative is an opaque scalar, the attention map provides a substantive improvement in model auditability.

*Integrated Gradients* (IG) is a gradient-based attribution method that integrates the gradient of a model's output with respect to its input along a straight-line path from a chosen baseline input to the actual input. The resulting attribution satisfies the *completeness* axiom (the per-feature attributions sum to the difference in model output between baseline and input) and is, unlike vanilla saliency, robust to gradient saturation. In the present context IG is integrated over the lookback window to produce a per-sensor importance vector, surfaced in the dashboard as a bar chart adjacent to the wireframe view.

## 2.7 Summary

The chapter has established that predictive maintenance for tactical UAVs is a problem of estimating Remaining Useful Life from multivariate time-series under a regime of scarce run-to-failure data; that deep sequence models — LSTMs, 1D-CNNs, and Transformers — are the established tooling for that estimation; that digital twins and edge-served inference are the architectural pattern under which such models are operationalised; and that explainability is not a peripheral concern but a precondition for operational trust. The next chapter places these elements in the context of the recent peer-reviewed literature.

---

# 3. Literature Review

This chapter reviews the literature pertinent to ARES PdM under five thematic groupings: classical and emerging predictive-maintenance practice; data-driven RUL estimation on C-MAPSS; UAV-specific prognostic studies; digital-twin architectures for aviation; and explainability as a regulatory and operational concern. Each subsection closes with an explicit identification of the gap addressed by this dissertation.

## 3.1 Predictive Maintenance in Aviation: From Practice to Data-Driven Models

The transition from preventive to predictive maintenance in aviation has been gradual and is conditioned on the regulatory caution that characterises the sector. Costa et al. (2026), in a PRISMA systematic review of twenty studies between 2017 and 2025, document that the field has converged on hybrid CNN–RNN deep-learning architectures as the dominant prognostic substrate, while identifying four persistent barriers to adoption: explainability of model outputs, regulatory certification of AI components, heterogeneity of source data across operator fleets, and the practical difficulty of scaling experimental pipelines from a single airframe to a fleet. The review further observes that civil aviation, with its open datasets and well-defined certification regime, dominates the literature, while military and tactical operators — for whom this dissertation is most directly relevant — are under-represented because of restricted data publication.

The implication for the present work is twofold. First, the choice of a *hybrid* methodological posture — combining a deep-learning model zoo with explicit explainability outputs — aligns with the consensus direction. Second, the under-representation of tactical-class platforms in the literature legitimates the synthesised UAV fleet as a methodological response, provided its construction is documented transparently.

## 3.2 Data-Driven RUL on C-MAPSS

C-MAPSS has, since its release, accumulated an extensive performance literature. Early work used hand-engineered features (kurtosis, skewness, spectral moments) fed into shallow regressors; subsequent work transitioned to LSTM and bi-directional LSTM architectures operating directly on the windowed multivariate input; the most recent body of work is dominated by attention-based and Transformer-style models, frequently in combination with convolutional front-ends. Reported RMSE on the simplest subset, FD001, has declined from upwards of twenty cycles in early work to the low teens in current state-of-the-art submissions, with comparable improvements on the more challenging subsets. The NASA Score, by virtue of its asymmetric penalty, exhibits much wider variance and is more sensitive to occasional late predictions on long-tailed engines.

Two methodological observations recur and inform this dissertation's design choices. First, RUL clipping (the piecewise-linear target) is near-universal at a clip value of 125 cycles; departures from this convention complicate cross-paper comparison and are therefore not adopted here. Second, ensembling and per-subset hyperparameter tuning produce headline numbers that are difficult to reproduce without access to the specific training configuration; this dissertation therefore reports single-seed, single-configuration results across a single training driver, accepting some loss of headline performance in exchange for reproducibility.

The gap addressed: a Transformer-based RUL pipeline for C-MAPSS that is reported jointly with results on a UAV-class dataset under an identical training driver, providing a controlled cross-domain comparison rather than a single-domain benchmark.

## 3.3 UAV-Specific Prognostic Studies

The directly UAV-focused literature is younger and more fragmented. Ozkat et al. (in the *International Journal of Micro Air Vehicles*) report a vibration-based RUL estimator for a DJI M600 multirotor with a grooved propeller blade. Their pipeline samples the IMU at 800 Hz, applies a moving-median plus bandpass filter, computes a spectrogram and extracts a mean peak-frequency feature, and trains a two-layer LSTM (100 + 50 cells, dropout 0.2) to forecast five future values from the last seven. They report RMSE in the range 1.35 to 3.71 Hz over three replications with a maximum look-ahead of ten seconds; their healthy fundamental frequency is approximately 40 Hz, and the failure threshold is set at 50 Hz mean peak frequency. The study is the closest single analogue in the literature to the LSTM branch of the present work, and it informs three design choices here: the use of vibration channels as a primary input (carried forward in the synthesised UAV's per-motor vibration RMS series), the LSTM as one of the three baseline architectures, and the framing of the problem as forward forecasting from a rolling window.

Bondyra et al. (2022), in the journal *Drones*, present *CompactNeuroUAV*, an architecture combining a convolutional feature-compaction front-end with a bidirectional GRU temporal encoder, evaluated on the ALFA dataset of fixed-wing fault flights (a Carbon Z T-28 with simulated engine shutdown, elevator-stuck, rudder full-deflection, and aileron-stuck faults). The architecture produces three outputs — per-parameter deviation, set-of-parameter deviation, and fault-class probability — and the authors flag the need for an on-board neural processing unit to meet real-time latency targets at the edge. This work is the principal direct competitor architecture to the Transformer branch of the present work; the present dissertation departs from it in three respects: a Transformer encoder is preferred to a BiGRU on grounds of attention auditability; the regression head (RUL) is retained alongside the classification head (fault class) as a joint output; and the platform is multirotor rather than fixed-wing.

Darrah et al., in their PHM 2021 contribution on a UAV simulation testbed, document a PostgreSQL-backed Simulink simulation tracking forty-four run-to-failure UAVs across 3 624 flights, with battery internal-resistance growth as the dominant failure mode (degradation of the order of one thousand per cent over the asset lifetime, with safety-threshold violation typically around the eightieth flight). Their data-management framework explicitly enumerates reproducibility, explainability, extensibility, and maintainability as first-class design requirements. The synthesised UAV fleet in the present dissertation is conceptually descended from this work, although it operates at smaller scale (twenty airframes, ten flights apiece) consistent with an MSc resourcing envelope.

A complementary thread is the survey of Sai et al. (2023, IEEE), which positions UAV predictive maintenance within a broader AI-for-UAVs taxonomy spanning path planning, control, networking, object detection, collision avoidance, and swarm intelligence. The survey identifies IMU-based vibration analysis and LSTM-based anomaly detection on healthy-only training data as the two principal data-driven PdM modalities currently in the literature, and it further notes that reinforcement-learning controllers are beginning to be applied to motor thermal management as a pre-emptive degradation control. Al-Darraji et al. (2021) provide a complementary engineering taxonomy of UAV navigation sensors organised by Level-of-Autonomy, useful here principally for grounding the sensor-channel taxonomy of the synthesised UAV fleet.

The gap addressed: existing UAV PdM studies have evaluated either a single architecture on a single bespoke dataset (Ozkat) or a single architecture on a public fault-flight dataset (Bondyra). A controlled three-way architectural comparison on a UAV-class dataset, reported alongside C-MAPSS results from the same training driver, has not, to the author's knowledge, been published.

## 3.4 Digital Twins for Aviation Predictive Maintenance

Mehdipour et al. (2026, *IET Digital Twins*) provide a PRISMA-ScR scoping review of eighty-three digital-twin studies in aviation. Three contributions of that review bear directly on the architectural choices in this dissertation. First, the authors categorise digital-twin data into three types — time-series, graphical, and natural-language — and observe that *multi-model* databases handle this heterogeneity more naturally than either pure relational or pure NoSQL alternatives. Second, they quantify the data scale at which a real-time twin must operate, with a contemporary twin-engine commercial aircraft generating of the order of twenty terabytes per engine per hour. Third, and most directly relevant here, they recommend a three-layer *edge–fog–cloud* architecture for real-time digital-twin inference, on the grounds that centralised cloud serving cannot meet the latency budgets imposed by safety-of-flight applications. The FastAPI-on-localhost serving pattern adopted in the present work is a specific instance of the edge tier in that taxonomy.

The gap addressed: published aviation digital-twin work tends to focus on the architectural pattern in the abstract or on the data-management substrate; an end-to-end implementation in which a deep-learning prognostic model is served from an edge process to an operator dashboard, in a reproducible open-source form, is rarer.

## 3.5 Explainability as Regulatory and Operational Necessity

Costa et al. (2026) and Mehdipour et al. (2026) both identify explainability as a *barrier* to adoption rather than as a desirable add-on. The framing matters: in a certification regime under which an aviation safety authority must approve the use of a model in service, the absence of an explainable account of the model's prediction is not a stylistic deficiency but a blocker. Sundararajan et al.'s Integrated Gradients (in the original 2017 ICML paper) and the broader attention-visualisation literature provide the techniques; the operational question is whether those techniques can be surfaced inside the operator's workflow rather than as a separate post-hoc analysis. This is the question the present dissertation addresses on the dashboard side.

A complementary thread is the physics-informed approach exemplified by Duan et al. (2026) in *Batteries*, who couple a thermodynamics and fluid-dynamics residual to a deep-learning loss for UAV lithium-ion battery RUL, reporting improvements over SVM, LSTM, and GAN baselines across a wide range of temperatures, C-rates, humidities, and wind speeds. The present dissertation does not implement a physics-informed loss but does include a procedural battery model in the synthesised UAV fleet whose voltage sag depends on internal resistance, current draw, and state of charge in the manner motivated by Duan et al.'s work.

The gap addressed: most XAI work in PdM is reported as a separate analysis chapter; integration of attention maps and Integrated Gradients attributions into the *operator-facing* dashboard, as primary visual elements rather than auxiliary diagnostics, is not the dominant pattern in the published literature.

## 3.6 Synthesis

Across the five thematic groupings, four cross-cutting observations frame the contribution of this dissertation. The dominant data-driven prognostic substrate in aviation is now hybrid deep learning, with attention-based Transformers under-explored relative to LSTM and CNN-RNN combinations. Public UAV run-to-failure data is scarce, and synthesis under a documented procedural model is a defensible methodological response. Real-time digital-twin inference favours an edge-served architecture for latency reasons. And explainability is not optional. ARES PdM is positioned at the conjunction of these four observations.

---

# 4. Methodology

This chapter specifies the data, models, training protocol, and explainability procedures used in this dissertation. The intent is reproducibility: a reader with the source repository and an Apple Silicon machine should be able to reproduce the headline numbers reported in Chapter 6 by executing `make phase1`. All randomness — fleet generation, mini-batch shuffling, weight initialisation — is seeded with a fixed value of 1 337, set centrally in `ml/config.py`.

## 4.1 Datasets

### 4.1.1 NASA C-MAPSS

The C-MAPSS dataset comprises four subsets, FD001 through FD004, each containing a training set of run-to-failure trajectories and a test set of trajectories truncated at a randomly chosen point before failure, together with a vector of true RUL values for the test trajectories. Each trajectory contributes twenty-one sensor channels (fan and core temperatures, pressures, speeds, fuel and bleed quantities) and three operational settings. FD001 contains a single operating condition and a single fault mode; FD004 contains six operating conditions and two simultaneous fault modes, and is correspondingly the most challenging subset. The combined test set across all four subsets contains 707 engines, and the combined sliding-window training set, after preprocessing, contains 125 819 windows.

Acquisition of the dataset is automated through `scripts/fetch_cmapss.py`, which falls back from the primary mirror at `phm-datasets.s3.amazonaws.com` to a GitHub raw mirror. The script is idempotent: re-execution against an existing local copy is a no-op.

Preprocessing is identical across the four subsets. After per-subset z-score normalisation, fourteen sensor channels (indices 2, 3, 4, 7, 8, 9, 11, 12, 13, 14, 15, 17, 20, 21 in the original C-MAPSS numbering) are retained as the informative subset following constant-channel pruning, in line with the convention established in the literature. The RUL target is computed as the residual cycle count from each timestep to the trajectory's terminal cycle and clipped at a maximum of 125 cycles (Heimes 2008). Sliding windows of length thirty cycles are extracted with a stride of one, yielding the input tensors of shape *(window=30, channels=14)* on which the models are trained.

### 4.1.2 Synthesised UAV Fleet

The UAV fleet is generated by `ml/data/uav_synth.py` and produces twenty airframes, each flying ten missions of duration uniformly distributed between 800 and 1 300 seconds (approximately thirteen to twenty-two minutes). Each airframe is assigned at instantiation to one of four health classes — *healthy*, *bearing wear*, *ESC thermal*, or *battery degradation* — drawn from a categorical distribution with probabilities `[0.25, 0.30, 0.25, 0.20]`. Within each non-healthy airframe a single motor (of four) is selected at each flight as the locus of the injected fault.

The procedural flight model produces twenty-one sensor channels per timestep at 10 Hz. These channels are: time `t`; flight phase index `phase`; per-motor vibration RMS `vib_rms_m1..m4` (with healthy baselines around 0.25–0.65 g RMS, modulated by load and corrupted by Gaussian noise); per-motor RPM `motor_rpm_m1..m4` (a load-modulated baseline near 8 200 RPM with Gaussian noise); per-motor ESC temperature `esc_temp_m1..m4` (a load-modulated baseline near 28–50 °C); body-frame IMU acceleration in three axes `imu_accel_x/y/z`; body-frame IMU gyroscopic rate in three axes `imu_gyro_x/y/z`; battery pack voltage `batt_voltage`, current `batt_current`, and state-of-charge `batt_capacity_pct`; and `altitude` integrated from acceleration. The procedural battery model derives pack voltage from a nominal cell voltage, the internal resistance, and the instantaneous current draw, with internal resistance scaling by a fault-conditional life fraction.

Fault progression is parameterised by a `life_frac` quantity defined as the cumulative-flight-hours-into-life divided by the global constant `END_OF_LIFE_HOURS = 4.0`. Healthy airframes still age uniformly across all components — a small global creep is added to vibration and ESC temperature so that the model can estimate RUL from sensors rather than predicting a constant. Bearing-wear faults inject a slow drift plus a cyclic scrape on the chosen motor's vibration channel and a mild secondary heating on its ESC; ESC-thermal faults inject a heat-soak excursion that accumulates with cumulative load; battery-degradation faults inflate internal resistance by a factor of up to 1.6× over life. The choice of `END_OF_LIFE_HOURS = 4.0` is a deliberate calibration: with ten flights of approximately fifteen minutes each per airframe, the airframe reaches `life_frac ≈ 0.75` by the final flight, producing RUL labels that span roughly the 25–100 range across the fleet. An earlier value of 200 hours, used in an initial iteration of the work, produced an RUL distribution so concentrated near the upper bound that a constant predictor achieved competitive performance and the regression collapsed to mean prediction.

A stratified hold-out split is applied at the *drone* level: each of the four fault classes is guaranteed to appear in both the training and validation partitions. Sliding windows of length fifty timesteps (corresponding to five seconds at the 10 Hz tick rate) are extracted with stride one, and the RUL target is clipped at 100 flight-hours. The resulting partitions contain 301 017 training windows and 85 292 validation windows.

### 4.1.3 Reproducibility

For C-MAPSS the canonical train–test split released with the dataset is preserved. All randomness is seeded as described above. Per-run training history is serialised to `artifacts/runs/<arch>_<dataset>/history.json` and is the data source for the training-curve panel in the Lab view of the dashboard.

## 4.2 Model Architectures

All three architectures are implemented in PyTorch and share an identical input contract — a *(batch, window, channels)* tensor — and a near-identical output contract: a scalar RUL prediction per window, plus, when configured for the UAV variant, a four-way fault-classification logit vector from a parallel head on the same shared encoder.

### 4.2.1 LSTM

The LSTM model (`ml/models/lstm.py`) comprises a two-layer stacked LSTM with hidden width 96 and inter-layer dropout 0.25. The final-timestep hidden state is fed to a regression head consisting of a linear layer projecting from 96 to 64 units, a ReLU activation, dropout, and a final linear projection to a one-dimensional output. The fault head, when active, is a single linear layer from the same 96-dimensional hidden state to four logits.

### 4.2.2 1D-CNN

The 1D-CNN (`ml/models/cnn.py`) treats the input as a *(channels, window)* signal. It comprises four convolutional blocks, each consisting of a one-dimensional convolution with sixty-four output channels and kernel size five (with same-padding), a batch-normalisation layer, and a ReLU activation. The output is reduced by global average pooling across the time dimension, passed through dropout, and projected through a regression head of the same shape as the LSTM's. The fault head, when active, is a single linear layer from the sixty-four-dimensional pooled feature.

### 4.2.3 Transformer Encoder

The Transformer model (`ml/models/transformer.py`) projects the input channels to a model dimension of 96 through a linear embedding, adds a fixed sinusoidal positional encoding of the same dimension across the window, and applies two custom encoder layers each consisting of a four-headed multi-head self-attention (with attention weights averaged across heads at output), a residual-and-LayerNorm step, a feed-forward sublayer with hidden dimension 192 and GELU activation, and a second residual-and-LayerNorm step. The output is mean-pooled across the time dimension and projected through a regression head consisting of a LayerNorm, a linear projection to 64, a GELU activation, dropout, and a final linear projection to a one-dimensional output. Dropout is set to 0.2. The encoder layer is implemented in-house rather than via `nn.TransformerEncoderLayer` so that the post-softmax attention matrix from the final layer can be exposed to downstream explainability code.

## 4.3 Training Protocol

A single configuration-driven driver, `ml/train.py`, orchestrates training of all six models. The CLI accepts a `--dataset` argument selecting `cmapss` or `uav` and an `--arch` argument selecting `lstm`, `cnn`, or `transformer`. Hyperparameters are loaded from `ml/config.py`, which exposes the device selection through a `get_device()` helper that prefers Apple Silicon's Metal Performance Shaders backend when available, falls back to CUDA on machines so equipped, and to CPU otherwise.

The optimiser is AdamW with weight decay 1×10⁻⁴; the learning rate is 1×10⁻³; the schedule is cosine annealing over the full epoch budget. The regression loss is mean squared error. For the UAV variant, the fault head is trained with cross-entropy, and the joint loss is the sum of the regression loss and the classification loss weighted by a coefficient of 0.5. Training is run for forty epochs at batch size 512 on C-MAPSS and for thirty epochs at batch size 256 on the synthesised UAV fleet. There is no early stopping; the cosine schedule decays the learning rate to zero at the final epoch, and the model state at that final epoch is the one used for evaluation.

A practical concern specific to the Transformer on Apple Silicon is that running a large validation set through the model in a single batch can exhaust the unified memory and trigger an MPS out-of-memory failure. Validation inference is therefore wrapped in a `_batched_predict(model, X, batch=256)` helper in `ml/export.py` that streams windows through the model in fixed-size batches and concatenates the outputs. The helper is also used during the export step that bakes per-asset payloads.

## 4.4 Explainability Extraction

### 4.4.1 Attention

The Transformer encoder layer is implemented in-house and returns the post-softmax attention matrix from each layer alongside its output (`attn_w` of shape *(batch, window, window)*, with multi-head attention averaged across the head dimension at the call site). The attention matrix from the final encoder layer is captured at inference time, serialised into the per-asset JSON payload as the field `attention_2d`, and rendered by the dashboard's Diagnostics view as a heatmap with axes labelled by relative timestep.

### 4.4.2 Integrated Gradients

Integrated Gradients are computed using the Captum library's `IntegratedGradients` implementation against the trained Transformer's regression output. The resulting attribution tensor of shape *(window, channels)* is reduced along the time axis by absolute-value mean to produce a *(channels,)* sensor-importance vector, which is normalised for display and serialised as the field `sensor_importance` in the per-asset payload, alongside the parallel `sensor_names` array. The dashboard's Diagnostics view renders this vector as a horizontal bar chart with the highest-attributed channel emphasised.

## 4.5 Live Inference Pipeline

Live inference is served by a FastAPI application defined in `backend/main.py` and run under uvicorn on port 8001. Three endpoints are exposed: `GET /healthz` returns service liveness; `POST /predict` accepts a single windowed input and returns a single prediction; and `WS /stream` serves a streaming session.

The streaming protocol is parameterised by query string: `fault` selects the fault class to inject, `hours` selects the simulated cumulative-flight-hours-into-life, `seed` selects the per-stream random seed, `rate_hz` selects the telemetry rate (default 10), and `stride` selects the prediction sub-rate (default 5, i.e. predict every fifth tick). On connection the server emits a single `meta` message describing the model architecture, device, and feature names, then a sequence of `tick` messages each containing the current sensor frame, and finally an `end` message at simulation termination. A `prediction` field is attached every `stride`-th tick once the rolling window buffer has accumulated enough history. The synthetic flight generator on the server side reuses the same procedural model as `ml/data/uav_synth.py`, ensuring that the live-inference distribution is consistent with the training distribution.

## 4.6 Evaluation Methodology

Quantitative evaluation reports the following per-architecture, per-dataset metrics: RMSE (cycles for C-MAPSS, flight-hours for UAV), the NASA Score for C-MAPSS, fault-classification accuracy for the UAV variant, total wall-clock training time, and number of training and validation windows. Per-epoch validation RMSE is also recorded across the full forty- or thirty-epoch run, providing the training-curve evidence presented in Chapter 6.

Qualitative evaluation comprises three exhibits per asset surfaced in the dashboard: the per-architecture RUL prediction overlay against the ground-truth trajectory in the Lab view; the attention heatmap in the Diagnostics view; and the sensor-importance bar chart in the Diagnostics view.

---

# 5. System Design and Implementation

This chapter documents the implementation of ARES PdM as a software system. It complements the methodology chapter by addressing the architectural choices made in service of the research objectives, with particular emphasis on the contract between the machine-learning pipeline and the operator dashboard.

## 5.1 Repository Layout and Conventions

The project is organised as a single repository with three principal subtrees. The Python ML pipeline lives under `ml/`, the FastAPI inference backend under `backend/`, and the Next.js dashboard under `app/`, `components/`, and `lib/`. Public artifacts written by the ML pipeline and read by the dashboard are written into `public/data/`, which is git-ignored. Trained checkpoints and per-run history are written into `artifacts/`, also git-ignored. The intent of this layout is that the repository can be cloned and reproduced from `make phase1` without committing any binary artifacts to version control.

A `Makefile` exposes phase-level targets — `make fetch` to download C-MAPSS, `make synth` to regenerate the UAV fleet, `make train` to train all six models, `make train-uav` for the UAV-only subset, `make export` to bake the per-asset payloads, `make figures` to regenerate the thesis figures, and `make backend` to launch the FastAPI server — together with a `phase1` aggregator that runs the full pipeline end-to-end. The dashboard development server is started by `npm run dev`.

## 5.2 The Static-Artifact Contract

The single most consequential architectural decision in ARES PdM is the static-artifact contract that mediates between the ML pipeline and the dashboard. Three files are written by `ml/export.py` and read by the dashboard.

`public/data/manifest.json` contains the fleet roster: an array of asset records, each with an identifier, a display name, an asset class (turbofan or quadrotor), the current RUL, a status enumeration (NOMINAL, WARNING, CRITICAL), and a `data_source` flag distinguishing baked from live assets. The manifest is fetched once on application boot, hydrating the Zustand store via the `AppBootstrap.tsx` component.

`public/data/assets/<id>.json` contains the per-asset payload, fetched on demand when the asset is selected. It carries six fields of substance: `predictions_per_arch` (a map from architecture name to RUL trajectory); `rmse_per_arch` (a map from architecture name to scalar RMSE); `sensor_names` (a string array of length F); `sensor_window` (a *(T, F)* array of the most recent input window); `attention_2d` (a *(T, T)* matrix from the Transformer); `sensor_importance` (an *(F,)* vector from Integrated Gradients); and an `anomaly` object describing the dominant fault component, severity, and timestamp.

`public/data/results.json` contains the Model Lab payload: an array of training-run objects, each with architecture name, dataset name, headline metrics (`rmse`, `score`, `fault_acc`, `epochs`, `train_size`, `test_size`, `seconds`), and a per-epoch `history` array recording train loss, validation RMSE, and the cosine-scheduled learning rate at each epoch.

The contract has three properties that recur as design patterns elsewhere in the system. It is *static*, in that no server is required to read it; it is *complete*, in that everything the dashboard renders for a given asset is contained in a single payload; and it is *symmetric*, in that the same field shapes are produced by the offline export and the live inference path. The third property is what permits the dashboard's component layer to remain mode-agnostic.

## 5.3 The Dashboard

The dashboard is a Next.js 14 application using the App Router, React 18, TypeScript, and Tailwind CSS for styling. Three-dimensional rendering is provided by `@react-three/fiber` with `drei` helpers; charting is provided by `recharts`; client-side state is managed by `zustand`. Four functional views are exposed.

The **Armory** view is the fleet inventory. A three-dimensional carousel renders procedural geometry for each of the twelve assets — eight turbofans and four quadrotors — under a single `UAVModel.tsx` component that branches on asset class. No GLTF models are loaded; geometry is constructed programmatically from primitives. Selecting an asset navigates to its diagnostic view.

The **Mission** view is the live operator HUD. It comprises a radar panel, a telemetry panel, a health bar, and a live chart, with controls to start, pause, and reset the mission. A switch toggles between PROCEDURAL mode (a heuristic flight envelope generated client-side by `lib/telemetry.ts`, requiring no backend) and LIVE INFERENCE mode (a WebSocket subscription to the FastAPI backend mediated by `lib/live.ts` and `lib/useLiveStream.ts`). The component layer of the HUD does not branch on mode; the Zustand store's `ingestLiveTick` method translates raw WebSocket frames into the abstract `TelemetryFrame` shape that the components consume, so all four panels work identically in both modes.

The **Diagnostics** view is the digital-twin inspection surface. A wireframe rendering of the selected asset (`WireframeView.tsx`) is rendered alongside the Transformer's attention matrix (`AttentionHeatmap.tsx`), the Integrated Gradients sensor-importance vector, and a maintenance-readout panel (`MaintenanceReadout.tsx`) that produces a verbal summary of the predicted health state.

The **Lab** view is the academic-evaluation surface. A results table (`ResultsTable.tsx`) compares all six models on their respective datasets across RMSE, NASA Score (where applicable), classification accuracy (where applicable), training-set size, validation-set size, training time, and epochs. Training-loss and validation-RMSE curves are rendered per model from the `history` array of `results.json` by `TrainingCurves.tsx`. A per-asset prediction overlay panel (`PerAssetOverlay.tsx`) allows the user to select any asset and overlay the trained architectures' predictions against the ground-truth RUL trajectory.

## 5.4 The Inference Backend

The FastAPI backend comprises three Python modules. `backend/inference.py` loads the trained UAV Transformer checkpoint at process start and exposes a thread-safe `predict(window)` function returning a regression scalar and a fault-class probability vector. `backend/synth_stream.py` implements the server-side flight generator and the rolling-window buffer, reusing the same procedural simulation as `ml/data/uav_synth.py` so that the live-inference distribution matches the training distribution. `backend/main.py` is the FastAPI application proper, defining the three endpoints described in §4.5 and wiring the application's startup event to load the model and report `arch`, `device`, and `len(feature_names)` to the console.

The application is run under uvicorn on port 8001 (chosen specifically to avoid the stale uvicorn instance frequently occupying port 8000 on the development machine), with the `--reload` flag during development.

## 5.5 The Two Modes of the Mission View

The Mission view's mode switch is the most operationally important piece of architectural plumbing in the dashboard. PROCEDURAL mode exists because a demo cannot assume that the Python toolchain is installed on the demo machine; it generates a heuristic but plausible flight envelope client-side and exercises every panel and chart in the HUD without any backend dependency. LIVE INFERENCE mode exists because a digital twin without a live data stream is not a digital twin; it surfaces the actual outputs of the trained Transformer as the HUD's primary content. The two modes are reconciled at the store layer rather than at the component layer: `ingestLiveTick` performs a structural translation from the raw UAV channel layout to the abstract `TelemetryFrame` slots that the components already consume from PROCEDURAL mode. The result is that the panels render identical UI in both modes, with the only difference being the provenance of the underlying numbers.

## 5.6 Reproducibility and Deployment Notes

A new contributor is expected to be able to reproduce the trained-model artifacts on Apple Silicon by cloning the repository, creating a Python 3.13 virtual environment, installing the pinned dependencies in `requirements.txt`, running `make phase1`, installing the Node dependencies with `npm install`, starting the dashboard with `npm run dev`, and, in a third shell, starting the backend with `make backend`. Two environmental considerations are documented in the project's CLAUDE.md guide. First, port 3000 is frequently occupied on the development machine, and Next.js auto-migrates to 3001 in this case; the backend is configured for 8001 specifically to avoid the analogous collision on port 8000. Second, after re-training, the backend must be restarted (`pkill -f "uvicorn backend.main" && make backend`) to pick up the new checkpoints, because the model is loaded once at process start.

Two consistency-of-thresholds points are also worth noting. The mapping `rul_to_status` (with `<35` CRITICAL and `<65` WARNING in RUL units) lives in two places by design: server-side in `ml/export.py`, where it is applied at bake time, and client-side in `lib/assets.ts`, where it is re-applied when live RUL values arrive. The duplication is intentional because the live re-classification path must not require a server round-trip. C-MAPSS RUL units are *cycles*, not percentages, and the dashboard accordingly clamps the roster bar's display width to `min(100, rul)`; conflating "RUL units" with "% of life" was an early-stage source of UI confusion and is documented in CLAUDE.md as a gotcha.

---

# 6. Results and Evaluation

This chapter reports the quantitative and qualitative results of training and serving the six-model zoo described in Chapter 4 under the system documented in Chapter 5. All numbers are reported at seed 1 337 under a single training driver invocation; ensemble or per-subset hyperparameter-tuned numbers, which would be higher, are not reported.

## 6.1 Quantitative Results: C-MAPSS

The combined test set across FD001–FD004 contains 707 engines, evaluated in the conventional fashion against true RUL values supplied with the dataset. The combined sliding-window training set contains 125 819 windows. All three models are trained for forty epochs at batch size 512 with the cosine-annealed AdamW optimiser described in §4.3. Headline numbers are summarised in Table 6.1.

**Table 6.1 — C-MAPSS Headline Results (combined test set, *n* = 707)**

| Architecture | RMSE (cycles) | NASA Score | Train time (s) |
|---|---:|---:|---:|
| 1D-CNN        | 18.49 | 6 515.8 | 87.6 |
| LSTM          | 16.47 | 4 224.0 | 164.8 |
| Transformer   | **16.40** | **4 073.9** | 413.8 |

The Transformer attains the lowest RMSE and the lowest NASA Score, with the LSTM close behind on RMSE but somewhat further behind on the score, and the 1D-CNN trailing on both metrics. The score gap between the 1D-CNN and the recurrent or attention-based models is markedly larger than the RMSE gap, which is consistent with the score's asymmetric penalty: a model whose RMSE is only a few cycles higher can still incur a substantially worse score if its excess error is distributed disproportionately on the late side of the true RUL. The ranking of the three architectures by training time inverts the ranking by accuracy: the Transformer takes approximately 4.7× longer to train than the 1D-CNN, reflecting the quadratic cost of self-attention in the sequence length.

The Transformer's combined RMSE of 16.40 is consistent with mid-tier published results for single-seed, single-configuration C-MAPSS evaluations. Frontier numbers in the very-low-teens are reported in papers that combine architectural innovations with per-subset hyperparameter tuning and ensembling; the present work foregoes both in exchange for reproducibility under a single training driver.

The per-epoch validation-RMSE curves (rendered in the dashboard's Lab view from the `history` field of `results.json`) show that all three models reach their final-epoch RMSE through the cosine-annealed schedule rather than at a clear early plateau, which justifies the choice not to apply early stopping. The Transformer's validation RMSE in particular continues to decline slowly through the final third of training, consistent with the larger model benefiting from the full forty epochs.

## 6.2 Quantitative Results: Synthesised UAV Fleet

The synthesised UAV fleet, comprising twenty airframes flown for ten flights each at 10 Hz, yields 301 017 training windows and 85 292 validation windows after the stratified hold-out split described in §4.1.2. All three models are trained for thirty epochs at batch size 256 with the same optimiser and schedule. Each model is trained with a joint regression-and-classification objective, weighted at 1.0 for the RMSE term and 0.5 for the cross-entropy fault term. Headline numbers are summarised in Table 6.2.

**Table 6.2 — Synthesised UAV Headline Results (validation set, *n* = 85 292 windows)**

| Architecture | RMSE (flight-hours) | Fault Accuracy | Train time (s) |
|---|---:|---:|---:|
| 1D-CNN        | 4.08 | 0.615 | 321.2 |
| LSTM          | 4.75 | 0.880 | 514.9 |
| Transformer   | **3.45** | **0.955** | 1 253.3 |

Two observations bear emphasis. First, the Transformer is the strongest model on both heads simultaneously, with an RMSE 15.4 % lower than the 1D-CNN's and a fault accuracy 34.0 percentage points higher. Second, the 1D-CNN's regression performance (4.08 RMSE) is in fact better than the LSTM's (4.75 RMSE), but its fault accuracy (61.5 %) is markedly worse, falling between random chance (25 % on a balanced four-way problem, but the class distribution is non-uniform) and the LSTM's 88.0 %. The likely reading is that the 1D-CNN's locally-convolutional inductive bias is well-suited to identifying gradual degradation trends — which dominate the regression target — but less well-suited to discriminating between the four fault signatures, three of which (bearing, ESC thermal, battery) manifest in distinct sensor channels rather than as locally-distinguishable temporal features. The Transformer's global attention and the LSTM's recurrent state both extract a richer representation for the classification head; the Transformer extracts the richer of the two.

The training-time ranking again inverts the accuracy ranking: the Transformer is approximately 3.9× slower to train than the 1D-CNN. On the development hardware (Apple Silicon, MPS backend) the absolute training time of approximately twenty-one minutes for the UAV Transformer remains practical for an MSc workflow.

The per-epoch validation-RMSE curves on the UAV branch show all three models continuing to improve through the cosine schedule, with the Transformer crossing below 2.0 RMSE on the validation set in late epochs (the headline 3.45 number is computed against the held-out validation windows after the final epoch and is dominated by the regression head's residuals on the most-degraded windows).

## 6.3 The Two Datasets in Comparison

The two datasets are not directly comparable on absolute metrics — RMSE in cycles cannot be transposed to RMSE in flight-hours, and the NASA Score has no analogue on the UAV branch — but the *relative* performance of the three architectures across the two datasets is informative. On C-MAPSS the Transformer's lead over the LSTM on RMSE is small (0.07 cycles, well within the noise of single-seed evaluation), and the lead is most pronounced on the NASA Score. On the UAV branch the Transformer's lead is much wider on the regression head (1.30 flight-hours over the LSTM, 0.63 over the 1D-CNN) and decisive on the classification head. The reading is that the architectural differences matter most where the task is most heterogeneous: the joint regression-and-classification UAV task, with four interacting fault signatures across twenty-one channels, is a regime in which the Transformer's global attention has a clearer advantage than on the more homogeneous C-MAPSS regression task.

## 6.4 Qualitative Results: Dashboard Surfaces

Three qualitative outputs are rendered in the dashboard for each asset. The Transformer's final-layer attention matrix is exposed as a heatmap in the Diagnostics view; the Integrated Gradients sensor-importance vector is rendered as a horizontal bar chart in the same view; and the per-architecture RUL prediction trajectories are overlaid against ground truth in the Lab view's per-asset overlay panel. These outputs are consumed directly from the per-asset JSON payload described in §5.2 and require no live backend connection.

The intended evaluation of these surfaces is operational rather than statistical: the question is whether the visualisations admit a usable interrogation of the model's prediction by an operator confronted with a low predicted RUL. The judgement that they do is, at the time of writing, the author's; no controlled operator study has been conducted, and the framework's qualitative evaluation should accordingly be read as an evaluation of the dashboard's *plausibility* as a clinical instrument rather than as a demonstration of its clinical utility under task. A controlled operator study is identified in §8.2 as a candidate for future work.

## 6.5 Live Inference

The FastAPI backend serves predictions over the WebSocket protocol described in §4.5, with the synthetic flight generator producing telemetry at 10 Hz and the Transformer running inference every fifth tick once the rolling buffer is full. The live HUD is exercised by switching the Mission view's mode from PROCEDURAL to LIVE INFERENCE; the rendered telemetry, health bar, and live chart are populated from the WebSocket stream rather than from the client-side procedural envelope. End-to-end live-inference performance has been verified qualitatively (smooth HUD updates, coherent fault progression) but has not been characterised quantitatively in this work; quantitative latency profiling is identified in §8.2 as a candidate for future work.

## 6.6 Summary

The Transformer is the strongest of the three architectures on both datasets and on both metric families. Its lead is modest on C-MAPSS RMSE, decisive on the NASA Score, and decisive on both the regression and classification heads of the synthesised UAV task. The 1D-CNN is the fastest to train and the strongest on the UAV regression head among the non-attention baselines, but is the weakest on the UAV classification head by a substantial margin. The LSTM occupies an intermediate position on the UAV classification head and is competitive with the Transformer on C-MAPSS RMSE while training in less than half the time.

---

# 7. Discussion

This chapter situates the Chapter 6 results in their operational and methodological context. It addresses why the Transformer outperforms its alternatives, the sensitivity of the synthesised UAV results to a critical configuration constant, the bounds within which the headline metrics can be transported to a real-world setting, and the threats to validity that should temper any operational reading of the framework.

## 7.1 Why the Transformer Wins

The Transformer's lead over the LSTM and the 1D-CNN is consistent with the broader sequence-modelling literature, but the proximate explanation differs between the two datasets. On C-MAPSS the Transformer's lead is small on RMSE (0.07 cycles over the LSTM) but more substantial on the NASA Score (150 score units over the LSTM, ~2 440 over the 1D-CNN). The asymmetric NASA Score penalises late predictions disproportionately, so the score gap implies that the Transformer's residuals are distributed less unfavourably on the late side than the LSTM's, and that the 1D-CNN's residuals are distributed considerably more unfavourably on the late side than either. A natural reading is that the global-attention inductive bias allows the Transformer to integrate evidence across the full thirty-cycle window when condition transitions or non-stationarities are present, whereas the 1D-CNN, with a finite receptive field constructed from four kernel-five layers, has a more local view and is more likely to over-shoot on the late side when the recent window is misleading.

On the synthesised UAV fleet the Transformer's lead is substantially wider on both heads, and this is best read as a consequence of the joint regression-and-classification objective. The 1D-CNN's regression performance is competitive (4.08 RMSE versus the Transformer's 3.45), but its classification performance (61.5 %) is markedly worse than the LSTM's (88.0 %) and the Transformer's (95.5 %). The reading offered in §6.2 — that the local-convolutional inductive bias is well-suited to gradual degradation but ill-suited to discriminating between fault signatures that manifest in distinct sensor channels — is consistent with the gap and is, in the author's judgement, the most plausible single explanation. The Transformer's advantage over the LSTM on the classification head (95.5 % vs 88.0 %) is also wider than on the regression head, which suggests that attention's ability to integrate information across the full fifty-step window is particularly valuable for the classification task, where the discriminative evidence may be concentrated at one or two timesteps within the window rather than smeared across it.

A secondary observation is that the Transformer's training time is approximately 2.5× the LSTM's on C-MAPSS and approximately 2.4× the LSTM's on the UAV fleet. The performance improvement is therefore not free; on a constrained edge platform the cost-benefit calculus might tip the other way. In the present application, with the trained model deployed to a local FastAPI process on a developer-class machine, the training-time differential is not a binding constraint, but it would become so on a strict edge-compute deployment.

## 7.2 The END_OF_LIFE_HOURS Calibration

The synthesised UAV fleet's `END_OF_LIFE_HOURS` constant, currently 4.0, is documented in the project's CLAUDE.md file as a configuration whose calibration was the subject of an explicit prior bug. An earlier iteration of the work used a value of 200 hours, which produced an RUL distribution so narrow — given that ten flights of approximately fifteen minutes each accumulate only about two and a half hours of life — that a constant predictor near the upper bound of the distribution achieved competitive performance. The regression collapsed to mean prediction and the apparent advantage of any architecture over a trivial baseline disappeared. The current value of 4.0, which produces an end-of-life `life_frac` near 0.75 by the final flight and an RUL distribution spanning roughly 25–100 across the fleet, is broad enough to make the regression non-trivial.

The calibration is honest but is also a candid acknowledgement that the synthesised UAV results are *configuration-conditional* in a way that the C-MAPSS results are not. C-MAPSS is fixed by NASA; the UAV synth is fixed by the author. A reader who reproduces the work under a different `END_OF_LIFE_HOURS` will obtain different headline numbers, and the relevant operational reading is that the synthesised fleet validates the *pipeline* — its ability to learn a non-trivial regression on UAV-shaped data — rather than establishes a transferable absolute benchmark. The C-MAPSS results are retained explicitly to provide that transferable benchmark.

## 7.3 Generalisation Gap

The C-MAPSS results, at 16.40 combined RMSE, are competitive with the mid-tier of the published literature; they do not match the very-low-teens numbers reported in frontier work that combines architectural innovations with per-subset hyperparameter tuning and ensembling. The present work foregoes both, in service of reproducibility under a single training driver and a single seed. A reader who interprets the headline number as a claim about the architectural ceiling rather than about the present pipeline's reproducibility would be over-reading the result.

The synthesised UAV results, at 3.45 flight-hour RMSE and 95.5 % fault accuracy, do not transfer directly to a real tactical-class multirotor for the reason given in §7.2. The fleet is synthesised, the four fault signatures are injected with known parameters, and the classification problem is by construction easier than its real-world analogue, in which fault signatures co-occur, drift over time, and interact with operational variability. The right reading of the UAV results is that they demonstrate the pipeline's capacity to learn a non-trivial joint regression-and-classification problem on UAV-shaped data, not that they establish a frontier on a real-world fleet.

## 7.4 Operational Implications

If the framework were to be operationalised on a real fleet, three operational observations arising from the present implementation would carry forward. First, the edge-served inference architecture, with the FastAPI process collocated with the operator's workstation, is consistent with the digital-twin literature's prescription against centralised cloud serving for real-time prognostics. Second, the static-artifact contract between the ML pipeline and the dashboard is robust to backend outage: even if the live inference path fails, the dashboard continues to serve the most recent baked artifacts, with the asset's status appropriately stale-flagged through the manifest's `data_source` field. Third, the explainability layer — attention maps plus Integrated Gradients — is a substantive contribution to operator trust. An operator confronted with a low predicted RUL can interrogate which timesteps and which sensors drove the prediction, which is the operationally useful sense of "explainable", and which is the sense in which existing PdM dashboards in the published literature are typically deficient.

## 7.5 Threats to Validity

Four threats to validity are explicitly acknowledged. First, the *single-seed* condition: results are reported at seed 1 337 only, and the variance across seeds is not characterised. A multi-seed evaluation would strengthen any claim about the architectural ranking, particularly the small RMSE gap between the LSTM and the Transformer on C-MAPSS, which is well within the typical seed-level variance in this literature. Second, the *synthesised-fleet* condition: as discussed in §7.3, the UAV results are conditional on a procedural model whose calibration is the author's. Third, the *attention-as-explanation* caveat: the literature on attention as a feature-importance proxy is mixed, and the framework's attention-based explanations should be understood as auditable traces of the model's processing rather than as causal accounts of the prediction. Fourth, the *operator-evaluation* gap: the dashboard has not been put in front of a working maintenance engineer for evaluation under a controlled task, and any claim about its operational utility is therefore a claim about its *plausibility* rather than its *demonstrated* utility.

## 7.6 Reflections on the Process

A reflective note on the process by which the work was conducted is worth recording. The single most consequential methodological lesson was the discovery, late in the practical phase, of the `END_OF_LIFE_HOURS` mis-calibration described in §7.2. The lesson is methodological rather than technical: it is easy, when constructing a synthesised dataset, to introduce a parameter that makes the problem trivially solvable without making the triviality visible in the headline metrics. The mitigation, retained in the final implementation, is to report results on both the synthesised fleet and on an external benchmark (C-MAPSS); a degradation in the external-benchmark numbers would have caught a similar mis-calibration earlier than the project's eventual discovery of it.

A second reflective note concerns the static-artifact contract described in §5.2. The contract was originally adopted for the pragmatic reason that the dashboard needed to render figures for the dissertation in the absence of a running backend. It became, over the course of the project, the architectural backbone of the system, in the sense that essentially every design decision downstream of it was simplified by its existence. If the project were to be repeated, the static-artifact contract would be designed earlier and more carefully, with the live-inference path fitted to it rather than the contract derived from the live path.

---

# 8. Conclusions and Future Work

## 8.1 Conclusions

This dissertation has presented ARES PdM, a digital-twin framework for predictive maintenance of tactical unmanned aerial vehicles, in which three deep-learning architectures — an LSTM, a 1D-CNN, and a Transformer encoder — are trained on a turbofan benchmark and a synthesised UAV fleet under a single configuration-driven training driver, and in which the resulting predictions are surfaced to an operator dashboard in both an offline-replay and a live-inference mode under a single static-artifact contract. The framework's principal substantive contribution is its integration of attention-map and Integrated Gradients explainability artifacts as primary visual elements of the operator's workflow, rather than as auxiliary post-hoc analysis. Its principal architectural contribution is the static-artifact contract that reconciles offline reproducibility with live serving without forcing the dashboard's component layer to branch on mode.

The Transformer architecture attains the strongest headline metrics on both datasets — 16.40 RMSE and a NASA Score of 4 074 on the C-MAPSS combined test set, and 3.45 flight-hour RMSE with 95.5 % fault-classification accuracy on the synthesised UAV fleet. The C-MAPSS results are competitive with, though not at the frontier of, the mid-tier published literature; the synthesised UAV results validate the pipeline rather than establish a transferable absolute benchmark. The qualitative outputs of the explainability layer are surfaced in the dashboard and admit operator-level interrogation of predictions, although a controlled operator study has not been conducted.

In direct answer to the research question posed in §1.2: a single, reproducible deep-learning pipeline can simultaneously deliver competitive Remaining Useful Life estimates on a turbofan benchmark and a synthetic UAV fleet, and can expose the basis of those estimates through attention and gradient-based attribution at a fidelity sufficient for plausible operator-level decision-making. The qualifying clauses to that answer — the single-seed condition, the synthesised-fleet caveat, the operator-evaluation gap — are documented in §7.5.

## 8.2 Future Work

Four extensions are identified as natural continuations of the present work. A *multi-seed evaluation* would replace the present single-seed numbers with means and confidence intervals across multiple seeds, strengthening claims about the architectural ranking, particularly the narrow Transformer–LSTM gap on C-MAPSS RMSE. A *real-fleet evaluation* would acquire run-to-failure data on a tactical multirotor and apply the present pipeline without modification, providing the transferable absolute benchmark that the synthesised fleet does not. A *quantitative live-inference characterisation* would profile end-to-end WebSocket latency and prediction throughput under controlled load. A *controlled operator-evaluation study* would put the dashboard in front of working maintenance engineers under a defined diagnostic task and measure the degree to which the explainability layer supports their workflow. None of these extensions has been undertaken in the present work; all four are identified here as candidates for subsequent investigation.

---

# 9. References

*Note to supervisor: references below are listed in author–year style for convenience and will be re-formatted to the institution's preferred style (Harvard or IEEE) before final submission. Entries marked † paraphrase the project bibliography and require DOI verification before submission.*

Al-Darraji, I., Piromalis, D., Kakei, A. A., Khan, F. Q., Stojmenovic, M., Tsaramirsis, G., and Papageorgas, P. G. (2021). A Technical Framework for Selection of Autonomous UAV Navigation Technologies and Sensors. *Computers, Materials & Continua*. †

Bondyra, A., Kołodziejczak, M., Kulikowski, R., and Giernacki, W. (2022). Real-Time Monitoring of Parameters and Diagnostics of the Technical Condition of Small Unmanned Aerial Vehicle's Units Based on Deep BiGRU-CNN Models. *Drones*. †

Costa, M., et al. (2026). A Systematic Literature Review of AI-Driven Predictive Maintenance in Aviation. *Applied Sciences*. †

Darrah, T., Quinones-Grueiro, M., Biswas, G., and Kulkarni, C. (2021). A Data Management Framework and UAV Simulation Testbed for the Study of System-level Prognostics Technologies. *Annual Conference of the Prognostics and Health Management Society*. †

Duan, Y., et al. (2026). A Physics-Informed Deep Neural Network Model for UAV Lithium-Ion Battery Remaining Useful Life Prediction Under Coupled Environmental Conditions. *Batteries*. †

Heimes, F. O. (2008). Recurrent Neural Networks for Remaining Useful Life Estimation. *International Conference on Prognostics and Health Management*.

Hochreiter, S., and Schmidhuber, J. (1997). Long Short-Term Memory. *Neural Computation*, 9(8), 1735–1780.

Kingma, D. P., and Ba, J. (2015). Adam: A Method for Stochastic Optimization. *International Conference on Learning Representations*.

Kokhlikyan, N., et al. (2020). Captum: A Unified and Generic Model Interpretability Library for PyTorch. *arXiv:2009.07896*.

Loshchilov, I., and Hutter, I. (2019). Decoupled Weight Decay Regularization. *International Conference on Learning Representations* (AdamW).

Loshchilov, I., and Hutter, I. (2017). SGDR: Stochastic Gradient Descent with Warm Restarts. *International Conference on Learning Representations* (cosine annealing).

Mehdipour, M., et al. (2026). Integrating Digital Twins, Data Management and Artificial Intelligence for Aviation Predictive Maintenance: A Systematic Scoping Review. *IET Digital Twins*. †

Murray, R. W. (2011). Skillful Writing of an Awful Research Paper. *Analytical Chemistry*, 83(3), 633.

Ozkat, E. C., et al. Data-Driven Predictive Maintenance Model to Estimate RUL in a Multi-Rotor UAS. *International Journal of Micro Air Vehicles*. †

Paszke, A., et al. (2019). PyTorch: An Imperative Style, High-Performance Deep Learning Library. *Advances in Neural Information Processing Systems*, 32.

Sai, S., et al. (2023). A Comprehensive Survey on Artificial Intelligence for Unmanned Aerial Vehicles. *IEEE Open Journal of Vehicular Technology*, 4. †

Saxena, A., Goebel, K., Simon, D., and Eklund, N. (2008). Damage Propagation Modeling for Aircraft Engine Run-to-Failure Simulation. *International Conference on Prognostics and Health Management* (C-MAPSS).

Sundararajan, M., Taly, A., and Yan, Q. (2017). Axiomatic Attribution for Deep Networks. *Proceedings of the 34th International Conference on Machine Learning*, ICML 2017.

Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., and Polosukhin, I. (2017). Attention Is All You Need. *Advances in Neural Information Processing Systems*, 30.

*[Approximately 20–40 additional references to be added across the PHM, deep-learning, digital-twin, and software-architecture literatures during final polish.]*

---

# 10. Appendices

## Appendix A — Repository and Reproduction Instructions

The full source repository is structured as described in §5.1 with subtrees `ml/`, `backend/`, and `app/`/`components/`/`lib/`. Reproduction from a clean checkout proceeds as in §5.6: clone, create a Python 3.13 virtual environment, install pinned dependencies from `requirements.txt`, run `make phase1`, install Node dependencies with `npm install`, start the dashboard with `npm run dev`, and start the backend with `make backend`. The Makefile targets `fetch`, `synth`, `train`, `train-uav`, `export`, `figures`, and `backend` are individually runnable for partial reproduction.

## Appendix B — UAV Synthesis Parameters

The procedural UAV fleet is parameterised by the constants in `ml/data/uav_synth.py` and `ml/config.py`: number of airframes (default 20, CLI flag `--n_drones`), flights per airframe (default 10, CLI flag `--flights_per_drone`), flight duration (uniform on 800–1 300 s), telemetry rate (10 Hz; `DT = 0.1`), end-of-life threshold (`END_OF_LIFE_HOURS = 4.0`), number of fault classes (4, named `healthy`, `bearing`, `esc_thermal`, `battery` in `config.UAV['fault_classes']`), per-airframe fault-class distribution (`p = [0.25, 0.30, 0.25, 0.20]`), seed (`SEED = 1337`). The twenty-one telemetry channels are listed in §4.1.2. The training-time UAV configuration sets sliding-window length to 50, RUL clip to 100, batch size to 256, and epochs to 30.

## Appendix C — Hyperparameter Tables

**Common to all architectures:** optimiser AdamW with weight decay 1×10⁻⁴; learning rate 1×10⁻³; cosine-annealed schedule over the full epoch budget; gradient clipping not applied; regression loss MSE; UAV fault loss cross-entropy with head weight 0.5; seed 1 337.

**LSTM (`ml/models/lstm.py`):** input projection: linear `input_dim → hidden`; recurrent body: 2-layer LSTM, hidden size 96, inter-layer dropout 0.25; regression head: Linear(96, 64) → ReLU → Dropout(0.25) → Linear(64, 1); fault head: Linear(96, n_classes).

**1D-CNN (`ml/models/cnn.py`):** body: 4 × [Conv1d(channels=64, kernel=5, padding=2) → BatchNorm1d → ReLU]; pool: global average over time; dropout 0.3; regression head: Linear(64, 64) → ReLU → Dropout(0.3) → Linear(64, 1); fault head: Linear(64, n_classes).

**Transformer (`ml/models/transformer.py`):** input projection: Linear(input_dim, d_model=96); positional encoding: fixed sinusoidal up to length 256; encoder layers: 2 × [MultiheadAttention(d_model=96, nhead=4) with averaged head weights → residual+LayerNorm → FFN(96→192→96, GELU) → residual+LayerNorm]; pool: mean over time; dropout 0.2; regression head: LayerNorm(96) → Linear(96, 64) → GELU → Dropout(0.2) → Linear(64, 1); fault head: Linear(96, n_classes).

**Per-dataset:** C-MAPSS — sequence length 30, RUL clip 125, batch size 512, epochs 40, 14 sensor channels retained. UAV — sequence length 50, RUL clip 100, batch size 256, epochs 30, all 21 synth channels retained.

## Appendix D — Dashboard Routes and Files

The Next.js routes are: `/` (landing), `/armory` (fleet inventory), `/mission` (live HUD), `/diagnostics` (digital-twin inspection), `/lab` (Model Lab). Principal source files referenced in this dissertation: `lib/api.ts` (static fetchers); `lib/live.ts` (WebSocket client); `lib/useLiveStream.ts` (React hook); `lib/store.ts` (Zustand store); `lib/telemetry.ts` (procedural client-side flight envelope); `lib/assets.ts` (asset and status types); `components/AppBootstrap.tsx` (manifest hydration); `components/3D/UAVModel.tsx` (procedural geometry); `components/Mission/Radar.tsx`, `TelemetryPanel.tsx`, `HealthBar.tsx`, `LiveChart.tsx`, `LiveControls.tsx`; `components/Diagnostics/WireframeView.tsx`, `AttentionHeatmap.tsx`, `MaintenanceReadout.tsx`; `components/Lab/ResultsTable.tsx`, `TrainingCurves.tsx`, `PerAssetOverlay.tsx`; `backend/main.py`, `inference.py`, `synth_stream.py`; `ml/train.py`, `export.py`, `xai.py`, `figures.py`, `config.py`; `ml/data/cmapss.py`, `ml/data/uav_synth.py`; `ml/models/{lstm,cnn,transformer}.py`.

## Appendix E — Word Count

Approximately 13 000 words excluding front matter, references, and appendices.

---

*End of first draft.*
