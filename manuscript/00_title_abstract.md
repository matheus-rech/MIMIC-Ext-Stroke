# Title

**Stroke Digital Twins via Hybrid Bayesian-GAN Generative Models: Synthetic Patient Profiles and ICU Time-Series from MIMIC-IV**

# Abstract

**Background.** Stroke remains a leading cause of death and disability worldwide, yet developing and validating clinical decision-support tools is constrained by limited data access, patient privacy concerns, and the inability to perform counterfactual experiments on real patients. Digital twin frameworks that produce high-fidelity synthetic patient replicas offer a promising path to overcome these barriers, but existing generative approaches rarely combine static clinical profiles with realistic temporal ICU physiology for cerebrovascular disease.

**Objective.** To develop and evaluate a hybrid generative framework that creates stroke digital twins comprising both static clinical profiles and 72-hour ICU time-series, enabling privacy-preserving data sharing and individual-level counterfactual simulation.

**Methods.** We identified 8,500 stroke admissions (4,398 ischemic, 1,397 intracerebral hemorrhage, 436 subarachnoid hemorrhage, 282 transient ischemic attack, 1,987 other cerebrovascular) from MIMIC-IV v3.1. Static patient profiles (demographics, comorbidities, severity scores, outcomes) were modeled with a Bayesian Network learned via Hill Climbing with BIC scoring (pgmpy), yielding 31 directed edges encoding conditional dependencies. Temporal trajectories across 11 physiological channels (heart rate, systolic/diastolic/mean arterial pressure, respiratory rate, SpO2, temperature, GCS eye/verbal/motor/total) over 72 ICU hours were generated using a custom DoppelGANger architecture implemented as an LSTM-GAN in PyTorch. We benchmarked against CTGAN and TVAE baselines. A 12-metric evaluation framework assessed fidelity (Kolmogorov-Smirnov tests, correlation matrix preservation, discriminator AUC), utility (train-on-synthetic-test-on-real mortality prediction), privacy (membership inference attack F1, distance to closest record, attribute inference accuracy), temporal quality (dynamic time warping, autocorrelation preservation), and clinical plausibility through domain-specific physiological rules. Ten independent synthetic datasets were generated and combined using Rubin's rules.

**Results.** The synthetic cohort preserved key demographic and clinical distributions of the source population (median age 68 years [IQR 57--78], 50.1% female, 14.8% in-hospital mortality). The Bayesian Network maintained inter-variable correlations with low KS divergence across static features. The DoppelGANger model produced temporal trajectories with superior dynamic time warping alignment and autocorrelation fidelity compared with CTGAN and TVAE baselines. Train-on-synthetic-test-on-real mortality classifiers approached performance parity with models trained on real data. Privacy metrics confirmed low re-identification risk, with membership inference attack F1 scores near chance level and adequate distance to closest real records.

**Conclusions.** This hybrid Bayesian Network and LSTM-GAN framework generates realistic, privacy-preserving stroke digital twins that capture both static clinical complexity and dynamic ICU physiology. The counterfactual simulation capability -- modifying patient profiles (e.g., adding atrial fibrillation, altering age, introducing chronic kidney disease with diabetes) and regenerating corresponding ICU trajectories -- opens new avenues for personalized treatment effect estimation and in silico clinical trials in cerebrovascular disease.

# Keywords

Digital Twin; Stroke; Synthetic Data; Generative Adversarial Network; Bayesian Network; Intensive Care Units; Electronic Health Records; MIMIC
