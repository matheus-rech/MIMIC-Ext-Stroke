# 2. Methods

## 2.1 Study Design and Data Source

We conducted a retrospective cohort study using the Medical Information Mart for Intensive Care IV (MIMIC-IV) version 3.1, a publicly available critical care database maintained by the Massachusetts Institute of Technology and hosted on PhysioNet [1,2]. MIMIC-IV encompasses de-identified health records from patients admitted to Beth Israel Deaconess Medical Center (Boston, Massachusetts, USA) between 2008 and 2022, including demographics, diagnoses, laboratory results, vital signs, medications, and administrative data. Access to MIMIC-IV was obtained through PhysioNet credentialed data use agreement, and the study was approved by the institutional review board of Beth Israel Deaconess Medical Center, which granted a waiver of informed consent given the retrospective, de-identified nature of the data.

All analytical queries were executed using DuckDB (version 1.1), a columnar analytical database engine capable of querying compressed CSV files directly without requiring a dedicated database server. This approach facilitated reproducible, serverless extraction pipelines operating on the native MIMIC-IV file distribution.

## 2.2 Cohort Selection

We identified adult patients (aged 18 years or older) with a stroke-related diagnosis during an admission that included at least one intensive care unit (ICU) stay. Stroke diagnoses were ascertained from the hospital discharge coding tables using International Classification of Diseases, Ninth Revision (ICD-9) codes 433 through 436 and Tenth Revision (ICD-10) codes I60 through I67 and G45. These code ranges encompass ischemic stroke, intracerebral hemorrhage (ICH), subarachnoid hemorrhage (SAH), transient ischemic attack (TIA), and other cerebrovascular conditions.

ICU stays were filtered to retain only those with a length of stay (LOS) of at least 6 hours and no more than 30 days. The minimum threshold excluded brief observational encounters unlikely to reflect meaningful ICU-level care, while the maximum threshold excluded extreme outliers that may represent coding anomalies or prolonged custodial admissions. For patients with multiple qualifying ICU stays, only the first admission was retained to ensure statistical independence of observations.

Stroke subtype classification was derived from the primary ICD code assigned at discharge. Ischemic stroke was defined by ICD-10 codes I63.x or ICD-9 codes 433.x--434.x; intracerebral hemorrhage by I61.x; subarachnoid hemorrhage by I60.x; transient ischemic attack by G45.x or 435.x; and all remaining cerebrovascular codes were grouped as other cerebrovascular disease. The final analytic cohort comprised 8,500 unique patients.

## 2.3 Feature Extraction

Feature extraction proceeded along two axes: static clinical attributes summarizing each patient's admission profile, and time-series physiological measurements capturing temporal ICU trajectories.

### 2.3.1 Static Features

A total of 17 static variables were extracted per patient, organized into four domains.

**Demographics** included age at ICU admission (continuous, years), gender (binary), self-reported race (categorical), insurance type (categorical), and admission type (categorical: elective, emergency, urgent, or other).

**Comorbidities** were ascertained from ICD diagnostic codes recorded during the index hospitalization. Six comorbidity indicators were constructed as binary flags: hypertension (ICD-10 I10, ICD-9 401), diabetes mellitus (ICD-10 E11, ICD-9 250), atrial fibrillation (ICD-10 I48, ICD-9 427.3), dyslipidemia (ICD-10 E78, ICD-9 272), chronic kidney disease (ICD-10 N18, ICD-9 585), and coronary artery disease (ICD-10 I21--I25, ICD-9 410--412).

**Admission characteristics** comprised stroke subtype (as classified above), ICU length of stay (continuous, days), in-hospital mortality (binary), and first care unit (categorical, indicating the initial ICU to which the patient was admitted).

**First-24-hour laboratory values** were extracted as the first recorded result within 24 hours of ICU admission for six analytes: blood glucose, serum sodium, serum creatinine, hemoglobin, platelet count, and international normalized ratio (INR). These analytes were selected for their established prognostic relevance in acute stroke and critical care risk stratification.

### 2.3.2 Time-Series Features

Eleven physiological channels were extracted from the MIMIC-IV chartevents table over a 72-hour observation window beginning at ICU admission. Vital signs included heart rate (itemid 220045), systolic blood pressure (220050), diastolic blood pressure (220051), mean arterial pressure (220052), respiratory rate (220210), peripheral oxygen saturation (220277), and body temperature (223762). Notably, itemids 220050--220052 correspond exclusively to arterial line (invasive) blood pressure measurements; non-invasive blood pressure (NIBP) was not included. Because arterial lines are not universally placed, this design choice contributes to an approximately 58% missingness rate for the three blood pressure channels and may introduce selection bias toward more severely ill patients who receive invasive hemodynamic monitoring. Neurological status was captured through the three Glasgow Coma Scale (GCS) component scores -- eye opening (220739), verbal response (223900), and motor response (223901) -- along with a computed total GCS score (sum of the three components).

Raw measurements were aggregated into hourly time bins using the median value within each bin, producing a matrix of dimensions 72 (hours) by 11 (channels) per patient. Forward-fill imputation was applied within each ICU stay to propagate the most recent valid observation into subsequent empty bins, reflecting the clinical convention that the last documented value remains the best available estimate of a patient's physiological state until a new measurement is recorded.

## 2.4 Preprocessing

### 2.4.1 Encoding and Imputation

Categorical variables (gender, race, insurance type, stroke subtype, first care unit, and admission type) were encoded using one-hot representation. Continuous static variables with missing values were imputed using the column-wise median of the training set, and a binary missingness indicator flag was appended for each imputed variable to preserve information about the pattern of missing data.

### 2.4.2 Normalization

All continuous features were normalized to the range [--1, 1] using min-max scaling. Normalization parameters (feature-wise minimum and maximum values) were computed exclusively on the training partition and subsequently applied to the validation and test partitions, thereby preventing information leakage from held-out data into the preprocessing pipeline.

### 2.4.3 Data Partitioning

The cohort was divided into training (60%), validation (10%), and test (30%) partitions using stratified random splitting with in-hospital mortality as the stratification variable. This allocation ensured that the mortality rate was preserved across all partitions and that a sufficiently large test set was available for robust evaluation of generative model outputs.

### 2.4.4 Time-Series Preprocessing

Time-series data underwent forward-fill imputation as described in Section 2.3.2, followed by physiological range clipping to constrain values within clinically plausible bounds. Clipping thresholds were defined as follows: heart rate 20--300 beats per minute, systolic blood pressure 30--300 mmHg, diastolic blood pressure 10--200 mmHg, mean arterial pressure 20--250 mmHg, respiratory rate 2--80 breaths per minute, peripheral oxygen saturation 50--100%, body temperature 30--45 degrees Celsius, and GCS component scores within their respective valid ranges (eye 1--4, verbal 1--5, motor 1--6, total 3--15). Values falling outside these bounds were truncated to the nearest threshold, mitigating the influence of artefactual or erroneous charted entries.

## 2.5 Bayesian Network for Static Patient Profiles

A Bayesian Network (BN) was employed to model the joint probability distribution over static patient features, thereby capturing the multivariate dependency structure among demographics, comorbidities, admission characteristics, laboratory values, and outcomes.

**Structure learning** was performed using the Hill Climbing algorithm with the Bayesian Information Criterion (BIC) as the scoring function, implemented in pgmpy (version 1.0.0) [3]. The maximum in-degree was constrained to three parent nodes per variable to limit model complexity and reduce the risk of overfitting, while preserving sufficient flexibility to represent clinically meaningful conditional dependencies. Prior to structure learning, continuous variables were discretized using variable-specific strategies: age was binned into six expert-defined clinically meaningful categories (18--45, 45--55, 55--65, 65--75, 75--85, 85+ years), ICU length of stay into five expert-defined categories (0--1, 1--3, 3--7, 7--14, 14+ days), and laboratory values into quartile-based bins (four bins per analyte, with boundaries computed from the training data).

**Parameter estimation** employed the Bayesian Estimator with a Bayesian Dirichlet equivalent uniform (BDeu) prior and an equivalent sample size of 10. The BDeu prior provides a uniform distribution over parameter configurations conditional on the learned graph structure, offering a principled smoothing mechanism that is particularly valuable for rare category combinations.

The learned network comprised 31 directed edges encoding conditional dependencies among clinical variables. Notable learned relationships included edges from stroke subtype to in-hospital mortality, atrial fibrillation to stroke subtype, hypertension to chronic kidney disease, age to comorbidity burden, and diabetes to creatinine level -- each reflecting well-established clinical knowledge and thereby providing face validity for the learned structure.

**Synthetic profile generation** was performed using pgmpy's forward sampling algorithm (simulate function), which traverses the topological ordering of the directed acyclic graph and samples each variable from its conditional probability table given the sampled values of its parents. Following sampling, discretized variables were mapped back to continuous values through inverse discretization, sampling uniformly within each bin's range to preserve realistic continuous distributions while respecting the learned conditional dependency structure.

## 2.6 DoppelGANger for ICU Time-Series

Temporal physiological trajectories were generated using a custom DoppelGANger architecture [4] implemented as a long short-term memory (LSTM)-based generative adversarial network (GAN) in PyTorch (version 2.4).

### 2.6.1 Generator Architecture

The generator accepted two inputs: a metadata vector encoding the static patient profile and a noise vector sampled from a standard normal distribution. These vectors were concatenated and passed through a fully connected linear layer to produce an initial hidden representation, which was then fed into a two-layer LSTM with a hidden dimensionality of 128 units per layer. The LSTM output at each of the 72 time steps was projected through a final linear layer followed by a hyperbolic tangent (tanh) activation function, yielding a synthetic time-series of dimensions 72 (hours) by 11 (channels) with values bounded in [--1, 1].

### 2.6.2 Discriminator Architecture

The discriminator comprised an LSTM branch processing the sequential input and a parallel linear branch processing the metadata vector. The LSTM branch consumed the 72-step time-series and produced a summary representation from its final hidden state. The linear branch independently encoded the metadata. Both representations were concatenated and passed through fully connected layers to produce a single scalar logit indicating the probability that the input pair (time-series, metadata) was drawn from the real data distribution.

### 2.6.3 Training Procedure

The GAN was trained using binary cross-entropy with logits loss, with the Adam optimizer configured with a learning rate of 0.0002 and momentum parameters (beta1, beta2) of (0.5, 0.999). Training proceeded for 5,000 epochs (as specified in the project configuration; the code default of 500 epochs was overridden to ensure convergence on the full cohort). Gradient penalty or spectral normalization was not employed, as the LSTM-based architecture and moderate-dimensional output space provided sufficient training stability in our experiments. Training was accelerated using the Apple Silicon Metal Performance Shaders (MPS) backend.

### 2.6.4 Conditioning Mechanism

To ensure coherence between static profiles and temporal trajectories, the BN-generated static patient profiles were normalized using the same scaling parameters as the training data and concatenated with the noise vector prior to input to the generator. This conditioning mechanism enabled the generator to learn associations between patient characteristics (e.g., stroke subtype, age, comorbidity profile) and the corresponding expected patterns in ICU physiological time-series.

## 2.7 Hybrid Generation Pipeline

The complete synthetic patient generation pipeline proceeded in four sequential stages. First, the trained Bayesian Network generated N static patient profiles via forward sampling. Second, each static profile was encoded into a fixed-length metadata vector through the same one-hot encoding and normalization transformations applied to the real training data. Third, the trained DoppelGANger generator produced a 72-hour, 11-channel time-series for each static profile, conditioned on the corresponding metadata vector. Fourth, the static profile and temporal trajectory were combined into a single synthetic patient record.

Following the recommendations of El Emam et al. (2024) [5] for the evaluation of synthetic data generators, we generated at least 10 independent synthetic datasets of the same size as the training set. This replication strategy enables estimation of between-synthesis variability and application of Rubin's combining rules for pooled statistical inference (see Section 2.11).

## 2.8 Baseline Comparisons

Two established tabular generative models served as baselines for the static feature component: the Conditional Tabular GAN (CTGAN) and the Tabular Variational Autoencoder (TVAE), both proposed by Xu et al. (2019) [6] and implemented through the Synthetic Data Vault (SDV) library (version 1.34). CTGAN employs a conditional generator with mode-specific normalization to handle mixed data types, while TVAE adapts variational autoencoders with evidence lower bound loss for tabular data. Both models were trained on the identical feature set used for the Bayesian Network to ensure a fair comparison. These baselines address static features only; no temporal baselines were compared, as CTGAN and TVAE do not natively support sequential data generation.

## 2.9 Evaluation Framework

We employed a comprehensive 12-metric evaluation framework organized into five domains: fidelity, clinical plausibility, utility, privacy, and temporal fidelity.

### 2.9.1 Fidelity

Four metrics assessed how closely synthetic data matched the statistical properties of the real data.

**Dimension-wise distributional similarity** was quantified using the two-sample Kolmogorov-Smirnov (KS) test applied independently to each continuous column, yielding a per-feature D-statistic and associated p-value. Lower D-statistics indicate closer distributional agreement.

**Correlation preservation** was measured as the Frobenius norm of the difference between the Pearson correlation matrices of the real and synthetic datasets. A smaller Frobenius distance indicates superior preservation of inter-variable linear dependencies.

**Discriminator score** was computed as the cross-validated area under the receiver operating characteristic curve (AUC) of a logistic regression classifier trained to distinguish real from synthetic records. An AUC approaching 0.5 indicates that the classifier cannot reliably differentiate between the two datasets, signifying high distributional fidelity.

**Medical concept abundance** was assessed using the Manhattan distance (L1 norm) between the categorical frequency distributions of the real and synthetic datasets. This metric captures whether the relative prevalence of clinical categories (e.g., stroke subtypes, comorbidity combinations) is preserved in the synthetic output.

### 2.9.2 Clinical Plausibility

Seven domain-specific physiological rules were applied as hard constraints to evaluate whether synthetic records fell within clinically valid ranges: age between 18 and 120 years; GCS total between 3 and 15; ICU length of stay greater than 0 days; systolic blood pressure strictly greater than diastolic blood pressure; peripheral oxygen saturation between 50% and 100%; heart rate between 20 and 300 beats per minute; and body temperature between 30 and 45 degrees Celsius. The proportion of synthetic records satisfying all rules simultaneously was reported as the clinical plausibility rate.

### 2.9.3 Utility

Downstream utility was assessed through the Train on Synthetic, Test on Real (TSTR) paradigm, wherein predictive models were trained exclusively on synthetic data and evaluated on the held-out real test set. Performance was compared against a Train on Real, Test on Real (TRTR) reference in which models were trained on the real training set. The prediction task was in-hospital mortality classification. Two classifiers were evaluated: logistic regression and random forest, both implemented in scikit-learn (version 1.5) with default hyperparameters. The primary metric was the area under the receiver operating characteristic curve (AUROC), and the utility gap was defined as the absolute difference between TRTR and TSTR AUROC values. A smaller gap indicates that synthetic data preserves the discriminative signal necessary for downstream clinical prediction.

### 2.9.4 Privacy

Three metrics quantified the risk of re-identification and information leakage from synthetic data back to individual real patients.

**Membership Inference Attack (MIA)** was implemented as a nearest-neighbor distance-based classifier that attempts to determine whether a given synthetic record was generated from a real record present in the training set versus a record not seen during training. Attack performance was quantified as the F1 score, with values below 0.20 considered indicative of adequate privacy protection.

**Distance to Closest Record (DCR)** was computed as the Euclidean distance between each synthetic record and its nearest neighbor in the real training set. The mean, median, and 5th percentile of the DCR distribution were reported, with higher values indicating greater privacy preservation and lower risk of producing near-duplicates of real patients.

**Attribute Inference Attack (AIA)** assessed whether knowledge of a subset of synthetic features could be used to predict a withheld sensitive attribute of the corresponding real record. A random forest classifier was trained to infer the target attribute, and prediction accuracy was reported. Accuracy close to the marginal base rate indicates minimal information leakage.

### 2.9.5 Temporal Fidelity

Two metrics specifically evaluated the quality of generated time-series data.

**Dynamic time warping (DTW) distance** was computed between pairs of real and synthetic sequences for each physiological channel, producing a distance matrix summarizing temporal alignment quality. Lower DTW distances indicate that synthetic trajectories exhibit temporal dynamics (trends, oscillations, transient events) consistent with real ICU recordings.

**Autocorrelation function comparison** was performed by computing the sample autocorrelation function for each vital sign channel in both real and synthetic datasets and measuring the mean absolute difference across lags. Preservation of autocorrelation structure confirms that the generative model captures the temporal persistence and periodicity inherent in ICU physiological signals.

## 2.10 Counterfactual Simulation (Digital Twin Application)

A central motivation for the hybrid generative framework is the capacity to perform counterfactual reasoning at the individual patient level -- a defining characteristic of the digital twin paradigm. Counterfactual simulations were conducted by modifying one or more attributes of a given patient's static profile (e.g., introducing atrial fibrillation, increasing age by a decade, adding chronic kidney disease in combination with diabetes mellitus) and regenerating the corresponding 72-hour ICU time-series trajectories conditioned on the modified profile.

For each counterfactual scenario, 20 to 30 synthetic time-series samples were generated to characterize the distribution of plausible trajectories under the modified clinical context. Factual trajectories (conditioned on the original, unmodified profile) were generated in parallel using the same number of samples. Comparisons between factual and counterfactual trajectory distributions were expressed as the difference in mean trajectories with accompanying standard deviations, enabling visual and quantitative assessment of the expected physiological impact of the simulated clinical modification.

Individual Treatment Effect (ITE) estimation was derived from the difference in summary statistics (e.g., mean heart rate, mean arterial pressure, GCS trajectory slope) between factual and counterfactual trajectory distributions. While these estimates reflect learned associations rather than causal effects -- a critical distinction -- they nonetheless provide a principled framework for hypothesis generation regarding how patient-level characteristics modulate expected ICU physiological trajectories.

## 2.11 Statistical Analysis

Continuous variables were summarized as median with interquartile range (IQR), and group comparisons were performed using the Kruskal-Wallis test for non-normally distributed data. Categorical variables were presented as counts with percentages, and comparisons were conducted using the chi-squared test. When pooling estimates across the 10 independently generated synthetic datasets, Rubin's combining rules were applied to obtain pooled point estimates and confidence intervals that account for both within-synthesis and between-synthesis variability [7]. The significance threshold for all hypothesis tests was set at p < 0.05 (two-sided).

All analyses were performed in Python 3.12. Key software dependencies included DuckDB 1.1 for data extraction, pgmpy 1.0.0 for Bayesian Network construction and sampling, PyTorch 2.4 for DoppelGANger implementation and training, SDV 1.34 for CTGAN and TVAE baselines, and scikit-learn 1.5 for evaluation classifiers. Code and configuration files will be made available upon publication at a public repository.
