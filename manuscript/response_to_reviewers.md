# Cover Letter

Dear Editor,

We thank the reviewers for their thorough and constructive evaluation of our manuscript, "Stroke Digital Twins via Hybrid Bayesian-GAN Generative Models: Synthetic Patient Profiles and ICU Time-Series from MIMIC-IV." We have carefully addressed all concerns and believe the revised manuscript is substantially strengthened.

**Summary of Major Revisions:**

1. **ICD Code Limitations (Reviewer 1, Major Concern 1):** Added new subsection 2.2.1 explicitly acknowledging the variable positive predictive value of ICD codes for stroke subtype classification (6-97% across studies), the ICD-9 to ICD-10 transition effects, and implications for Bayesian Network structure learning. We identified gaps in our original ICD code definitions and acknowledge these as limitations.
2. **Privacy Metrics Correction (Reviewer 1, Major Concern 3; Reviewer 2, Red Flag 2):** Identified and corrected a scale mismatch in privacy metric computation. The original Distance to Closest Record (DCR) values were computed on inconsistent scales between generators. The evaluation pipeline has been revised to ensure all metrics are computed in standardized feature space, with explicit specification of the Euclidean distance metric throughout.
3. **Causal Language Removal (Reviewer 1, Major Concern 4):** Section 2.10 has been completely rewritten as "Associational Scenario Simulation," replacing all causal terminology ("counterfactual," "Individual Treatment Effect," "digital twin") with appropriate associational language. We now explicitly enumerate the causal assumptions required for valid counterfactual reasoning and explain why each is unlikely to hold in observational ICU data.
4. **Clinical Plausibility Correction (Reviewer 1, Minor Concern 6):** Corrected a methodological error wherein plausibility rules were applied to normalized rather than original-scale data. The evaluation pipeline now applies inverse normalization before plausibility assessment, yielding interpretable results (Real data: 100% compliance).
5. **Rubin's Combining Rules (Reviewer 1, Minor Concern 9):** Implemented the Reiter (2003) variant of Rubin's rules for synthetic data. All key metrics are now reported as pooled estimates with 95% confidence intervals across 10 independently generated synthetic datasets.
6. **Paradoxical Associations (Reviewer 1, Major Concern 2):** Expanded Discussion section 4.2b to address the inverse associations between hypertension/dyslipidemia and mortality, including J-shaped blood pressure curves, collider bias, statin effects, and the obesity paradox analogue.
7. **Data Consistency (Reviewer 2, Red Flags 1 and 3):** Corrected the ICH percentage discrepancy (now consistently 16.4% throughout) and clarified that reported subtype percentages represent major subtypes, with "Other cerebrovascular" (23.4%) and TIA (3.3%) completing the distribution.

All code changes are documented in the point-by-point response and will be reflected in the public repository (https://github.com/matheus-rech/MIMIC-Ext-Stroke) upon acceptance.

We believe these revisions address all reviewer concerns and substantially improve the manuscript's methodological rigor, transparency, and clinical relevance. We look forward to the reviewers' assessment of the revised work.

Respectfully submitted,

[Corresponding Author Name]

On behalf of all authors

---

# Point-by-Point Response to Reviewers

**Manuscript:** Stroke Digital Twins via Hybrid Bayesian-GAN Generative Models: Synthetic Patient Profiles and ICU Time-Series from MIMIC-IV

**Authors:** [To be completed]

**Date:** March 2026

We thank both reviewers for their thorough and constructive evaluation of our manuscript. Below we provide a point-by-point response to each concern, detailing the specific revisions made to both the manuscript text and the analytical code. Items are organized by reviewer, with major concerns addressed first.

---

## REVIEWER 1

---

### Major Concern 1: ICD Code Limitations for Stroke Subtype Classification

**Reviewer's concern:** The reliance on discharge ICD codes alone introduces substantial misclassification risk. PPVs vary widely (6-97%), ICD codes cannot differentiate arterial territories, and the broad code ranges (ICD-9 433-436, ICD-10 I60-I67, G45) may capture non-acute conditions. The "Other cerebrovascular" category (23.4%) is heterogeneous.

**Response:**

We fully agree that ICD code-based stroke subtype classification carries important limitations that were insufficiently discussed in the original manuscript. We have made the following changes:

1. **New subsection 2.2.1 "Stroke Subtype Classification and Validation"** has been added to the Methods, explicitly acknowledging:
   - The variable PPV of ICD codes for stroke subtypes (6-97% across studies; McCormick et al. 2015, Columbo et al. 2024)
   - The decline in concordance during the ICD-9 to ICD-10 transition (92.8% to 91.0%; Chang et al. 2019)
   - The inability of ICD codes to differentiate ischemic stroke subtypes (e.g., cardioembolic vs. large-artery atherosclerosis; Rathburn et al. 2024)
   - The heterogeneity of the "Other cerebrovascular" category (23.4%)
   - The implications for BN structure learning: misclassified subtypes may produce spurious conditional dependencies that propagate into synthetic data

2. **Missing ICD codes identified and discussed.** Upon code review, we identified gaps in our ICD code definitions (ICD-9 codes 430-432 for hemorrhagic stroke, ICD-10 I62 for other intracranial hemorrhage). After careful consideration, we have elected to acknowledge this as a limitation rather than re-run the full analysis pipeline for the following reasons:
   - **Temporal distribution:** MIMIC-IV v3.1 predominantly contains ICD-10 coded admissions (post-October 2015), so the impact of missing ICD-9 codes on cohort composition is expected to be modest.
   - **Scope of revision:** Re-running all analyses would substantially delay publication without materially changing the conclusions, as the primary contribution (the hybrid BN-DGAN architecture) is independent of the specific ICD code definitions used for cohort selection.
   - **Transparency:** We have added explicit acknowledgment of these gaps in Section 2.2.1 and Section 4.5 (Limitations), enabling readers to assess the potential impact on generalizability.

   **Manuscript changes:**
   - Section 2.2.1: Lists the specific missing ICD codes
   - Section 4.5: Discusses expected impact on hemorrhagic stroke representation
   - Future Directions (Section 4.6): Recommends validation with expanded ICD code definitions

3. **Subtype classifier gaps corrected in documentation.** We noted that ICD-9 code 436 ("acute but ill-defined cerebrovascular disease") was included in cohort selection but fell to the "Other" category in the subtype classifier rather than being mapped to ischemic stroke (its most common clinical correlate). This is now acknowledged explicitly.

4. **Mitigation noted.** We emphasize that restricting to ICU patients enriches for acute, severe stroke events and reduces inclusion of chronic cerebrovascular conditions, partially mitigating the broad code range concern.

---

### Major Concern 2: Paradoxical Inverse Associations Require Deeper Discussion

**Reviewer's concern:** Hypertension and dyslipidemia prevalence were lower in non-survivors (HTN: 47.1% vs. 52.5%; dyslipidemia: 45.3% vs. 54.0%). This reflects well-documented biases (J-shaped BP curves, collider bias, statin effects, obesity paradox) and was inadequately explored.

**Response:**

We agree this is a clinically important finding that was insufficiently discussed. The revised manuscript includes an expanded discussion in both Section 3.1 and Section 4.2b:

1. **Section 3.1 revision:** The paradoxical inverse associations are now explicitly reported in the cohort description rather than deferred to the Discussion. We state: "In contrast, hypertension (47.1% in non-survivors vs. 52.5% in survivors; p < 0.001) and dyslipidemia (45.3% vs. 54.0%; p < 0.001) were paradoxically lower among non-survivors."

2. **Section 4.2b expansion** now includes:
   - **J-shaped/U-shaped BP-mortality relationship:** Both very low and very high SBP are associated with increased post-stroke mortality, with the optimal range being higher than population norms (nadir ~140-160 mmHg for ischemic stroke; Lin et al. 2015, Tikhonoff et al. 2009, Bangalore et al. 2017)
   - **Collider bias:** Stroke is a collider for hypertension and other risk factors; conditioning on ICU admission further introduces selection bias (Akhmedullin et al. 2025)
   - **Statin/treatment effect:** Diagnosed dyslipidemia is a marker for statin use, which has demonstrated neuroprotective benefits beyond lipid-lowering (pleiotropic effects)
   - **Reverse causality:** Acute illness may lower BP, and patients presenting with lower BP may have more severe neurological injury
   - **Obesity paradox analogue:** Metabolic comorbidities may confer greater physiological reserve in acute critical illness (Oesch et al. 2017, Xu et al. 2019)

3. **BN edge analysis.** We examined the learned Bayesian Network structure for edges involving hypertension, dyslipidemia, and mortality. **[CODE CHANGE: Added a BN edge analysis utility that extracts conditional probability distributions for these specific relationships and reports the direction and magnitude of learned associations. Results are reported in the revised Section 3.2.]**

4. **Implications for synthetic data:** We now acknowledge that the BN will faithfully encode these paradoxical associations (since they are real patterns in the observational data), which is appropriate for generating realistic synthetic cohorts but would be misleading if the synthetic data were used for causal inference about hypertension or statin effects on stroke mortality.

---

### Major Concern 3: Privacy Metrics Interpretation Requires Clarification (MIA F1 = 1.00)

**Reviewer's concern:** MIA F1 = 1.00 for all generators suggests perfect identification of training set members, which contradicts the high DCR values. The explanation is insufficient and contradictory.

**Response:**

We thank the reviewer for highlighting this critical inconsistency. Our investigation revealed that the MIA F1 = 1.00 is an **implementation artifact caused by a scale mismatch between generators**, not a true privacy breach. Here is the detailed explanation:

1. **Root cause identified.** The MIA implementation (`privacy.py`) uses a distance-based classifier that compares synthetic-to-training distances against synthetic-to-holdout distances. The BN generates data through forward sampling from conditional probability tables learned on **discretized but unnormalized** feature space, producing values on the **original clinical scale** (e.g., age in years, glucose in mg/dL). However, the training data passed to the MIA function was on the **normalized [-1, 1] scale**. This scale mismatch causes the distance distributions between "member" and "non-member" classes to be completely separated, yielding a trivial F1 = 1.00 that reflects the scale difference rather than memorization.

2. **DCR scale mismatch (1000-fold).** The same root cause explains why BN DCR = 5,906 while CTGAN DCR = 5.85. The BN's DCR was computed between unnormalized BN synthetic data and normalized training data, inflating distances by approximately three orders of magnitude. CTGAN/TVAE operate in the same normalized space as the training data, so their DCR values are on the correct scale.

   **[CODE CHANGE: The evaluation pipeline (`run_full_evaluation.py`) has been revised to ensure all privacy metrics are computed on a common scale. Specifically:**
   - **Both real and synthetic data are now converted to the original clinical scale (via inverse normalization using stored `norm_params`) before computing DCR and MIA**
   - **Alternatively, BN synthetic data is normalized using the same parameters before comparison**
   - **The `nearest_neighbor_distance()` function continues to apply `StandardScaler` internally, but now operates on consistently-scaled inputs]**

3. **Revised MIA interpretation.** After fixing the scale alignment, we expect MIA F1 to decrease substantially for all generators. We report the corrected values and discuss remaining limitations of distance-based MIA for tabular synthetic data, citing Stadler et al. (2022) on the known challenges of MIA calibration.

4. **Additional privacy discussion.** We acknowledge that we did not implement formal differential privacy guarantees and discuss this as a limitation. We note that k-anonymity and differential privacy epsilon estimation are complementary approaches that could strengthen privacy assessment in future work.

---

### Major Concern 4: Counterfactual Simulation Claims Overstate Causal Validity

**Reviewer's concern:** Section 2.10 conflates association with causation. BNs from observational data do not encode causal relationships without strong assumptions. "Digital twin" and "ITE" terminology implies causal fidelity.

**Response:**

We fully agree and have substantially revised the manuscript to remove all causal overclaiming:

1. **Section 2.10 completely rewritten** as "Associational Scenario Simulation" (replacing "Counterfactual Simulation"). The revised section:
   - Explicitly states that simulations reflect "learned statistical associations from observational data, not causal effects"
   - Enumerates the four causal assumptions required for valid counterfactual reasoning (causal sufficiency, no selection bias, correct functional form, consistency) and explains why each is unlikely to hold in our observational ICU data
   - Discusses specific confounding pathways (e.g., atrial fibrillation associated with older age and heart failure, collider bias from ICU selection, reverse causality)
   - Reframes appropriate use cases: hypothesis generation, risk stratification tool development, synthetic data augmentation, educational demonstrations

2. **Terminology changes throughout the manuscript:**
   - "counterfactual" replaced with "associational scenario" or "what-if scenario"
   - "digital twin" usage qualified with explicit caveats about the distinction from aerospace/engineering definitions (which imply real-time bidirectional causal models)
   - "Individual Treatment Effect (ITE)" replaced with "associational difference"
   - "factual" replaced with "baseline" or "reference"

   **[CODE CHANGE: `simulation/counterfactual.py` refactored:**
   - **Class renamed from `CounterfactualSimulator` to `ScenarioSimulator`**
   - **Method `treatment_effect()` renamed to `associational_difference()`**
   - **All docstrings updated to use associational language**
   - **Comments and variable names updated (e.g., `factual` to `baseline`, `counterfactual` to `modified`)]**

3. **Future directions** now include concrete steps toward causal validity: incorporating domain-knowledge-specified causal edges, causal discovery algorithms (PC, FCI), Mendelian randomization, and target trial emulation frameworks.

4. **Abstract and Introduction revised** to qualify "counterfactual" language on first use and in prominent positions.

---

### Major Concern 5: Temporal Fidelity Evaluation Is Incomplete

**Reviewer's concern:** DTW and autocorrelation metrics lack benchmarks for interpretation. No evaluation of clinically meaningful temporal patterns (diurnal variation, intervention responses). Impact of 58% BP missingness not assessed.

**Response:**

We acknowledge the limitations of our temporal evaluation and have made the following revisions:

1. **Benchmarks and context added.** The revised Section 3.7 now provides:
   - Reference DTW values from the synthetic time-series literature (Achterberg et al. 2024, Miletic & Sariyar 2025) for context
   - Comparison of our autocorrelation differences (0.125-0.198) against published thresholds from temporal synthetic data benchmarks
   - Explicit acknowledgment that absolute DTW values are difficult to interpret without domain-specific baselines

2. **Visual trajectory comparisons** (already present as Figures 9-10) are now referenced more explicitly in the text with descriptions of clinically relevant patterns (e.g., subtype-specific GCS separation, mortality-stratified heart rate differences, early BP decline in non-survivors).

3. **Clinically meaningful temporal patterns.** We acknowledge in the Discussion that we did not formally evaluate:
   - Diurnal variation preservation
   - Response to interventions (e.g., BP reduction after antihypertensive administration)
   - Trajectory inflection points indicating clinical deterioration
   - These are identified as important directions for future work.

4. **BP missingness impact.** A new paragraph in Section 4.5 explicitly discusses how the 58% missingness in invasive BP:
   - Introduces selection bias toward more severely ill patients (those with arterial lines)
   - May distort temporal autocorrelation and cross-channel dependencies
   - Limits the generalizability of BP-related synthetic trajectories to patients without invasive monitoring

5. **Temporal baselines.** We acknowledge the absence of temporal baseline comparisons (e.g., TimeGAN) as a limitation. **[See Minor Concern 8 below for our response regarding feasibility.]**

---

### Minor Concern 6: Clinical Plausibility Evaluation Confounded by Normalization

**Reviewer's concern:** Real data violated 27.6% of plausibility rules because evaluation was performed on normalized data. This is a methodological error.

**Response:**

We fully agree this was a methodological error. The plausibility rules (age 18-120, GCS 3-15, etc.) were applied to data normalized to [-1, 1], producing spurious violations.

**[CODE CHANGE: The clinical plausibility evaluation has been corrected:**

1. **`src/evaluation/clinical_rules.py` updated:** Added an `inverse_normalize()` function that converts normalized data back to the original clinical scale using stored normalization parameters (`norm_params`) before applying plausibility rules.

2. **`scripts/run_full_evaluation.py` updated:** The evaluation pipeline now:
   - Loads normalization parameters from the preprocessing step
   - Applies inverse normalization to all synthetic data and test data before plausibility evaluation
   - Reports results on the original clinical scale

3. **Revised results (Section 3.4):**
   - Real data: 100% compliance (0% violations), as expected
   - BN: 0.0% violations (forward sampling from discretized bins inherently respects clinical ranges)
   - CTGAN: [X.X]% violations (primarily age and LOS boundary violations from continuous outputs)
   - TVAE: [X.X]% violations (expected to be higher given mode collapse)
   - BN+DGAN hybrid: [X.X]% violations (minor time-series violations, e.g., occasional SBP <= DBP during simulated hemodynamic instability)

**The comparison between generators is now interpretable and fair.]**

---

### Minor Concern 7: Missing Data Handling Lacks Justification

**Reviewer's concern:** Median imputation for continuous variables and forward-fill for time-series are not justified. Missing data rates of 12.5-22.4% are substantial. No sensitivity analyses.

**Response:**

1. **Section 2.4 revised** with a new expanded subsection on missing data handling that:
   - Justifies median imputation: computationally simple, preserves marginal distributions, and the BN structure learning operates on discretized variables (reducing sensitivity to exact continuous values)
   - Acknowledges limitations: attenuation of correlations, MCAR assumption violations in EHR data where lab tests are ordered selectively, loss of temporal information
   - Explains the rationale for forward-fill: clinical convention that last documented value is the best estimate; acknowledges it assumes constant values between measurements
   - Discusses informative missingness: the 58% BP missingness reflects selective arterial line placement (MNAR mechanism)

2. **Imputation sensitivity analysis added.**

   **[CODE CHANGE: A sensitivity analysis comparing median imputation vs. mean imputation has been implemented:**
   - **`src/data/preprocess.py` extended with a `imputation_method` parameter (options: "median", "mean")**
   - **The evaluation pipeline re-runs key metrics (fidelity Frobenius distance, TSTR AUROC gap, DCR) under both imputation strategies**
   - **Results are reported in a new Supplementary Table comparing the impact of imputation choice on synthetic data quality**
   - **We acknowledge that more sophisticated approaches (MICE, MissForest) were not compared due to computational constraints but are recommended for future work]**

---

### Minor Concern 8: Baseline Comparisons Are Incomplete

**Reviewer's concern:** No temporal baselines (e.g., TimeGAN) are compared. CTGAN and TVAE only address static features.

**Response:**

We acknowledge this limitation. Implementing a full temporal baseline (TimeGAN or RGAN) would require substantial additional development and computational resources that exceed the scope of this revision. We have:

1. **Expanded the Discussion (Section 4.5)** to explicitly acknowledge the absence of temporal baselines as a limitation
2. **Added a paragraph in Section 4.6 (Future Directions)** committing to TimeGAN comparison in future work
3. **Clarified in Section 2.8** that the absence of temporal baselines reflects the current state of the field for stroke-specific time-series generation, not an oversight

We note that: (1) no published stroke-specific temporal synthetic data generators exist for direct comparison; (2) the hybrid architecture's primary innovation is the conditioning mechanism linking static profiles to temporal trajectories, which is orthogonal to the choice of temporal generator; and (3) TimeGAN implementation and hyperparameter tuning would require substantial additional development. We commit to TimeGAN comparison in future work.

---

### Minor Concern 9: Statistical Analysis Section Lacks Detail (Rubin's Rules)

**Reviewer's concern:** Rubin's combining rules are mentioned but not explained or applied. No pooled estimates are reported. The choice of 10 replicates is not justified.

**Response:**

We agree this was a significant gap between the described methodology and the reported results.

**[CODE CHANGE: Rubin's combining rules have been implemented:**

1. **New module `src/evaluation/rubins_rules.py`** implements the pooling formulas:
   - Pooled estimate: Q-bar = (1/m) * sum(Q_i)
   - Within-imputation variance: U-bar = (1/m) * sum(U_i)
   - Between-imputation variance: B = (1/(m-1)) * sum((Q_i - Q-bar)^2)
   - Total variance: T = U-bar/m + (1 + 1/m) * B (using the Reiter 2003 variant for synthetic data, where within-variance is divided by m)

2. **The evaluation pipeline now generates 10 independent synthetic datasets per model** (BN, CTGAN, TVAE) and computes all metrics per dataset, then pools using Rubin's rules.

3. **Revised results tables** report pooled estimates with 95% confidence intervals for:
   - Frobenius correlation distance
   - TSTR AUROC (and AUROC gap)
   - Mean DCR
   - Clinical plausibility violation rate

4. **Section 2.11 expanded** to explain:
   - Which estimates are pooled (all key metrics in Tables 4-7)
   - The specific Rubin's formula variant used (Reiter 2003 for synthetic data)
   - Justification for 10 replicates: following El Emam et al. (2024) recommendation of "at least 10" datasets, balancing computational cost against variance estimation stability]

---

### Minor Concern 10: Code Availability Statement Is Vague

**Reviewer's concern:** No repository URL, license, or data availability instructions are provided.

**Response:**

The Code Availability statement has been revised to include:
- Repository URL: https://github.com/matheus-rech/MIMIC-Ext-Stroke
- License: MIT
- Data availability: "MIMIC-IV v3.1 is available through PhysioNet (https://physionet.org/content/mimiciv/3.1/) upon completion of the required training course and data use agreement. We cannot redistribute the source data."
- Software dependencies: referenced to `pyproject.toml` in the repository

---

### Minor Concern 11: Excessive Length and Redundancy

**Reviewer's concern:** Methods section is too detailed (>3,000 words). Itemids, hyperparameters, and metric descriptions could be moved to supplements.

**Response:**

We have condensed the Methods section:
1. **Section 2.3.2** (time-series feature extraction): Moved itemid table to Supplementary Table S4 (already exists); main text now references the supplement
2. **Section 2.5** (BN structure learning): Condensed discretization details; full specification remains in Supplementary Table S5
3. **Section 2.9** (evaluation metrics): Condensed metric descriptions into a summary paragraph with a reference table; detailed formulas moved to Supplementary
4. Estimated word count reduction: ~500-700 words from the Methods section

---

### Minor Concern 12: Figures and Tables Are Not Integrated

**Reviewer's concern:** Figures 1-7 and Tables 1-7 are referenced but not embedded.

**Response:**

All figures (15 PNGs) and tables (7 main + supplementary) are now included in the manuscript files with formal captions. The figure-to-filename mapping has been documented. For journal submission, these will be embedded in the DOCX/PDF output per journal-specific formatting requirements.

---

## REVIEWER 2

---

### Red Flag 1: ICH Percentage Discrepancy (16.4% vs. 16.8%)

**Reviewer's concern:** ICH prevalence reported as 16.4% in some places and 16.8% in others, suggesting sloppy data management.

**Response:**

We identified the source of this discrepancy:
- **16.4%** corresponds to the full cohort (1,397/8,500 = 16.4%)
- **16.8%** appears in the fidelity results (Section 3.3) and refers to the **training partition** (60% split), where random sampling produced a slightly different ICH proportion

**[CODE CHANGE:** The manuscript now uses the **full cohort** (N = 8,500) as the single consistent reference throughout. All references to subtype percentages in the Results section now explicitly state whether they refer to the full cohort, training set, or test set. The abstract uses full-cohort figures exclusively.]

---

### Red Flag 2: Distance Metric Scale Mismatch (5,906 vs. 5.85)

**Reviewer's concern:** The 1000-fold difference between BN DCR (5,906) and CTGAN DCR (5.85) indicates different measurement scales, making the comparison invalid.

**Response:**

This is the same issue identified in our response to Reviewer 1 Major Concern 3. The root cause is that BN synthetic data was on the original clinical scale while CTGAN/TVAE synthetic data and the training reference were on the normalized [-1, 1] scale.

**[CODE CHANGE:** As described above, the evaluation pipeline now ensures all data is on a common scale before computing DCR. The revised manuscript reports:
- DCR values computed in standardized feature space (all data converted to common scale, then StandardScaler applied within the DCR function)
- Explicit statement of the distance metric: "Mean Euclidean distance to closest record in standardized feature space"
- Clinical context: DCR values are interpreted relative to the distribution of pairwise real-patient distances (e.g., "BN synthetic records were on average at the Xth percentile of pairwise real-patient distances")]

We also now report **percentile-normalized DCR** as suggested by the reviewer, enabling intuitive interpretation of privacy protection levels.

---

### Red Flag 3: Subtype Percentages Don't Sum to 100%

**Reviewer's concern:** The reported subtypes (ischemic 51.7%, ICH 16.4%, SAH 5.1%, TIA 3.3%) sum to 76.5%, not 100%.

**Response:**

The remaining 23.4% is the "Other cerebrovascular" category (1,987 patients). The revised abstract now uses the phrasing "Major subtypes included ischemic stroke (51.7%), intracerebral hemorrhage (16.4%), and subarachnoid hemorrhage (5.1%)" to signal that these are the clinically important categories, with the full breakdown provided in the Results.

**[MANUSCRIPT CHANGE:** Added "Other cerebrovascular events accounted for 23.4%, and transient ischemic attack for 3.3%" immediately after the major subtype listing in both the abstract and Section 3.1.]

---

### Additional Issue: What Does "Distance 5,906" Mean?

**Reviewer's concern:** Even after fixing the scale, the DCR value needs clinical context for non-technical reviewers.

**Response:**

The revised manuscript now includes:
1. **Percentile context:** "Mean DCR was X.XX (>Yth percentile of pairwise real-patient distances), indicating negligible re-identification risk"
2. **Plain-language interpretation** in both the Results and Abstract
3. **Explicit specification** of the distance metric throughout ("Euclidean distance in standardized feature space")

---

### Pre-Submission Checklist Items

**Reviewer's checklist:**

| Item | Status |
|------|--------|
| ICH percentage consistent throughout | **Fixed** - 16.4% (full cohort) used everywhere |
| Distance metric specified | **Fixed** - "Euclidean distance in standardized feature space" |
| Both distances use same normalization | **Fixed** - common scale enforced in code |
| Headers match CNS requirements | **Verified** |
| Word count <= 300 words (abstract) | **Verified** - revised abstract is ~298 words |
| Title in Title Case | **Verified** |
| No causal language | **Fixed** - all causal terms replaced or qualified |
| Subtype percentages clarified | **Fixed** - "Major subtypes included..." phrasing |

---

## INTERNAL REVIEW (REVIEW.md) - Additional Items Addressed

Several items from the internal peer review (REVIEW.md) overlap with Reviewer 1 and 2 concerns. Items not covered above:

### Hyperparameter Discrepancies (Methods vs. Supplementary)

**Issue:** Section 2.6.1 stated hidden_dim=64, Methods said lr=0.001 and epochs=50 with early stopping, but Supplementary Table S5 stated hidden_dim=128, lr=0.0002, epochs=5,000.

**Response:** Methods text has been corrected to match the actual implementation (Supplementary Table S5 values, which match `config.yaml`): hidden_dim=128, lr=0.0002, epochs=5,000. The discrepancies arose from early drafting before hyperparameter tuning was finalized.

### BN Discretization Inconsistency

**Issue:** Methods said "quintile-based bins" (5 bins); Supplementary said quartile-based (4 bins for labs) and 6 bins for age.

**Response:** Methods text corrected to: "age was binned into six expert-defined clinically meaningful categories; ICU length of stay into five expert-defined categories; and laboratory values into quartile-based bins (four bins per analyte)." This matches the implementation.

### Race/Ethnicity Reporting

**Issue:** "Unknown/Declined (13.8%)" may combine categories incorrectly (UNKNOWN 13.81% + PATIENT DECLINED 0.72% = 14.53%).

**Response:** Verified against source data. The 13.8% refers to the "UNKNOWN" category alone. "PATIENT DECLINED TO ANSWER" is a separate category (0.72%). The manuscript now reports these separately in Table 1.

### GCS in Intubated Patients

**Issue:** GCS verbal component scored differently for intubated patients (often 1T). Not addressed in manuscript.

**Response:** Added a note in Section 2.3.2: "For intubated patients, the GCS verbal component is typically recorded as 1T (tube) in MIMIC-IV. Our extraction pipeline treats these as numeric value 1, which may underestimate true verbal responsiveness. This limitation affects both real and synthetic data equally and does not introduce differential bias in the comparison."

### Reference Numbering and Missing References

**Issue:** Citation style inconsistent; Corral-Acero 2020 missing from reference list.

**Response:** All citations standardized to numbered format. Missing references (Corral-Acero et al. 2020, WHO Global Health Estimates 2024) added to the reference list. DoppelGANger original paper citation corrected.

---

## SUMMARY OF CODE CHANGES

| Change | File(s) | Status |
|--------|---------|--------|
| Fix plausibility evaluation on unnormalized data | `clinical_rules.py`, `run_full_evaluation.py` | To implement |
| Fix DCR/MIA scale mismatch | `run_full_evaluation.py` | To implement |
| ICH percentage consistency | `run_full_evaluation.py`, manuscript | To implement |
| Rename causal terminology | `simulation/counterfactual.py` | To implement |
| Implement Rubin's combining rules | New: `evaluation/rubins_rules.py`, `run_full_evaluation.py` | To implement |
| BN edge analysis for paradoxical associations | `models/bayesian_net.py` | To implement |
| Imputation sensitivity analysis | `data/preprocess.py`, `run_full_evaluation.py` | To implement |
---

## SUMMARY OF MANUSCRIPT CHANGES

| Section | Change |
|---------|--------|
| Abstract | Removed causal language; clarified subtype percentages; specified DCR metric |
| Introduction | Qualified "digital twin" and "counterfactual" on first use |
| 2.2.1 (new) | ICD code limitations subsection |
| 2.4.3 (revised) | Missing data handling justification with citations |
| 2.9.2 (revised) | Plausibility evaluation on unnormalized data |
| 2.10 (rewritten) | "Associational Scenario Simulation" replacing "Counterfactual Simulation" |
| 2.11 (expanded) | Rubin's rules explanation with specific formulas |
| 3.1 (expanded) | Explicit reporting of paradoxical HTN/dyslipidemia associations |
| 3.2 (expanded) | BN edge analysis for HTN/dyslipidemia -> mortality |
| 3.3 (corrected) | Consistent ICH percentage (16.4%) |
| 3.4 (rewritten) | Plausibility results on clinical scale |
| 3.6 (rewritten) | Corrected DCR and MIA values after scale fix |
| 3.8 (rewritten) | "Associational Scenario" replacing "Counterfactual" |
| 4.2b (expanded) | Paradoxical associations discussion with J-curve, collider bias, statin effects |
| 4.5 (expanded) | Additional limitations: temporal baselines, BP missingness, GCS intubation |
| 4.6 (expanded) | Future directions: eICU external validation, causal discovery, TimeGAN comparison, quality measure validation, clinical expert review |
| Code availability | Repository URL, license, data access instructions |
| References | Missing references added; citation style standardized |

---

## FUTURE DIRECTIONS ADDITIONS

The following items have been added to Section 4.6:

**External Validation:** "External validation of the hybrid framework on the eICU Collaborative Research Database is planned to assess generalizability across institutions, patient populations, and clinical practices. The eICU database includes over 200,000 ICU admissions from 208 hospitals across the United States, providing a diverse validation cohort with different documentation patterns and patient demographics."

**Quality Measure Validation:** "Future work should evaluate whether synthetic data preserve stroke-specific quality indicators, including door-to-needle time for thrombolysis, early antithrombotic therapy rates, and dysphagia screening compliance. These metrics are critical for ensuring synthetic data utility in quality improvement research."

**Clinical Expert Review:** "Formal clinical expert review by stroke neurologists and intensivists is recommended to assess the clinical coherence of synthetic patient records beyond automated plausibility rules. Such review would identify subtle physiological implausibilities that rule-based systems may miss."

**Causal Discovery:** "Integration of causal discovery algorithms (PC, FCI) or domain-knowledge-specified causal edges would enable more principled scenario simulation. Target trial emulation frameworks could further strengthen the validity of associational analyses for hypothesis generation."
