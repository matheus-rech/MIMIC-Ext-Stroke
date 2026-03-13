# Peer Review: Stroke Digital Twins via Hybrid Bayesian-GAN Generative Models

**Reviewer:** Senior Peer Reviewer (Medical Informatics)
**Target Journals:** JAMIA / npj Digital Medicine
**Review Date:** 2026-03-13

---

## Overall Assessment

This manuscript presents a hybrid Bayesian Network plus DoppelGANger (BN+DGAN) framework for generating stroke digital twins from MIMIC-IV v3.1, combining static clinical profiles with 72-hour ICU time-series. The cohort characterization (Section 3.1) is thorough and the data are internally consistent with the CSV source tables. However, the manuscript has a critical structural problem: the quantitative evaluation results (Sections 3.3 through 3.8) contain placeholder values ([XX.X], [0.XX], etc.) throughout, meaning the core claims of the paper -- fidelity superiority, privacy adequacy, utility parity -- are entirely unsupported by reported numbers. In its current form, the manuscript is a well-written shell with excellent framing but no completed evaluation results, and cannot be submitted to any journal until these placeholders are replaced with actual computed metrics.

---

## Scores

| Criterion | Score (1-5) | Notes |
|-----------|-------------|-------|
| Scientific Rigor | 3 | Methods are well-described and reproducible in principle, but all quantitative evaluation results are placeholders. The 12-metric framework is comprehensive but unvalidated without numbers. |
| Clinical Accuracy | 4 | Cohort demographics and clinical interpretations are sound and match source data. Minor discrepancies noted below. |
| Novelty and Contribution | 4 | The hybrid BN+DGAN architecture and stroke-specific application are genuinely novel. Positioning against prior work is fair. |
| Writing Quality | 4 | Professional, journal-ready prose. Well-organized sections. Some redundancy between Introduction paragraph 3 and Methods 2.9. Minor style issues noted. |
| Completeness | 2 | Placeholder results throughout Sections 3.3-3.8 are a fatal gap. No actual evaluation tables (Tables 4-7) exist. TVAE baseline mentioned in abstract/methods but absent from results. |
| Figure and Table Quality | 3 | Figures referenced in text appear to exist on disk. Supplementary tables are well-formatted. Main-text Tables 4-7 are referenced but never provided. |
| Reference Accuracy | 4 | Key papers are cited. A few important references are missing (see below). Citation numbering in Methods does not match the reference list format. |

---

## Critical Issues (MUST fix before submission)

1. **All quantitative evaluation results are placeholder values.** Sections 3.3 (Fidelity), 3.4 (Clinical Plausibility), 3.5 (Utility/TSTR), 3.6 (Privacy), 3.7 (Temporal Fidelity), and 3.8 (Counterfactual Simulation) contain [XX.X] and [0.XX] placeholders instead of actual computed metrics. This affects approximately 60+ individual data points. Without these numbers, the paper's core claims are entirely unsupported. Tables 4 through 7 are referenced in the text but never provided anywhere in the manuscript or supplementary materials.

2. **TVAE baseline disappears from results.** The abstract mentions "benchmarked against CTGAN and TVAE baselines" and Section 2.8 describes TVAE in detail, but the Results section only compares BN vs. CTGAN. TVAE results are completely absent. Either include TVAE results or remove it from the abstract and methods.

3. **Inconsistency in Bayesian Network hyperparameters between Methods and Supplementary.** Section 2.5 states continuous variables were "discretized into quintile-based bins" (5 bins). Supplementary Table S5 states lab discretization uses "Quartile-based (4 bins per lab)" and age uses "6 bins." These are contradictory. The methods text must accurately reflect the actual implementation.

4. **Inconsistency in DGAN hyperparameters between Methods and Supplementary.** Section 2.6.1 states "hidden dimensionality of 64 units per layer"; Supplementary Table S5 states "Hidden dimension: 128." Section 2.6.3 states "learning rate of 0.001"; Supplementary Table S5 states "Learning rate: 0.0002." Section 2.6.3 states "minimum of 50 epochs with early stopping"; Supplementary Table S5 states "Epochs: 5,000." These are significant discrepancies that undermine reproducibility. The methods text and supplementary table must agree.

5. **Missing formal figure and table cross-references.** The manuscript references "Figure 1" through "Figure 10" and "Table 1" through "Table 7" in the results, but no figure or table objects are embedded in the manuscript files. For journal submission, these must be formally included or referenced with proper captions. The existing PNG files on disk (15 figures) do not have a clear mapping to the numbered references in the text.

---

## Major Issues (Should fix)

1. **Race/ethnicity reporting in manuscript vs. data.** The results state "Unknown/Declined (13.8%)" but the cohort_summary_stats.csv shows "UNKNOWN" at 13.81% and "PATIENT DECLINED TO ANSWER" at 0.72% as separate categories. If these were combined, the total would be 14.53%, not 13.8%. The manuscript should clarify which categories were grouped and report the correct percentage.

2. **"Black/African American" prevalence.** The manuscript reports 7.0% but the CSV shows the "BLACK/AFRICAN AMERICAN" category alone at 6.98%. There are additional Black categories (BLACK/CARIBBEAN ISLAND 0.91%, BLACK/CAPE VERDEAN 0.55%, BLACK/AFRICAN 0.47%) totaling approximately 8.9% if combined. The manuscript should specify whether sub-categories were aggregated and report the number consistently.

3. **Missing data rates stated imprecisely.** Section 3.1 states "laboratory values missing in 12.5-13.7% of cases." The CSV shows: creatinine 12.5%, hemoglobin 13.1%, platelets 13.1%, sodium 13.2%, glucose 13.7%, and INR 22.4%. The range 12.5-13.7% excludes INR (22.4%), which is mentioned separately, but this should be stated more explicitly to avoid ambiguity.

4. **Causal language creep.** Despite appropriate disclaimers in the Discussion (Section 4.4, paragraph on counterfactual limitations), the Introduction (paragraph 3) and abstract use phrases like "counterfactual simulation" and "individual-level counterfactual simulation" without immediate qualification. For JAMIA/npj Digital Medicine reviewers, the causal implications of "counterfactual" are strong. Consider adding "correlation-based" or "associative" qualifiers earlier in the manuscript, particularly in the abstract.

5. **Blood pressure itemids capture only arterial line measurements.** The methods specify itemids 220050/220051/220052 which are arterial blood pressure only. Section 4.5 acknowledges ~58% missingness but does not note this in the Methods (Section 2.3.2) where the extraction is described. This selection bias should be flagged at the point of feature definition, not only in limitations.

6. **No formal sample size justification.** The cohort of 8,500 is described as the result of inclusion/exclusion criteria, which is appropriate, but there is no discussion of whether this sample size provides adequate power for the evaluation metrics, particularly for the rare SAH subtype (n=436) and the privacy assessments.

7. **Rubin's rules application described but never shown.** Section 2.11 describes Rubin's combining rules for pooling across 10 synthetic datasets, but no pooled estimates with confidence intervals are reported anywhere in the results. This methodological commitment is unfulfilled.

8. **ICD code mapping for comorbidities may undercount.** The methods specify single root codes (e.g., ICD-10 I10 for hypertension, E11 for diabetes). However, I10 alone captures only "essential (primary) hypertension" and misses secondary hypertension codes (I15.x). E11 captures type 2 diabetes but misses type 1 (E10.x). This should be acknowledged as a potential source of underascertainment.

9. **Abstract states "median age 68 years" for synthetic cohort.** The abstract says "The synthetic cohort preserved key demographic and clinical distributions of the source population (median age 68 years...)." However, this number matches the real cohort. Since no actual synthetic cohort statistics are reported (all placeholders), this claim is unverifiable. The abstract should report synthetic data statistics only after they are computed.

---

## Minor Issues (Nice to fix)

1. **Reference numbering inconsistency.** The Methods section uses bracketed numbers [1]-[7] while the Introduction uses author-year citations (e.g., "[Corral-Acero 2020]"). The reference list (06_references.md) uses its own numbered format [1]-[25]. A single citation style should be adopted throughout.

2. **Admission type categories.** Methods Section 2.3.1 describes admission type as "elective, emergency, urgent, or other" but the actual data shows 9 admission types (EW EMER., OBSERVATION ADMIT, URGENT, SURGICAL SAME DAY ADMISSION, ELECTIVE, DIRECT EMER., EU OBSERVATION, DIRECT OBSERVATION, AMBULATORY OBSERVATION). The methods should describe how these were recoded, or the text should reflect the actual categories used.

3. **"First care unit" not described in results.** This variable is listed as a static feature (Section 2.3.1) and appears in the summary stats (13 distinct units), but is never discussed in the Results or used in any reported analysis. Its inclusion should be justified or it should be noted as a conditioning variable only.

4. **Introduction length.** At approximately 1,100 words in a single unbroken section, the Introduction is dense. JAMIA guidelines suggest structured flow; consider adding subsection headers or at minimum paragraph breaks that map to: (a) clinical burden, (b) data challenges, (c) generative model landscape, (d) gap and contribution.

5. **Supplementary Table S2 duplicates main text Table 2.** The subtype-stratified cohort characteristics in Supplementary Table S2 appear to be the same data described as "Table 2" in Section 3.1. Clarify whether these are the same table or different.

6. **Typo/style: "artefactual" (Section 2.4.4).** While correct in British English, JAMIA uses American English conventions. Use "artifactual" for consistency.

7. **The term "digital twin" as used here.** The framework generates synthetic populations, not patient-specific continuously-updating replicas. The counterfactual simulation is closer to what-if population modeling than the engineering definition of a digital twin. Consider a brief explicit statement distinguishing your usage from the aerospace/engineering definition, which implies real-time bidirectional data flow.

8. **Software version specificity.** Methods cite "PyTorch (version 2.4)" and "scikit-learn (version 1.5)" but the Supplementary also references "DuckDB (v0.9+)" while Methods Section 2.1 says "DuckDB (version 1.1)." Standardize version numbers.

9. **Corral-Acero 2020 and Laubenbacher 2024 cited in Introduction but not in reference list.** The Introduction cites "[Corral-Acero 2020; Laubenbacher 2024]" but the reference list only has Laubenbacher [22] and Bjornsson [21]. Corral-Acero is missing entirely. Additionally, WHO 2024 and Tsao 2023 from the Introduction do not clearly map to references [13] and [14] (GBD 2021 and Tsao 2023).

---

## Data Accuracy Check

### Numbers that MATCH between manuscript and CSV data:

- Total cohort: 8,500 -- MATCH
- Median age: 68.0 (IQR 57.0-78.0) -- MATCH
- Gender F: 50.1% (n=4,258) -- MATCH
- Gender M: 49.9% (n=4,242) -- MATCH
- Ischemic: 4,398 (51.7%) -- MATCH
- ICH: 1,397 (16.4%) -- MATCH
- SAH: 436 (5.1%) -- MATCH
- TIA: 282 (3.3%) -- MATCH
- Other: 1,987 (23.4%) -- MATCH
- Hypertension: 51.7% -- MATCH (51.67% in CSV)
- Diabetes: 28.1% -- MATCH (28.09% in CSV)
- Atrial fibrillation: 31.2% -- MATCH (31.25% in CSV)
- Dyslipidemia: 52.8% -- MATCH (52.75% in CSV)
- CKD: 17.8% -- MATCH (17.79% in CSV)
- CAD: 25.2% -- MATCH (25.19% in CSV)
- In-hospital mortality: 14.8% (n=1,258) -- MATCH
- ICH mortality: 22.0% -- MATCH
- SAH mortality: 20.9% -- MATCH
- Ischemic mortality: 17.0% -- MATCH
- TIA mortality: 6.0% -- MATCH
- Other mortality: 4.9% -- MATCH
- Non-survivor age: 73.0 vs. CSV shows 73.0 (IQR 62.0-82.0) -- MATCH
- Survivor age: 68.0 vs. CSV shows 68.0 (IQR 57.0-78.0) -- MATCH
- Non-survivor male: 52.5% -- MATCH (661/1258 = 52.5%)
- Survivor female: 50.6% -- MATCH (3661/7242 = 50.6%)
- ICU LOS: 2.7 days (IQR 1.3-5.9) -- MATCH
- Non-survivor ICU LOS: 3.4 vs. 2.6 -- MATCH
- SAH ICU LOS: 5.7 days -- MATCH
- All mortality-stratified comorbidity percentages -- MATCH
- All mortality-stratified laboratory values -- MATCH
- Comorbidity co-occurrence percentages -- MATCH (30.9%, 19.4%, 19.0%, 0.4%)
- All subtype-stratified values in Supplementary Table S2 -- MATCH against table1_by_subtype.csv

### Numbers with DISCREPANCIES:

1. **Survivor age in manuscript text.** Section 3.1 states "non-survivors were significantly older (median 73.0 vs. 68.0 years)." The CSV for survivors shows median 68.0, which matches. However, the IQR for survivors in the CSV is (57.0-78.0) which is identical to the overall IQR -- this is expected given survivors are 85% of the cohort. No discrepancy, but worth noting the survivor IQR was never explicitly stated in the manuscript.

2. **Hypertension in non-survivors.** Section 3.1 lists non-survivors as having "higher comorbidity burden" and then enumerates AF, diabetes, CKD, and CAD as significantly higher in non-survivors. However, the CSV shows hypertension was actually LOWER in non-survivors (47.1%) vs. survivors (52.5%), p<0.001. The manuscript does not mention this -- it is a notable omission because the direction contradicts the stated "higher comorbidity burden" narrative for hypertension specifically. This should be explicitly discussed as a finding (possibly related to survivor treatment effect or competing risks).

3. **Dyslipidemia in non-survivors.** Similarly, dyslipidemia was LOWER in non-survivors (45.3%) vs. survivors (54.0%), p<0.001. This is also omitted from the mortality comparison narrative. Both hypertension and dyslipidemia showing inverse associations with mortality are clinically important and warrant comment (possibly reflecting the "obesity paradox" or statin/antihypertensive treatment effects).

4. **Gender in mortality stratification.** The manuscript states non-survivors were "more frequently male (52.5% vs. 49.4%)." The CSV shows survivor males at 49.4% and non-survivor males at 52.5%, which matches. However, the manuscript says "p = 0.046" while the CSV also shows 0.046. Match, but this borderline significance should be interpreted cautiously.

5. **Overall sodium IQR.** The manuscript overall table1 CSV shows Sodium Median (IQR) = 139.0 (137.0-142.0), but the mortality table shows survivors at 139.0 (137.0-141.0) and non-survivors at 139.0 (136.0-142.0). The overall IQR upper bound (142.0) is wider than the survivor subgroup (141.0), which is mathematically possible but worth verifying.

---

## Missing Elements for Journal Submission

1. **Completed evaluation results.** All placeholder values in Sections 3.3-3.8 must be filled with actual computed metrics. This is the single most important gap.

2. **Tables 4-7.** These are referenced in the results but do not exist in any manuscript file. Table 4 (correlation preservation comparison), Table 5 (clinical plausibility rules), Table 6 (TSTR mortality prediction), and Table 7 (privacy metrics) must be created.

3. **Author information.** No author list, affiliations, corresponding author, or ORCID identifiers are provided.

4. **Conflicts of interest statement.** Required by both JAMIA and npj Digital Medicine.

5. **Funding statement.** Required even if unfunded.

6. **Ethics approval details.** Section 2.1 mentions IRB approval but does not provide a protocol number or the specific IRB.

7. **STROBE or RECORD checklist.** For observational studies using routinely collected health data, JAMIA expects adherence to reporting guidelines.

8. **CRediT author contribution statement.** Required by npj Digital Medicine.

9. **Word count.** JAMIA Research and Applications papers have a 4,000-word limit for the main text. The current manuscript (excluding references and supplementary) is likely well above this. npj Digital Medicine has no strict word limit but recommends conciseness.

10. **Formal figure captions.** The 15 PNG figures exist on disk but none have formal captions in the main manuscript. Only supplementary figures have captions.

11. **Missing key references:**
    - Corral-Acero et al. (2020) -- cited in Introduction but absent from reference list
    - WHO Global Health Estimates (2024) -- cited in Introduction as "[WHO 2024]" but not in references
    - The original DoppelGANger paper by Lin et al. (2020) is cited as [11] but referenced in Methods as [4]
    - Patki et al. (2016) SDV paper [25] is in the reference list but never cited in the text

12. **Code availability.** The supplementary references a GitHub repository (https://github.com/matheus-rech/MIMIC-Ext-Stroke) -- confirm this exists and is accessible.

13. **Limitations around GCS interpretation.** The GCS verbal component is scored differently for intubated patients (often recorded as 1T). The manuscript does not address how intubated patients were handled for GCS, which is critical in an ICU population where many patients may be mechanically ventilated.

---

## Summary Recommendation

**Decision: Major Revision (Revise and Resubmit)**

The manuscript presents a novel and clinically relevant framework with strong methodological design. The cohort characterization is thorough and accurate. However, it cannot be submitted in its current state due to the absence of all quantitative evaluation results (the core scientific contribution). Once the evaluation pipeline is completed and results are populated, the manuscript has strong potential for publication in JAMIA or npj Digital Medicine, contingent on resolving the hyperparameter discrepancies between methods and supplementary materials, addressing the missing TVAE comparison, and adding required journal submission elements (author information, COI, funding, reporting checklist).

The highest-priority action items are:
1. Complete the evaluation pipeline and populate all placeholder values
2. Create Tables 4-7
3. Reconcile Methods vs. Supplementary hyperparameter discrepancies
4. Include TVAE results or remove from scope
5. Address the hypertension/dyslipidemia inverse mortality association finding
6. Add all required journal submission metadata

---

*Review generated 2026-03-13*
