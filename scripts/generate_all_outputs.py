#!/usr/bin/env python3
"""
Generate comprehensive publication-quality figures and tables
for the MIMIC-IV stroke cohort digital twin project.
"""

import warnings
warnings.filterwarnings("ignore")

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from scipy import stats
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE = Path(__file__).resolve().parent.parent
COHORT = BASE / "outputs" / "cohort"
FIG_DIR = BASE / "outputs" / "figures"
TBL_DIR = BASE / "outputs" / "tables"
FIG_DIR.mkdir(parents=True, exist_ok=True)
TBL_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
sns.set_theme(style="whitegrid", font_scale=1.1)
PAL = sns.color_palette("Set2")
PAL10 = sns.color_palette("tab10")
DPI = 300

# Column name mapping (actual -> expected alias)
LAB_MAP = {
    "lab_glucose": "glucose_admit",
    "lab_sodium": "sodium_admit",
    "lab_creatinine": "creatinine_admit",
    "lab_hemoglobin": "hemoglobin_admit",
    "lab_platelets": "platelet_admit",
    "lab_inr": "inr_admit",
}

COMORBIDITIES = [
    "has_hypertension", "has_diabetes", "has_afib",
    "has_dyslipidemia", "has_ckd", "has_cad",
]
COMORB_LABELS = {
    "has_hypertension": "Hypertension",
    "has_diabetes": "Diabetes",
    "has_afib": "Atrial Fibrillation",
    "has_dyslipidemia": "Dyslipidemia",
    "has_ckd": "CKD",
    "has_cad": "CAD",
}

LAB_NICE = {
    "lab_glucose": "Glucose (mg/dL)",
    "lab_sodium": "Sodium (mEq/L)",
    "lab_creatinine": "Creatinine (mg/dL)",
    "lab_hemoglobin": "Hemoglobin (g/dL)",
    "lab_platelets": "Platelets (10³/µL)",
    "lab_inr": "INR",
}

LAB_NORMAL = {
    "lab_glucose": (70, 100),
    "lab_sodium": (136, 145),
    "lab_creatinine": (0.7, 1.3),
    "lab_hemoglobin": (12.0, 17.5),
    "lab_platelets": (150, 400),
    "lab_inr": (0.8, 1.1),
}

SUBTYPE_ORDER = ["ischemic", "ich", "sah", "tia", "other"]
SUBTYPE_NICE = {"ischemic": "Ischemic", "ich": "ICH", "sah": "SAH", "tia": "TIA", "other": "Other"}


def load_data():
    print("Loading static features …")
    sf = pd.read_parquet(COHORT / "static_features.parquet")
    print(f"  Static features: {sf.shape}")
    print("Loading timeseries …")
    ts = pd.read_parquet(COHORT / "timeseries_processed.parquet")
    print(f"  Timeseries: {ts.shape}")
    return sf, ts


# ===================================================================
# FIGURES
# ===================================================================

def fig_demographics(sf):
    """1. demographics.png — 3-panel."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Age histogram
    ax = axes[0]
    med_age = sf["anchor_age"].median()
    ax.hist(sf["anchor_age"], bins=30, color=PAL[0], edgecolor="white", alpha=0.85)
    ax.axvline(med_age, color="red", ls="--", lw=2, label=f"Median = {med_age:.0f}")
    ax.set_xlabel("Age (years)")
    ax.set_ylabel("Count")
    ax.set_title("Age Distribution")
    ax.legend()

    # Gender bar
    ax = axes[1]
    gc = sf["gender"].value_counts()
    bars = ax.bar(gc.index, gc.values, color=[PAL[1], PAL[2]], edgecolor="white")
    for b in bars:
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 30,
                f"{int(b.get_height())}", ha="center", fontsize=11)
    ax.set_ylabel("Count")
    ax.set_title("Gender Distribution")

    # Race bar
    ax = axes[2]
    rc = sf["race"].value_counts().head(8)
    bars = ax.barh(rc.index[::-1], rc.values[::-1], color=PAL[3], edgecolor="white")
    for b in bars:
        ax.text(b.get_width() + 20, b.get_y() + b.get_height()/2,
                f"{int(b.get_width())}", va="center", fontsize=10)
    ax.set_xlabel("Count")
    ax.set_title("Race / Ethnicity")

    fig.tight_layout()
    fig.savefig(FIG_DIR / "demographics.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print("  [saved] demographics.png")


def fig_stroke_subtypes(sf):
    """2. stroke_subtypes.png."""
    fig, ax = plt.subplots(figsize=(8, 5))
    vc = sf["stroke_subtype"].value_counts().reindex(SUBTYPE_ORDER)
    total = vc.sum()
    bars = ax.bar([SUBTYPE_NICE[s] for s in vc.index], vc.values, color=PAL[:5], edgecolor="white")
    for b, v in zip(bars, vc.values):
        pct = v / total * 100
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 30,
                f"{v}\n({pct:.1f}%)", ha="center", fontsize=11)
    ax.set_ylabel("Count")
    ax.set_title("Stroke Subtype Distribution")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "stroke_subtypes.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print("  [saved] stroke_subtypes.png")


def fig_comorbidities(sf):
    """3. comorbidities.png — horizontal bar."""
    fig, ax = plt.subplots(figsize=(8, 5))
    prev = sf[COMORBIDITIES].mean() * 100
    prev = prev.sort_values()
    labels = [COMORB_LABELS[c] for c in prev.index]
    bars = ax.barh(labels, prev.values, color=PAL[4], edgecolor="white")
    for b, v in zip(bars, prev.values):
        ax.text(b.get_width() + 0.5, b.get_y() + b.get_height()/2,
                f"{v:.1f}%", va="center", fontsize=11)
    ax.set_xlabel("Prevalence (%)")
    ax.set_title("Comorbidity Prevalence")
    ax.set_xlim(0, prev.max() * 1.15)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "comorbidities.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print("  [saved] comorbidities.png")


def fig_mortality_by_subtype(sf):
    """4. mortality_by_subtype.png."""
    fig, ax = plt.subplots(figsize=(8, 5))
    mort = sf.groupby("stroke_subtype")["hospital_expire_flag"].mean().reindex(SUBTYPE_ORDER) * 100
    bars = ax.bar([SUBTYPE_NICE[s] for s in mort.index], mort.values, color=PAL[:5], edgecolor="white")
    for b, v in zip(bars, mort.values):
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.3,
                f"{v:.1f}%", ha="center", fontsize=11)
    ax.set_ylabel("Mortality Rate (%)")
    ax.set_title("In-Hospital Mortality by Stroke Subtype")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "mortality_by_subtype.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print("  [saved] mortality_by_subtype.png")


def fig_mortality_by_age(sf):
    """5. mortality_by_age.png — by age decile with CI."""
    fig, ax = plt.subplots(figsize=(8, 5))
    sf = sf.copy()
    sf["age_decile"] = pd.cut(sf["anchor_age"], bins=range(10, 110, 10))

    def mort_ci(g):
        n = len(g)
        k = g.sum()
        p = k / n if n > 0 else 0
        se = np.sqrt(p * (1 - p) / n) if n > 0 else 0
        return pd.Series({"rate": p * 100, "ci": 1.96 * se * 100, "n": n})

    agg = sf.groupby("age_decile", observed=True)["hospital_expire_flag"].apply(mort_ci).unstack()
    labels = [str(i) for i in agg.index]
    x = np.arange(len(labels))
    ax.bar(x, agg["rate"], yerr=agg["ci"], capsize=4, color=PAL[0], edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("Mortality Rate (%)")
    ax.set_xlabel("Age Group")
    ax.set_title("In-Hospital Mortality by Age Decile (95% CI)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "mortality_by_age.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print("  [saved] mortality_by_age.png")


def fig_admission_labs(sf):
    """6. admission_labs.png — 2x3 grid with normal range shading."""
    lab_cols = list(LAB_NICE.keys())
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    for ax, col in zip(axes.flat, lab_cols):
        data = sf[col].dropna()
        ax.hist(data, bins=40, color=PAL[1], edgecolor="white", alpha=0.85)
        lo, hi = LAB_NORMAL[col]
        ax.axvspan(lo, hi, alpha=0.15, color="green", label="Normal range")
        ax.set_title(LAB_NICE[col])
        ax.set_ylabel("Count")
        ax.legend(fontsize=9)
    fig.suptitle("Admission Laboratory Values", fontsize=14, y=1.01)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "admission_labs.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print("  [saved] admission_labs.png")


def fig_los_distribution(sf):
    """7. los_distribution.png — violin + box by subtype."""
    fig, ax = plt.subplots(figsize=(10, 6))
    order = [SUBTYPE_NICE[s] for s in SUBTYPE_ORDER]
    tmp = sf.copy()
    tmp["Subtype"] = tmp["stroke_subtype"].map(SUBTYPE_NICE)
    # cap for visualisation
    tmp["los_cap"] = tmp["los"].clip(upper=tmp["los"].quantile(0.99))
    sns.violinplot(data=tmp, x="Subtype", y="los_cap", order=order,
                   inner=None, palette=PAL[:5], alpha=0.4, ax=ax)
    sns.boxplot(data=tmp, x="Subtype", y="los_cap", order=order,
                width=0.15, palette=PAL[:5], boxprops=dict(alpha=0.8),
                fliersize=2, ax=ax)
    ax.set_ylabel("ICU Length of Stay (days)")
    ax.set_xlabel("Stroke Subtype")
    ax.set_title("ICU LOS Distribution by Stroke Subtype")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "los_distribution.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print("  [saved] los_distribution.png")


def fig_sample_trajectories(ts):
    """8. sample_trajectories.png — 2x2 grid for 3 sample patients."""
    stay_ids = ts["stay_id"].unique()
    np.random.seed(42)
    chosen = np.random.choice(stay_ids, size=3, replace=False)
    vitals = [("hr", "Heart Rate (bpm)"), ("spo2", "SpO2 (%)"),
              ("sbp", "Systolic BP (mmHg)"), ("gcs_total", "GCS Total")]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    colors = PAL10[:3]
    for ax, (col, title) in zip(axes.flat, vitals):
        for i, sid in enumerate(chosen):
            sub = ts[ts["stay_id"] == sid].sort_values("hour")
            ax.plot(sub["hour"], sub[col], marker=".", markersize=3,
                    label=f"Patient {i+1}", color=colors[i], alpha=0.8)
        ax.set_xlabel("Hour")
        ax.set_ylabel(title)
        ax.set_title(title)
        ax.legend(fontsize=9)
    fig.suptitle("Sample Patient Trajectories (first 72h)", fontsize=14, y=1.01)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "sample_trajectories.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print("  [saved] sample_trajectories.png")


def fig_ts_missing(ts):
    """9. ts_missing_rates.png — horizontal bar of missing rates."""
    vital_cols = ["hr", "sbp", "dbp", "map", "rr", "spo2", "temp_c",
                  "gcs_eye", "gcs_verbal", "gcs_motor", "gcs_total"]
    miss = ts[vital_cols].isnull().mean().sort_values() * 100
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.barh(miss.index, miss.values, color=PAL[5], edgecolor="white")
    for b, v in zip(bars, miss.values):
        ax.text(b.get_width() + 0.5, b.get_y() + b.get_height()/2,
                f"{v:.1f}%", va="center", fontsize=10)
    ax.set_xlabel("Missing Rate (%)")
    ax.set_title("Time-Series Missing Data Rates")
    ax.set_xlim(0, miss.max() * 1.15)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "ts_missing_rates.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print("  [saved] ts_missing_rates.png")


def fig_correlation_heatmap(sf):
    """10. correlation_heatmap.png."""
    num_cols = ["anchor_age", "los", "hospital_expire_flag",
                "has_hypertension", "has_diabetes", "has_afib",
                "has_dyslipidemia", "has_ckd", "has_cad",
                "lab_glucose", "lab_sodium", "lab_creatinine",
                "lab_hemoglobin", "lab_platelets", "lab_inr"]
    corr = sf[num_cols].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
                center=0, square=True, linewidths=0.5, ax=ax,
                annot_kws={"size": 8})
    ax.set_title("Correlation Matrix — Static Features")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "correlation_heatmap.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print("  [saved] correlation_heatmap.png")


def fig_comorbidity_cooccurrence(sf):
    """11. comorbidity_cooccurrence.png."""
    cmat = sf[COMORBIDITIES].copy()
    n = len(cmat)
    labels = [COMORB_LABELS[c] for c in COMORBIDITIES]
    cooc = pd.DataFrame(np.zeros((len(COMORBIDITIES), len(COMORBIDITIES))),
                        index=labels, columns=labels)
    for i, c1 in enumerate(COMORBIDITIES):
        for j, c2 in enumerate(COMORBIDITIES):
            cooc.iloc[i, j] = (cmat[c1] & cmat[c2]).sum() / n * 100

    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(cooc, annot=True, fmt=".1f", cmap="YlOrRd",
                square=True, linewidths=0.5, ax=ax)
    ax.set_title("Comorbidity Co-occurrence (%)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "comorbidity_cooccurrence.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print("  [saved] comorbidity_cooccurrence.png")


def fig_age_comorbidity_stacked(sf):
    """12. age_comorbidity_stacked.png."""
    sf = sf.copy()
    bins = [0, 50, 65, 80, 200]
    labels_ag = ["<50", "50-64", "65-79", "80+"]
    sf["age_group"] = pd.cut(sf["anchor_age"], bins=bins, labels=labels_ag)
    prev = sf.groupby("age_group", observed=True)[COMORBIDITIES].mean() * 100
    prev.columns = [COMORB_LABELS[c] for c in prev.columns]

    fig, ax = plt.subplots(figsize=(10, 6))
    prev.plot(kind="bar", stacked=True, ax=ax, color=PAL[:6], edgecolor="white")
    ax.set_ylabel("Cumulative Prevalence (%)")
    ax.set_xlabel("Age Group")
    ax.set_title("Comorbidity Prevalence by Age Group")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    ax.set_xticklabels(labels_ag, rotation=0)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "age_comorbidity_stacked.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print("  [saved] age_comorbidity_stacked.png")


def fig_labs_by_mortality(sf):
    """13. labs_by_mortality.png — box plots stratified by survival."""
    lab_cols = list(LAB_NICE.keys())
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    tmp = sf.copy()
    tmp["Outcome"] = tmp["hospital_expire_flag"].map({0: "Survived", 1: "Died"})
    for ax, col in zip(axes.flat, lab_cols):
        sns.boxplot(data=tmp, x="Outcome", y=col, palette=[PAL[0], PAL[3]],
                    ax=ax, fliersize=2)
        ax.set_title(LAB_NICE[col])
        ax.set_xlabel("")
    fig.suptitle("Admission Labs by Mortality Outcome", fontsize=14, y=1.01)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "labs_by_mortality.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print("  [saved] labs_by_mortality.png")


def fig_gcs_trajectory_by_subtype(sf, ts):
    """14. gcs_trajectory_by_subtype.png — mean GCS with 95% CI first 72h."""
    merged = ts.merge(sf[["stay_id", "stroke_subtype"]].drop_duplicates(), on="stay_id")
    merged = merged[merged["hour"] <= 72]
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, st in enumerate(SUBTYPE_ORDER):
        sub = merged[merged["stroke_subtype"] == st]
        agg = sub.groupby("hour")["gcs_total"].agg(["mean", "sem"]).reset_index()
        agg["ci"] = 1.96 * agg["sem"]
        ax.plot(agg["hour"], agg["mean"], label=SUBTYPE_NICE[st], color=PAL10[i])
        ax.fill_between(agg["hour"], agg["mean"] - agg["ci"], agg["mean"] + agg["ci"],
                        alpha=0.15, color=PAL10[i])
    ax.set_xlabel("Hour")
    ax.set_ylabel("GCS Total")
    ax.set_title("Mean GCS Trajectory by Stroke Subtype (95% CI)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "gcs_trajectory_by_subtype.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print("  [saved] gcs_trajectory_by_subtype.png")


def fig_vital_trends_by_mortality(sf, ts):
    """15. vital_trends_by_mortality.png — HR and SBP first 48h by outcome."""
    merged = ts.merge(sf[["stay_id", "hospital_expire_flag"]].drop_duplicates(), on="stay_id")
    merged = merged[merged["hour"] <= 48]
    merged["Outcome"] = merged["hospital_expire_flag"].map({0: "Survived", 1: "Died"})

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, (col, title) in zip(axes, [("hr", "Heart Rate (bpm)"), ("sbp", "Systolic BP (mmHg)")]):
        for i, outcome in enumerate(["Survived", "Died"]):
            sub = merged[merged["Outcome"] == outcome]
            agg = sub.groupby("hour")[col].agg(["mean", "sem"]).reset_index()
            agg["ci"] = 1.96 * agg["sem"]
            c = PAL[0] if outcome == "Survived" else PAL[3]
            ax.plot(agg["hour"], agg["mean"], label=outcome, color=c)
            ax.fill_between(agg["hour"], agg["mean"] - agg["ci"],
                            agg["mean"] + agg["ci"], alpha=0.15, color=c)
        ax.set_xlabel("Hour")
        ax.set_ylabel(title)
        ax.set_title(f"{title} — First 48h by Outcome")
        ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "vital_trends_by_mortality.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print("  [saved] vital_trends_by_mortality.png")


# ===================================================================
# TABLES
# ===================================================================

def _median_iqr(s):
    """Return 'median (Q1-Q3)' string."""
    med = s.median()
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    return f"{med:.1f} ({q1:.1f}-{q3:.1f})"


def _n_pct(s, total=None):
    """Return 'n (%)' string."""
    n = int(s.sum())
    total = total or len(s)
    return f"{n} ({n/total*100:.1f}%)"


def _build_table1(df, label="Overall"):
    """Build a Table 1 for a dataframe subset."""
    rows = []
    n = len(df)
    rows.append(("N", "n", str(n)))
    # Age
    rows.append(("Age (years)", "Median (IQR)", _median_iqr(df["anchor_age"])))
    # Gender
    for g in ["F", "M"]:
        rows.append((f"Gender — {g}", "n (%)", _n_pct(df["gender"] == g, n)))
    # Stroke subtype
    for st in SUBTYPE_ORDER:
        nice = SUBTYPE_NICE[st]
        rows.append((f"Stroke subtype — {nice}", "n (%)", _n_pct(df["stroke_subtype"] == st, n)))
    # Comorbidities
    for c in COMORBIDITIES:
        rows.append((COMORB_LABELS[c], "n (%)", _n_pct(df[c], n)))
    # Labs
    for col in LAB_NICE:
        rows.append((LAB_NICE[col], "Median (IQR)", _median_iqr(df[col].dropna())))
    # LOS
    rows.append(("ICU LOS (days)", "Median (IQR)", _median_iqr(df["los"])))
    # Mortality
    rows.append(("In-hospital mortality", "n (%)", _n_pct(df["hospital_expire_flag"], n)))

    return pd.DataFrame(rows, columns=["Variable", "Statistic", label])


def table1_overall(sf):
    """1. table1_overall.csv."""
    t1 = _build_table1(sf, "Overall")
    t1.to_csv(TBL_DIR / "table1_overall.csv", index=False)
    print("  [saved] table1_overall.csv")


def _chi2_or_kw(sf, var, groupvar, is_continuous):
    """Return p-value string."""
    groups = [g[var].dropna() for _, g in sf.groupby(groupvar)]
    if is_continuous:
        stat, p = stats.kruskal(*groups)
    else:
        ct = pd.crosstab(sf[var], sf[groupvar])
        stat, p, _, _ = stats.chi2_contingency(ct)
    if p < 0.001:
        return "<0.001"
    return f"{p:.3f}"


def table1_by_subtype(sf):
    """2. table1_by_subtype.csv."""
    frames = {}
    for st in SUBTYPE_ORDER:
        sub = sf[sf["stroke_subtype"] == st]
        frames[SUBTYPE_NICE[st]] = _build_table1(sub, SUBTYPE_NICE[st])

    merged = frames[SUBTYPE_NICE[SUBTYPE_ORDER[0]]][["Variable", "Statistic"]].copy()
    for st in SUBTYPE_ORDER:
        merged = merged.merge(frames[SUBTYPE_NICE[st]][["Variable", SUBTYPE_NICE[st]]],
                              on="Variable", how="left")

    # p-values
    pvals = []
    cont_vars_set = {"Age (years)", "ICU LOS (days)"} | set(LAB_NICE.values())
    for _, row in merged.iterrows():
        vname = row["Variable"]
        if vname == "N":
            pvals.append("")
            continue
        # Determine actual column and whether continuous
        if vname == "Age (years)":
            pvals.append(_chi2_or_kw(sf, "anchor_age", "stroke_subtype", True))
        elif vname == "ICU LOS (days)":
            pvals.append(_chi2_or_kw(sf, "los", "stroke_subtype", True))
        elif vname == "In-hospital mortality":
            pvals.append(_chi2_or_kw(sf, "hospital_expire_flag", "stroke_subtype", False))
        elif vname.startswith("Gender"):
            if "F" in vname:
                pvals.append(_chi2_or_kw(sf, "gender", "stroke_subtype", False))
            else:
                pvals.append("")
        elif vname.startswith("Stroke subtype"):
            pvals.append("")
        elif vname in [COMORB_LABELS[c] for c in COMORBIDITIES]:
            col = [c for c in COMORBIDITIES if COMORB_LABELS[c] == vname][0]
            pvals.append(_chi2_or_kw(sf, col, "stroke_subtype", False))
        elif any(vname == LAB_NICE[c] for c in LAB_NICE):
            col = [c for c in LAB_NICE if LAB_NICE[c] == vname][0]
            pvals.append(_chi2_or_kw(sf, col, "stroke_subtype", True))
        else:
            pvals.append("")

    merged["p-value"] = pvals
    merged.to_csv(TBL_DIR / "table1_by_subtype.csv", index=False)
    print("  [saved] table1_by_subtype.csv")


def table1_by_mortality(sf):
    """3. table1_by_mortality.csv."""
    survived = sf[sf["hospital_expire_flag"] == 0]
    died = sf[sf["hospital_expire_flag"] == 1]
    t_surv = _build_table1(survived, "Survived")
    t_died = _build_table1(died, "Died")
    merged = t_surv.merge(t_died[["Variable", "Died"]], on="Variable", how="left")

    pvals = []
    for _, row in merged.iterrows():
        vname = row["Variable"]
        if vname == "N":
            pvals.append("")
            continue
        if vname == "Age (years)":
            pvals.append(_chi2_or_kw(sf, "anchor_age", "hospital_expire_flag", True))
        elif vname == "ICU LOS (days)":
            pvals.append(_chi2_or_kw(sf, "los", "hospital_expire_flag", True))
        elif vname == "In-hospital mortality":
            pvals.append("")
        elif vname.startswith("Gender"):
            if "F" in vname:
                pvals.append(_chi2_or_kw(sf, "gender", "hospital_expire_flag", False))
            else:
                pvals.append("")
        elif vname.startswith("Stroke subtype"):
            nice = vname.split("— ")[1] if "— " in vname else ""
            inv = {v: k for k, v in SUBTYPE_NICE.items()}
            if nice == SUBTYPE_NICE[SUBTYPE_ORDER[0]]:
                pvals.append(_chi2_or_kw(sf, "stroke_subtype", "hospital_expire_flag", False))
            else:
                pvals.append("")
        elif vname in [COMORB_LABELS[c] for c in COMORBIDITIES]:
            col = [c for c in COMORBIDITIES if COMORB_LABELS[c] == vname][0]
            pvals.append(_chi2_or_kw(sf, col, "hospital_expire_flag", False))
        elif any(vname == LAB_NICE[c] for c in LAB_NICE):
            col = [c for c in LAB_NICE if LAB_NICE[c] == vname][0]
            pvals.append(_chi2_or_kw(sf, col, "hospital_expire_flag", True))
        else:
            pvals.append("")

    merged["p-value"] = pvals
    merged.to_csv(TBL_DIR / "table1_by_mortality.csv", index=False)
    print("  [saved] table1_by_mortality.csv")


def table_comorbidity_cooccurrence(sf):
    """4. comorbidity_cooccurrence_table.csv."""
    labels = [COMORB_LABELS[c] for c in COMORBIDITIES]
    n = len(sf)
    cooc = pd.DataFrame(np.zeros((len(COMORBIDITIES), len(COMORBIDITIES))),
                        index=labels, columns=labels)
    for i, c1 in enumerate(COMORBIDITIES):
        for j, c2 in enumerate(COMORBIDITIES):
            cooc.iloc[i, j] = round((sf[c1] & sf[c2]).sum() / n * 100, 2)
    cooc.to_csv(TBL_DIR / "comorbidity_cooccurrence_table.csv")
    print("  [saved] comorbidity_cooccurrence_table.csv")


def table_cohort_summary(sf):
    """5. cohort_summary_stats.csv."""
    num_cols = ["anchor_age", "los",
                "lab_glucose", "lab_sodium", "lab_creatinine",
                "lab_hemoglobin", "lab_platelets", "lab_inr"]
    rows = []
    for c in num_cols:
        s = sf[c].dropna()
        rows.append({
            "Variable": c,
            "N": len(s),
            "Missing": sf[c].isnull().sum(),
            "Missing%": round(sf[c].isnull().mean() * 100, 1),
            "Mean": round(s.mean(), 2),
            "SD": round(s.std(), 2),
            "Median": round(s.median(), 2),
            "Q1": round(s.quantile(0.25), 2),
            "Q3": round(s.quantile(0.75), 2),
            "Min": round(s.min(), 2),
            "Max": round(s.max(), 2),
        })
    cat_cols = ["gender", "race", "stroke_subtype", "insurance",
                "first_careunit", "admission_type"]
    for c in cat_cols:
        vc = sf[c].value_counts()
        for val, cnt in vc.items():
            rows.append({
                "Variable": f"{c} = {val}",
                "N": cnt,
                "Missing": sf[c].isnull().sum(),
                "Missing%": round(sf[c].isnull().mean() * 100, 1),
                "Mean": round(cnt / len(sf) * 100, 2),
                "SD": np.nan, "Median": np.nan, "Q1": np.nan, "Q3": np.nan,
                "Min": np.nan, "Max": np.nan,
            })
    # Binary
    for c in COMORBIDITIES + ["hospital_expire_flag"]:
        n_pos = int(sf[c].sum())
        rows.append({
            "Variable": c,
            "N": n_pos,
            "Missing": 0,
            "Missing%": 0,
            "Mean": round(n_pos / len(sf) * 100, 2),
            "SD": np.nan, "Median": np.nan, "Q1": np.nan, "Q3": np.nan,
            "Min": np.nan, "Max": np.nan,
        })

    pd.DataFrame(rows).to_csv(TBL_DIR / "cohort_summary_stats.csv", index=False)
    print("  [saved] cohort_summary_stats.csv")


# ===================================================================
# MAIN
# ===================================================================

def main():
    sf, ts = load_data()

    print("\n=== Generating Figures ===")
    fig_demographics(sf)
    fig_stroke_subtypes(sf)
    fig_comorbidities(sf)
    fig_mortality_by_subtype(sf)
    fig_mortality_by_age(sf)
    fig_admission_labs(sf)
    fig_los_distribution(sf)
    fig_sample_trajectories(ts)
    fig_ts_missing(ts)
    fig_correlation_heatmap(sf)
    fig_comorbidity_cooccurrence(sf)
    fig_age_comorbidity_stacked(sf)
    fig_labs_by_mortality(sf)
    fig_gcs_trajectory_by_subtype(sf, ts)
    fig_vital_trends_by_mortality(sf, ts)

    print("\n=== Generating Tables ===")
    table1_overall(sf)
    table1_by_subtype(sf)
    table1_by_mortality(sf)
    table_comorbidity_cooccurrence(sf)
    table_cohort_summary(sf)

    print("\n=== Verifying outputs ===")
    figs = sorted(FIG_DIR.glob("*.png"))
    tbls = sorted(TBL_DIR.glob("*.csv"))
    print(f"  Figures ({len(figs)}):")
    for f in figs:
        size_kb = f.stat().st_size / 1024
        print(f"    {f.name}  ({size_kb:.0f} KB)")
    print(f"  Tables ({len(tbls)}):")
    for f in tbls:
        size_kb = f.stat().st_size / 1024
        print(f"    {f.name}  ({size_kb:.0f} KB)")
    print("\nDone!")


if __name__ == "__main__":
    main()
