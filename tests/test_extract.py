import pytest
from pathlib import Path


def test_extract_stroke_cohort_returns_dataframe():
    from src.data.extract import extract_stroke_cohort
    import yaml

    with open("config/config.yaml") as f:
        config = yaml.safe_load(f)

    df = extract_stroke_cohort(config)
    assert len(df) > 0
    assert "subject_id" in df.columns
    assert "hadm_id" in df.columns
    assert "stay_id" in df.columns
    assert "icd_code" in df.columns
    # Abdollahi 2025 found ~3,487; expect 2,000-6,000 range
    assert 1000 < len(df) < 10000, f"Unexpected cohort size: {len(df)}"


def test_no_duplicate_stays():
    from src.data.extract import extract_stroke_cohort
    import yaml

    with open("config/config.yaml") as f:
        config = yaml.safe_load(f)

    df = extract_stroke_cohort(config)
    # One row per patient (first ICU stay)
    assert df["subject_id"].is_unique, (
        f"Duplicate subject_ids found: {df['subject_id'].duplicated().sum()}"
    )
