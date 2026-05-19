from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "data" / "processed" / "spotify_cleaned_final.csv"
EXTRACT_DIR = ROOT / "tableau" / "extracts"
DOC_DIR = ROOT / "tableau" / "docs"
QUALITY_DIR = ROOT / "reports" / "quality"


FREQUENCY_BANDS = {
    "Never": "Low",
    "Rarely": "Low",
    "Once a week": "Medium",
    "Several times a week": "High",
    "Daily": "High",
}


def pct(series: pd.Series) -> float:
    return round(float(series.mean()) * 100, 1)


def normalize_frequency(value: object) -> str:
    if pd.isna(value):
        return "Unknown"
    label = str(value).strip()
    return FREQUENCY_BANDS.get(label, "Context-specific")


def mode_or_unknown(series: pd.Series) -> str:
    modes = series.dropna().astype(str).str.strip().mode()
    return modes.iat[0] if not modes.empty else "Unknown"


def build_fact(df: pd.DataFrame) -> pd.DataFrame:
    fact = df.copy()
    fact["frequency_band"] = fact["music_lis_frequency"].map(normalize_frequency)
    fact["recommendation_band"] = pd.cut(
        fact["music_recc_rating"],
        bins=[0, 2, 3, 5],
        labels=["Low rating", "Neutral rating", "High rating"],
        include_lowest=True,
    ).astype(str)
    fact["is_high_intent"] = (
        (fact["premium_yes"].eq(1))
        & (fact["music_recc_rating"].ge(4))
        & (fact["frequency_band"].eq("High"))
    ).astype(int)
    fact["record_count"] = 1
    return fact


def quality_checks(df: pd.DataFrame) -> pd.DataFrame:
    required = [
        "Age",
        "Gender",
        "spotify_listening_device",
        "spotify_subscription_plan",
        "premium_sub_willingness",
        "music_lis_frequency",
        "music_recc_rating",
        "primary_genre",
        "plan",
    ]
    rows = []
    for column in required:
        affected = int(df[column].isna().sum())
        rows.append(
            {
                "check_name": f"{column} completeness",
                "status": "Pass" if affected == 0 else "Review",
                "affected_rows": affected,
                "validity_rate": round((1 - affected / len(df)) * 100, 1),
            }
        )

    valid_rating = int(df["music_recc_rating"].between(1, 5).sum())
    rows.append(
        {
            "check_name": "Recommendation rating between 1 and 5",
            "status": "Pass" if valid_rating == len(df) else "Review",
            "affected_rows": len(df) - valid_rating,
            "validity_rate": round(valid_rating / len(df) * 100, 1),
        }
    )

    valid_target = int(df["premium_yes"].isin([0, 1]).sum())
    rows.append(
        {
            "check_name": "Premium conversion target is binary",
            "status": "Pass" if valid_target == len(df) else "Review",
            "affected_rows": len(df) - valid_target,
            "validity_rate": round(valid_target / len(df) * 100, 1),
        }
    )
    return pd.DataFrame(rows)


def segment_scorecard(df: pd.DataFrame) -> pd.DataFrame:
    sampled = df[df["segment_label"].ne("Unsampled")].copy()
    return (
        sampled.groupby("segment_label", dropna=False)
        .agg(
            records=("premium_yes", "size"),
            premium_conversion_rate=("premium_yes", lambda s: round(s.mean() * 100, 1)),
            avg_recommendation_rating=("music_recc_rating", lambda s: round(s.mean(), 2)),
            top_device=("spotify_listening_device", mode_or_unknown),
            top_plan=("plan", mode_or_unknown),
        )
        .reset_index()
        .sort_values("premium_conversion_rate", ascending=False)
    )


def conversion_view(df: pd.DataFrame, column: str, output_label: str) -> pd.DataFrame:
    return (
        df.groupby(column, dropna=False)
        .agg(records=("premium_yes", "size"), premium_conversion_rate=("premium_yes", lambda s: round(s.mean() * 100, 1)))
        .reset_index()
        .rename(columns={column: output_label})
        .sort_values("premium_conversion_rate", ascending=False)
    )


def rank_view(df: pd.DataFrame, column: str, output_label: str, limit: int = 8) -> pd.DataFrame:
    counts = (
        df[column]
        .fillna("Unknown")
        .astype(str)
        .str.strip()
        .value_counts()
        .head(limit)
        .reset_index()
    )
    counts.columns = [output_label, "records"]
    counts["share_of_max"] = round(counts["records"] / counts["records"].max() * 100, 1)
    return counts


def kpi_summary(df: pd.DataFrame) -> pd.DataFrame:
    high_intent = df[(df["music_recc_rating"].ge(4)) & (df["frequency_band"].eq("High"))]
    completeness = round((1 - (df.isna().sum().sum() / (df.shape[0] * df.shape[1]))) * 100, 1)
    return pd.DataFrame(
        [
            {"metric": "Total user records", "value": len(df), "display_value": f"{len(df):,}"},
            {
                "metric": "Premium conversion rate",
                "value": pct(df["premium_yes"]),
                "display_value": f"{pct(df['premium_yes'])}%",
            },
            {
                "metric": "High-intent users",
                "value": len(high_intent),
                "display_value": f"{len(high_intent):,}",
            },
            {
                "metric": "High-intent conversion rate",
                "value": pct(high_intent["premium_yes"]) if len(high_intent) else 0,
                "display_value": f"{pct(high_intent['premium_yes']) if len(high_intent) else 0}%",
            },
            {
                "metric": "Average recommendation rating",
                "value": round(float(df["music_recc_rating"].mean()), 2),
                "display_value": str(round(float(df["music_recc_rating"].mean()), 2)),
            },
            {
                "metric": "Data completeness rate",
                "value": completeness,
                "display_value": f"{completeness}%",
            },
        ]
    )


def story_points() -> pd.DataFrame:
    rows = [
        {
            "story_point": 1,
            "title": "Executive Overview",
            "headline": "28K+ user records show a 43.9% premium intent baseline.",
            "caption": "Start with the KPI view: total user volume, premium intent, high-intent cohort, and data completeness.",
            "primary_extract": "kpi_summary.csv",
        },
        {
            "story_point": 2,
            "title": "Conversion Levers",
            "headline": "Family, individual, and student plans show the strongest premium conversion signals.",
            "caption": "Use plan, listening frequency, and recommendation rating views to identify where conversion is strongest.",
            "primary_extract": "conversion_by_plan.csv",
        },
        {
            "story_point": 3,
            "title": "Segment and Data Quality",
            "headline": "Segment 0 is the strongest sampled segment, while validation checks support stakeholder trust.",
            "caption": "Pair behavioral segmentation with completeness and validity checks to show that insights are backed by governed data.",
            "primary_extract": "segment_scorecard.csv",
        },
        {
            "story_point": 4,
            "title": "Content and Device Footprint",
            "headline": "Melody, rap, and pop lead genre demand; smart speakers and smartphones lead listening footprint.",
            "caption": "Close with content and device mix to translate analytics into product and campaign decisions.",
            "primary_extract": "top_genres.csv",
        },
        {
            "story_point": 5,
            "title": "Recommended Actions",
            "headline": "Prioritize high-conversion plan cohorts, clean context-specific listening fields, and tailor campaigns by device footprint.",
            "caption": "Close the story with practical next steps from the 28K-row dataset: target family/student/individual plan interest, improve ambiguous listening-frequency capture, and use genre/device mix for campaign planning.",
            "primary_extract": "story_points.csv",
        },
    ]
    return pd.DataFrame(rows)


def write_calculated_fields_doc() -> None:
    content = """# Tableau Calculated Fields

Use these fields in Tableau if the workbook needs to be rebuilt from the extract.

## Premium Conversion Rate

```tableau
AVG([premium_yes])
```

Format as percentage.

## Record Count

```tableau
SUM([record_count])
```

## High Intent User

```tableau
IF [music_recc_rating] >= 4 AND [frequency_band] = "High" THEN 1 ELSE 0 END
```

## High Intent Conversion Rate

```tableau
SUM(IF [is_high_intent] = 1 THEN [premium_yes] END)
/
SUM(IF [is_high_intent] = 1 THEN [record_count] END)
```

## Data Quality Pass Rate

```tableau
AVG([validity_rate]) / 100
```
"""
    (DOC_DIR / "calculated_fields.md").write_text(content, encoding="utf-8")


def main() -> None:
    EXTRACT_DIR.mkdir(parents=True, exist_ok=True)
    DOC_DIR.mkdir(parents=True, exist_ok=True)
    QUALITY_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(SOURCE)
    df["music_recc_rating"] = pd.to_numeric(df["music_recc_rating"], errors="coerce")
    fact = build_fact(df)
    checks = quality_checks(df)

    fact.to_csv(EXTRACT_DIR / "spotify_user_conversion_fact.csv", index=False)
    kpi_summary(fact).to_csv(EXTRACT_DIR / "kpi_summary.csv", index=False)
    segment_scorecard(fact).to_csv(EXTRACT_DIR / "segment_scorecard.csv", index=False)
    checks.to_csv(EXTRACT_DIR / "data_quality_checks.csv", index=False)
    conversion_view(fact, "plan", "plan").to_csv(EXTRACT_DIR / "conversion_by_plan.csv", index=False)
    conversion_view(fact, "frequency_band", "frequency_band").to_csv(EXTRACT_DIR / "conversion_by_frequency.csv", index=False)
    conversion_view(fact, "music_recc_rating", "recommendation_rating").to_csv(
        EXTRACT_DIR / "conversion_by_recommendation_rating.csv", index=False
    )
    conversion_view(fact, "music_time_slot", "music_time_slot").to_csv(EXTRACT_DIR / "conversion_by_time_slot.csv", index=False)
    rank_view(fact, "primary_genre", "genre").to_csv(EXTRACT_DIR / "top_genres.csv", index=False)
    rank_view(fact, "spotify_listening_device", "device").to_csv(EXTRACT_DIR / "device_mix.csv", index=False)
    story_points().to_csv(EXTRACT_DIR / "story_points.csv", index=False)
    quality_lines = [
        "# Data Quality Report",
        "",
        f"Records profiled: {len(df):,}",
        "",
        "| Check | Status | Validity rate | Affected rows |",
        "|---|---:|---:|---:|",
    ]
    for row in checks.to_dict("records"):
        quality_lines.append(
            f"| {row['check_name']} | {row['status']} | {row['validity_rate']}% | {row['affected_rows']:,} |"
        )
    (QUALITY_DIR / "data_quality_report.md").write_text("\n".join(quality_lines) + "\n", encoding="utf-8")
    write_calculated_fields_doc()


if __name__ == "__main__":
    main()
