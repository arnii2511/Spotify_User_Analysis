# User Data Quality & Conversion Analytics Dashboard

Tableau-first analytics project analyzing 28K+ Spotify user interaction records to identify data quality issues, behavioral patterns, and conversion signals associated with premium subscription intent.

## Project Summary

This project presents a stakeholder-ready Tableau analytics workflow supported by Python-generated extracts and SQL validation checks. The analysis focuses on four questions:

- Which user segments show the strongest premium conversion intent?
- Which plan, listening, recommendation, genre, and device patterns explain conversion differences?
- Are the core reporting fields complete and valid enough for stakeholder reporting?
- What actions can be recommended from the observed user interaction data?

## Main Deliverable

The primary deliverable is the Tableau workbook:

```text
tableau/User_Data_Quality_Conversion_Analytics.twb
```

Workbook contents:

- 13 Tableau worksheets.
- 5 analytical Tableau dashboards.
- 1 Tableau storyboard dashboard.
- Connected extracts for KPI, conversion, segment, quality, genre, and device analysis.

## Key Metrics

- Total records analyzed: `28,546`
- Premium conversion intent: `43.9%`
- High-intent users: `3,065`
- High-intent conversion rate: `49.9%`
- Data completeness rate: `93.5%`

## Analytical Views

The workbook is organized around:

- Executive KPI overview.
- Premium conversion by plan, listening frequency, recommendation rating, and time slot.
- Segment-wise premium conversion and customer profile analysis.
- Data quality validation checks for completeness, rating bounds, and binary target consistency.
- Genre demand and device footprint analysis.
- Recommendation-oriented dashboard views for stakeholder storytelling.

## Repository Structure

```text
spotify-analysis/
|-- data/
|   |-- raw/Spotify_user_research.xlsx
|   `-- processed/
|       |-- spotify_cleaned.csv
|       `-- spotify_cleaned_final.csv
|-- notebooks/
|   `-- music-analysis.ipynb
|-- reports/
|   |-- figures/viz_*.png
|   |-- tableau_exports/
|   `-- quality/data_quality_report.md
|-- sql/
|   `-- validation_checks.sql
|-- src/
|   `-- pipeline/
|       |-- build_tableau_assets.py
|       `-- connect_tableau_sources.py
`-- tableau/
    |-- User_Data_Quality_Conversion_Analytics.twb
    |-- docs/calculated_fields.md
    |-- docs/tableau_story_build_plan.md
    `-- extracts/
        |-- spotify_user_conversion_fact.csv
        |-- kpi_summary.csv
        |-- conversion_by_plan.csv
        |-- conversion_by_frequency.csv
        |-- conversion_by_recommendation_rating.csv
        |-- conversion_by_time_slot.csv
        |-- segment_scorecard.csv
        |-- top_genres.csv
        |-- device_mix.csv
        |-- story_points.csv
        `-- data_quality_checks.csv
```

## Data Pipeline

Python scripts in `src/pipeline/` generate Tableau-ready CSV extracts from the processed dataset:

- `spotify_user_conversion_fact.csv`: fact-level analytical extract.
- `kpi_summary.csv`: KPI metrics for dashboard cards.
- `conversion_by_*.csv`: pre-aggregated conversion views.
- `segment_scorecard.csv`: segment-wise conversion and profile summary.
- `data_quality_checks.csv`: validation check output.
- `top_genres.csv` and `device_mix.csv`: ranked distribution views.

## Data Quality Layer

The project includes a validation layer for:

- Required field completeness.
- Recommendation rating bounds from 1 to 5.
- Binary premium conversion target consistency.
- Segment, plan, genre, and device reporting readiness.

Supporting files:

```text
reports/quality/data_quality_report.md
sql/validation_checks.sql
tableau/docs/calculated_fields.md
```

## Key Findings

- Family, individual, and student plan cohorts show the strongest premium conversion rates, clustered around 49-50%.
- Segment 0 is the strongest sampled segment, with 6,508 users and 49.1% conversion.
- Context-specific listening responses underperform, indicating a need to separate frequency from listening occasion in future data collection.
- Melody, rap, pop, and classical are the leading genre categories by record count.
- Smart speakers, smartphones, laptops, and tablets form the dominant listening-device footprint.

## Recommendations

- Prioritize premium campaigns toward family, individual, and student plan cohorts.
- Treat Segment 0 as the strongest initial audience for targeted premium messaging.
- Standardize listening-frequency fields so frequency and listening occasion are not mixed.
- Apply genre and device footprint patterns to campaign creative and product nudges.
- Keep data quality checks visible in stakeholder reporting to make insights more defensible.
