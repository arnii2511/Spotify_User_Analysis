# User Data Quality & Conversion Analytics Dashboard

Tableau-first analytics project for profiling 28K+ Spotify user records, identifying data quality gaps, and explaining behavioral indicators linked to premium conversion.

## Main Deliverable

Open this workbook in Tableau:

```text
tableau/User_Data_Quality_Conversion_Analytics.twb
```

The workbook is connected to the Tableau-ready extract:

```text
tableau/extracts/spotify_user_conversion_fact.csv
```

Current workbook state:

- The existing workbook has 3 starter worksheets and 1 starter dashboard.
- The connected extracts support the full story build.
- The target portfolio build should be 12+ worksheets, 5 dashboards, and 1 Tableau Story.

## Project Structure

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

## Rebuild Tableau Extracts

```bash
python src/pipeline/build_tableau_assets.py
python src/pipeline/connect_tableau_sources.py
```

This creates:

- `spotify_user_conversion_fact.csv` for the main Tableau workbook.
- `kpi_summary.csv` for KPI cards.
- `conversion_by_plan.csv`, `conversion_by_frequency.csv`, `conversion_by_recommendation_rating.csv`, and `conversion_by_time_slot.csv` for conversion lever charts.
- `segment_scorecard.csv` for segment-wise conversion analysis.
- `top_genres.csv` and `device_mix.csv` for content/product footprint charts.
- `story_points.csv` for Tableau Story captions and narrative sequence.
- `data_quality_checks.csv` for validation and exception reporting.
- `reports/quality/data_quality_report.md` for GitHub documentation.

`connect_tableau_sources.py` wires those extracts into the workbook data pane so they appear as connected data sources when the `.twb` opens.

## Tableau Build Note

Tableau visual authoring must happen inside Tableau Desktop: build worksheets first, combine them into dashboards, then add dashboards to a Story.

The KPI views are built from `tableau/extracts/kpi_summary.csv`, especially these metrics:

- `Total user records`
- `Premium conversion rate`
- `High-intent users`
- `Data completeness rate`

Use this guide for the exact worksheet -> dashboard -> story build:

```text
tableau/docs/tableau_story_build_plan.md
```

## Tableau Views

The workbook is organized around:

- Premium conversion levers by time slot and recommendation rating.
- High-value customer segmentation.
- Content, genre, and device/product usage views.
- Data quality and validation support files for completeness and exception checks.

Calculated field references are documented in:

```text
tableau/docs/calculated_fields.md
```

## SQL Support

SQL validation checks are in:

```text
sql/validation_checks.sql
```

These checks mirror the data quality layer used for Tableau reporting.

## Resume Framing

**User Data Quality & Conversion Analytics Dashboard - Github | Python, NumPy, Tableau, SQL**

- Analyzed 28K+ user records to identify data patterns, inconsistent fields, and behavioral indicators linked to premium conversion.
- Built a Tableau dashboard with KPI views for engagement, subscription conversion trends, segment-wise analysis, and data quality observations.
- Defined analytical metrics and validation checks for clearer reporting, exception identification, and stakeholder-friendly storytelling.
- Refined the analysis around plan conversion, listening frequency, recommendation ratings, segments, genre demand, device footprint, and data quality validation.
