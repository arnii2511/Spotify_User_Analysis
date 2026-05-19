# Tableau Story Build Plan

This follows Tableau's actual workflow:

```text
Worksheets -> Dashboards -> Story
```

The workbook already contains the connected data extracts. Build the following worksheets in Tableau Desktop, then combine them into dashboards, then add those dashboards to a Story.

Target build:

- 12 worksheets minimum.
- 5 dashboards.
- 1 Tableau Story containing the 5 dashboards as story points.

Three worksheets and one dashboard are not enough for the portfolio/resume version.

## Worksheets to Create

### KPI Worksheets

Data source: `KPI Summary`

Create one worksheet per KPI:

- `KPI - Total Records`: filter `metric` to `Total user records`, put `display_value` on Text.
- `KPI - Premium Intent`: filter `metric` to `Premium conversion rate`, put `display_value` on Text.
- `KPI - High Intent Users`: filter `metric` to `High-intent users`, put `display_value` on Text.
- `KPI - Data Completeness`: filter `metric` to `Data completeness rate`, put `display_value` on Text.

### Conversion Worksheets

Data sources:

- `Conversion by Plan`
- `Conversion by Frequency`
- `Conversion by Recommendation Rating`
- `Conversion by Time Slot`

Create:

- `WS - Conversion by Plan`: `plan` on Rows, `premium_conversion_rate` on Columns, bar chart, descending sort.
- `WS - Conversion by Frequency`: `frequency_band` on Rows, `premium_conversion_rate` on Columns.
- `WS - Conversion by Recommendation`: `recommendation_rating` on Rows, `premium_conversion_rate` on Columns.
- `WS - Conversion by Time Slot`: `music_time_slot` on Rows, `premium_conversion_rate` on Columns.
- `WS - Plan Records`: `plan` on Rows, `records` on Columns, bar chart. Use this as a supporting volume view.

### Segment and Quality Worksheets

Data sources:

- `Segment Scorecard`
- `Data Quality Checks`

Create:

- `WS - Segment Scorecard`: table with `segment_label`, `records`, `premium_conversion_rate`, `top_device`, `top_plan`.
- `WS - Quality Checks`: table with `check_name`, `validity_rate`, `affected_rows`, `status`.
- `WS - Segment Conversion Bars`: `segment_label` on Rows, `premium_conversion_rate` on Columns.

### Content and Device Worksheets

Data sources:

- `Top Genres`
- `Device Mix`

Create:

- `WS - Top Genres`: `genre` on Rows, `records` on Columns, bar chart, descending sort.
- `WS - Device Mix`: `device` on Rows, `records` on Columns, bar chart, descending sort.
- `WS - Genre Share`: `genre` on Rows, `share_of_max` on Columns.

## Dashboards to Create

### Dashboard 1: Executive KPI Overview

Use:

- `KPI - Total Records`
- `KPI - Premium Intent`
- `KPI - High Intent Users`
- `KPI - Data Completeness`

Add title:

```text
User Data Quality & Conversion Analytics Dashboard
```

Add subtitle:

```text
KPI views for 28,546 user records, premium conversion trends, segment signals, and validation checks.
```

### Dashboard 2: Conversion Levers

Use:

- `WS - Conversion by Plan`
- `WS - Conversion by Frequency`
- `WS - Conversion by Recommendation`
- `WS - Conversion by Time Slot`

Insight caption:

```text
Family, individual, and student plan cohorts show the strongest premium conversion signals.
```

### Dashboard 3: Segment and Data Quality

Use:

- `WS - Segment Scorecard`
- `WS - Quality Checks`

Insight caption:

```text
Segment 0 leads sampled conversion, while validation checks show core analytical fields are complete and bounded.
```

### Dashboard 4: Content and Device Footprint

Use:

- `WS - Top Genres`
- `WS - Device Mix`

Insight caption:

```text
Melody, rap, and pop lead genre demand; smart speakers and smartphones lead listening footprint.
```

### Dashboard 5: Recommendations

Use text objects and optionally add mini versions of plan, genre, and device sheets.

Recommended actions:

```text
1. Prioritize family, individual, and student plan cohorts.
2. Standardize context-specific listening-frequency responses.
3. Use genre and device footprint to tailor campaign and product nudges.
```

## Story to Create

In Tableau Desktop:

1. Click `New Story`.
2. Drag Dashboard 1 into the first story point.
3. Add a caption: `Executive KPI overview`.
4. Click `Blank`, then drag Dashboard 2.
5. Repeat for Dashboards 3, 4, and 5.

Official Tableau flow: Tableau stories are collections of sheets/dashboards arranged as story points; dashboards are built by dragging worksheets into a dashboard.
