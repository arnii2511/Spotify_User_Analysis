# Tableau Calculated Fields

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
