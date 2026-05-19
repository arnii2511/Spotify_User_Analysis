-- Validation checks that mirror the Tableau data quality view.
-- Replace spotify_user_conversion_fact with your warehouse table name.

SELECT
  COUNT(*) AS total_records,
  SUM(CASE WHEN Age IS NULL THEN 1 ELSE 0 END) AS missing_age,
  SUM(CASE WHEN Gender IS NULL THEN 1 ELSE 0 END) AS missing_gender,
  SUM(CASE WHEN spotify_listening_device IS NULL THEN 1 ELSE 0 END) AS missing_device,
  SUM(CASE WHEN spotify_subscription_plan IS NULL THEN 1 ELSE 0 END) AS missing_subscription_plan,
  SUM(CASE WHEN premium_sub_willingness IS NULL THEN 1 ELSE 0 END) AS missing_premium_willingness,
  SUM(CASE WHEN music_lis_frequency IS NULL THEN 1 ELSE 0 END) AS missing_music_frequency,
  SUM(CASE WHEN music_recc_rating IS NULL THEN 1 ELSE 0 END) AS missing_recommendation_rating,
  SUM(CASE WHEN primary_genre IS NULL THEN 1 ELSE 0 END) AS missing_primary_genre,
  SUM(CASE WHEN plan IS NULL THEN 1 ELSE 0 END) AS missing_plan,
  SUM(CASE WHEN music_recc_rating NOT BETWEEN 1 AND 5 THEN 1 ELSE 0 END) AS invalid_recommendation_rating,
  SUM(CASE WHEN premium_yes NOT IN (0, 1) THEN 1 ELSE 0 END) AS invalid_premium_target
FROM spotify_user_conversion_fact;

SELECT
  plan,
  COUNT(*) AS records,
  AVG(premium_yes) AS premium_conversion_rate
FROM spotify_user_conversion_fact
GROUP BY plan
ORDER BY premium_conversion_rate DESC;

SELECT
  segment_label,
  COUNT(*) AS records,
  AVG(premium_yes) AS premium_conversion_rate,
  AVG(music_recc_rating) AS avg_recommendation_rating
FROM spotify_user_conversion_fact
WHERE segment_label <> 'Unsampled'
GROUP BY segment_label
ORDER BY premium_conversion_rate DESC;
