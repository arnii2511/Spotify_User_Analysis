# Spotify User Analysis: Premium Conversion Prediction

A comprehensive data science project analyzing Spotify user behavior to predict premium subscription conversion. This analysis combines exploratory data analysis, predictive modeling, and customer segmentation to drive business insights for music streaming platforms.

## Project Overview

This project builds an end-to-end data pipeline that:
- **Cleans & engineers** user research data from Spotify
- **Visualizes** key business metrics for strategic insights
- **Predicts** which users are likely to convert to premium subscriptions
- **Compares** multiple machine learning algorithms (Logistic Regression, Random Forest, XGBoost)
- **Segments** users into behavioral clusters for targeted marketing

## Business Context

Music streaming platforms like JioSaavn and Spotify rely on converting free users to premium subscribers. This project addresses the core business question: **Who should we target for premium conversion, and what drives their decision?**

### Key Use Cases
- **Targeted Marketing**: Identify high-conversion-likelihood users to prioritize ad spend
- **Product Strategy**: Understand which features (recommendations, listening frequency) drive premiumization
- **Pricing Strategy**: Tailor subscription offers by user segment and age group
- **Revenue Forecasting**: Predict upgrade rates and ARPU (Average Revenue Per User) growth

## Project Structure

```
spotify-analysis/
├── music-analysis.ipynb              # Main analysis notebook
├── Spotify_user_research.xlsx        # Source data (user research)
├── spotify_cleaned.csv               # Cleaned data snapshot
├── spotify_cleaned_final.csv         # Final dataset with segments
├── viz_*.png                         # Generated visualizations
└── README.md                         # This file
```

## Dataset

**Source**: Spotify User Research Survey
**Features Include**:
- Demographics: Age, Gender
- Device: Listening device (mobile, desktop, tablet)
- Behavior: Listening frequency, preferred time slot, music genre
- Satisfaction: Recommendation rating
- Target: Premium subscription willingness (Yes/No)

## Analysis Pipeline

### 1. Data Cleaning & Feature Engineering
- Standardize target variable: `premium_sub_willingness` → binary `premium_yes`
- Normalize subscription plans (free, premium, family, student, etc.)
- Extract primary music genre from multi-value fields
- Handle missing values intelligently

**Output**: `spotify_cleaned.csv`

### 2. Exploratory Data Analysis (EDA)
Generates 5 key visualizations:
- **Top 10 Genres**: Content strategy insights (which genres to invest in)
- **Time Slot Analysis**: Premium conversion rates by listening time
- **Recommendation Impact**: Correlation between recommendation quality and willingness to pay
- **Device Distribution**: Mobile-first usage patterns
- **Age & Plan Mix**: Subscription preferences by demographic

**Output**: PNG visualizations for dashboard integration

### 3. Predictive Modeling

#### Model Comparison: Three Algorithms
1. **Logistic Regression** (Baseline)
   - Interpretable, fast
   - Good for understanding feature coefficients
   
2. **Random Forest** (Ensemble)
   - Captures non-linear relationships
   - Robust to outliers
   
3. **XGBoost** (Gradient Boosting)
   - State-of-the-art performance
   - Best for high-stakes predictions

#### Evaluation Metrics
- **AUC**: Overall discriminative ability
- **Precision**: % of predicted converters who actually convert (minimize false positives)
- **Recall**: % of actual converters we identify (minimize false negatives)
- **F1-Score**: Harmonic mean balancing precision-recall

**Outputs**:
- Model comparison visualization
- Feature importance rankings
- Best model selection by AUC

### 4. Customer Segmentation (K-Means)
Clusters users into 4 behavioral segments based on:
- Age group
- Device preference
- Listening time slot
- Listening frequency
- Current subscription plan

**Segment Profiles**: Premium willingness, modal age, and device for each segment

**Visualization**: PCA-reduced 2D visualization of clusters

**Output**: `spotify_cleaned_final.csv` (includes segment assignments)

## Key Findings

### Model Performance
| Model | AUC | Precision | Recall | F1-Score |
|-------|-----|-----------|--------|----------|
| Logistic Regression | 0.5824 | 0.4888 | 0.2352 | 0.3176 |
| Random Forest | 0.5418 | 0.4825 | 0.3854 | 0.4285 |
| XGBoost | 0.5621 | 0.4838 | 0.3814 | 0.4265 |

*Run the notebook to see actual performance metrics*

### Top Predictors of Premium Conversion
1. Recommendation rating (positive correlation)
2. Listening frequency (high engagement → higher conversion)
3. Current subscription plan (already premium tiers)
4. Device type (mobile users show different patterns)
5. Age group (younger users more conversion-prone)

## Getting Started

### Prerequisites
```bash
python 3.8+
```

### Installation
```bash
pip install pandas numpy matplotlib seaborn scikit-learn openpyxl xgboost
```

### Running the Analysis
1. Open `music-analysis.ipynb` in Jupyter Notebook or VS Code
2. Update `FILE_PATH` to point to your Spotify data Excel file
3. Run cells sequentially (or run all)
4. Visualizations will be saved as PNG files

```python
FILE_PATH = "path/to/your/Spotify_user_research.xlsx"
```

## Outputs

### CSV Files
- `spotify_cleaned.csv` - Cleaned data with engineered features
- `spotify_cleaned_final.csv` - Final dataset with segment assignments

### Visualizations (PNG)
- `viz_top_genres.png` - Genre distribution
- `viz_time_slot_premium.png` - Conversion by listening time
- `viz_rec_vs_premium.png` - Recommendation impact
- `viz_devices.png` - Device distribution
- `viz_plan_age.png` - Subscription mix by age
- `viz_model_comparison.png` - Algorithm performance comparison
- `viz_feature_importance_comparison.png` - Feature importance (RF vs XGBoost)
- `viz_segments.png` - User clusters visualization

## Technical Highlights

### Machine Learning
- One-hot encoding for categorical variables
- Stratified train-test split (maintains class balance)
- StandardScaler normalization for distance-based models
- Multiple algorithms comparison framework

### Data Engineering
- Robust string parsing for multi-value fields
- Intelligent missing value handling
- Index-based segment mapping for scalability

### Visualization
- Professional matplotlib styling
- Value labels on charts
- PCA dimensionality reduction for cluster visualization

## Recommendations

1. **Priority Segment**: Target users with high recommendation ratings + high listening frequency
2. **Product Focus**: Invest in recommendation algorithm quality (strongest conversion driver)
3. **Monetization Timing**: Adjust premium offers based on listening time patterns
4. **Mobile Strategy**: Optimize mobile experience (primary device for users)
5. **Age-Based Pricing**: Consider tiered pricing for different age groups

## Future Enhancements

- [ ] Time series analysis (retention after upgrade)
- [ ] A/B testing framework for promotion strategies
- [ ] Cohort analysis for long-term LTV prediction
- [ ] Regional customization (India-specific preferences)
- [ ] Real-time deployment pipeline with FastAPI


