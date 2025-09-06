# EGATUR – Predictive Models

## Cluster Classifier
- Logistic regression trained to replicate EDA clusters.
- Accuracy >99%.
- Assigns new tourist profiles to one of 4 clusters.

## Expenditure Regressor
- HistGradientBoosting model predicts daily average expenditure.
- Key predictors:
  - **Trip length** (negative exponential effect).
  - **Country of origin** (Russia/Rest of the World = +200€/day).
- Minor factors: accommodation, region.
- Negligible: purpose, season.

## Explainability
- SHAP values show how features affect predictions.
- Transparency: can explain both global patterns and individual cases.
