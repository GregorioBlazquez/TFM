# EGATUR EDA & Clustering Summary

## Exploratory Data Analysis (EDA)

- **Numerical variables:** Right skew, big outliers in expenditure and trip length.  
- **Categorical variables:** Trips concentrated in Catalonia, Balearic Islands, Andalusia, Canary Islands, Valencian Community, Madrid.  
- **Bivariate findings:**  
  - High spenders differ by metric (total vs daily vs weighted).  
  - Negative correlation between trip length and daily expenditure.  
- **Temporal:** Some seasonality but not dominant.  

## Clustering (K-means, 4 groups)

1. **Cluster 0 – Long-stay visitors (Valencian Community)**  
   - Longer trips, medium/low daily expenditure  
   - Non-market accommodation  
   - Leisure + personal purposes  

2. **Cluster 1 – Standard tourists (Island vacationers)**  
   - Balearic & Canary Islands  
   - Mostly hotels, leisure trips  
   - Strong summer seasonality  

3. **Cluster 2 – Personal/family visits**  
   - Long stays, non-market accommodation  
   - Purpose “Other” (friends/family)  
   - Spread across regions, more winter trips  

4. **Cluster 3 – Urban, high-spending tourists**  
   - Short trips, high daily expenditure  
   - Hotels, Catalonia & Madrid  
   - Mix leisure/business, strong presence of Russia + Rest of the world  

## Key Insights

- High dispersion & outliers → predictive modeling is challenging.  
- Clusters are useful for segmentation and as features.  
