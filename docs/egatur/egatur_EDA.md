# EGATUR – Key Insights (EDA)

- **Trip length vs. spending:** Longer trips → lower daily expenditure. Short trips → high-intensity spenders.
- **Regional concentration:** 6 regions account for most trips: Catalonia, Balearic Islands, Andalusia, Canary Islands, Valencia, Madrid.
- **Country effects:** Tourists from Russia & Rest of the World spend significantly more per day.
- **Outliers:** High spenders differ by metric (total, weighted, daily). Extreme trips are not consistent across measures.
- **Clusters:** Segmentation shows clear differences in purpose, accommodation, and geography.
- **SHAP base prediction:** The base prediction for SHAP values, which predicts nearly the mean expenditure, yields an average of €233.9 average daily expenditure.

# EGATUR – Tourist Clusters (EDA)

Based on clustering analysis of EGATUR microdata (2022–2025), four main tourist profiles were identified:

1. **Cluster 0 – Long-stay visitors in the Valencian Community**
   - Longer trips, medium-to-low daily expenditure.
   - Non-market accommodation common.
   - Mix of leisure and personal purposes.

2. **Cluster 1 – Standard tourists (Island vacationers)**
   - Balearic & Canary Islands, hotels.
   - Mostly summer leisure trips.

3. **Cluster 2 – Personal/family visits**
   - Long stays, non-market accommodation.
   - Purpose: “Other” (visiting relatives, personal reasons).
   - Spread across multiple regions.

4. **Cluster 3 – Urban, high-spending international tourists**
   - Catalonia & Madrid.
   - Hotels, short stays, high daily expenditure.
   - Mix of leisure and business.
   - Higher share of Russia + Rest of the World.
