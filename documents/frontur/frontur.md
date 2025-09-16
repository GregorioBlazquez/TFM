# FRONTUR – Data and Forecasting Models

**Source:** Spanish Statistical Office (INE) [Tourist Movement on Borders Survey (FRONTUR)](https://www.ine.es/dyngs/INEbase/en/operacion.htm?c=Estadistica_C&cid=1254736176996&menu=resultados&idp=1254735576863)
**Coverage:** September 2020 – June 2025 (monthly updates)  

---

## Data Structure
- `Period` → monthly timestamp (`YYYY-MM-01`, index).  
- `CCAA` → Autonomous Community (Andalusia, Catalonia, Madrid, Balearic Islands, Canary Islands, Valencia, OtherCCAA, Total (for Spain)).  
- `Total` → number of inbound tourists (float, tourists/month).  

---

## Key Patterns
- **Strong annual seasonality**: peaks in summer, lows in winter.  
- **Regions with strong seasonality**: Catalonia, Balearic Islands, Andalusia, Valencia (similar to national pattern).  
- **Regions with stable flows**: Madrid and Canary Islands (flatter curves, less dependent on season).  
- **Pandemic recovery**: sharp rise in 2020–21, then stabilization; slight upward trend in 2025.  

---

## Modeling
- **Main approach:** Seasonal ARIMA (`m=12`) via `auto_arima`.  
- **Performance:** ARIMA captures trend + seasonality best; confidence intervals widen sharply beyond 6 months.  
- Models created for **Spain (Total)** + each **CCAA**.  
- File format: `models/arima_model_<Region>.pkl`.  

## Limitations
- Data starts in 09/2020 → influenced by post-pandemic.
- Uncertainty grows beyond 6 months.
- No external explanatory variables (pure time series).
