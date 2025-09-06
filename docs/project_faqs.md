# Tourism Assistant FAQs

## What is this project?

This assistant is part of the Master’s Thesis (TFM) of **Gregorio Blázquez Martínez** (author of this project).  
It combines **predictive models (ARIMA)** with **retrieval-augmented generation (RAG)** on official tourism documents (INE reports such as FRONTUR and EGATUR, and complementary materials).

The goal is to explore how AI can help **analyze, explain and forecast tourism trends in Spain**.

## Is the data official or generated?

- **Official data**: When referring to FRONTUR, EGATUR or other INE reports, the assistant retrieves real numbers and text from official sources.  
- **Generated data**: Predictions or explanations outside the reports are created using AI models trained with historical series. These are **estimates, not official figures**.

## How can I use the assistant?

You can ask two types of questions:

- **Document-based**: “Summarize the last FRONTUR report”, “What is the average spending according to EGATUR?”.  
- **Prediction-based**: “How many tourists are expected in Spain in August 2025?”, “Forecast for Canarias in 2026”.  

The assistant will decide whether to search documents (**RAG**) or use predictive models (**API**).

## Can I trust the answers?

- For **official numbers and report summaries**, yes — they come directly from INE documents.  
- For **forecasts**, treat them as **projections** based on statistical models, not as guaranteed outcomes.  
