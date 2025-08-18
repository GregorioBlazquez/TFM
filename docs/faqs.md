# Tourism Assistant FAQs

## What is FRONTUR?
FRONTUR is the survey carried out by the Spanish National Statistics Institute (INE) to measure the number of inbound tourists arriving in Spain, their origin, and travel characteristics.

## What is EGATUR?
EGATUR is the survey carried out by INE that measures tourist spending: average expenditure per trip, daily expenditure, and distribution by categories.

## How often are FRONTUR and EGATUR published?
They are published monthly, with detailed breakdowns by nationality, region of destination, and other variables.

## Where can I access official data?
You can download official reports in PDF from the INE website or through their API.

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
