import requests

def predict_tourism(comunidad: str, periodo: str) -> dict:
    """
    Predicts the number of tourists for a given period.
    """
    print(f"Predicting tourism for {comunidad} in {periodo}...")

    response = requests.post(
        "http://localhost:8000/predict",  # FastAPI endpoint
        json={"comunidad": comunidad, "periodo": periodo}
    )
    
    if response.status_code == 200:
        return response.json()
    else:
        return {"error": f"API error: {response.status_code}"}