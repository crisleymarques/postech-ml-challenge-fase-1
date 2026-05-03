import pandas as pd
import torch
from fastapi import APIRouter, HTTPException, Request
from src.api.schemas import CustomerData, PredictionResponse

# Cria um roteador para os endpoints
router = APIRouter()

@router.get("/health", tags=["Health"])
async def health_check():
    """Endpoint para monitorar se a API está online."""
    return {"status": "ok", "message": "API modularizada de Churn operante."}

@router.post("/predict", tags=["Prediction"], response_model=PredictionResponse)
async def predict_churn(customer: CustomerData, request: Request):
    """Endpoint de inferência em tempo real."""
    try:
        # Recupera os modelos injetados no estado global da aplicação
        preprocessor = request.app.state.preprocessor
        model = request.app.state.model

        # Preparação do dado
        df_input = pd.DataFrame([customer.model_dump()])
        x_processed = preprocessor.transform(df_input)
        x_tensor = torch.tensor(x_processed, dtype=torch.float32)

        # Inferência
        with torch.no_grad():
            logits = model(x_tensor)
            probability = torch.sigmoid(logits).item()
            prediction = int(probability >= 0.5)

        # Retorna validado pelo schema PredictionResponse
        return PredictionResponse(
            churn_probability=round(probability, 4),
            churn_prediction=prediction,
            risk_level="High" if prediction == 1 else "Low"
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))