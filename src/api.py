import joblib
import pandas as pd
import torch
import torch.nn as nn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from contextlib import asynccontextmanager
from pathlib import Path

from src.train_mlp import TelcoMLP


# Variáveis globais para armazenar os artefatos carregados na memória
artifacts = {}

# Contrato de Entrada (Pydantic)
# ATENÇÃO: Ajuste os nomes e tipos exatos para que reflitam as colunas de teste do seu X_train final
class CustomerData(BaseModel):
    Gender: str
    Age: int
    Under30: str
    SeniorCitizen: str
    Married: str
    Dependents: str
    NumberofDependents: int
    Country: str
    State: str
    City: str
    ZipCode: int
    LatLong: str
    Latitude: float
    Longitude: float
    Population: int
    Quarter_x: str
    Quarter_y: str
    ReferredaFriend: str
    NumberofReferrals: int
    TenureinMonths: int
    Offer: str
    PhoneService: str
    AvgMonthlyLongDistanceCharges: float
    MultipleLines: str
    InternetService: str
    InternetType: str
    AvgMonthlyGBDownload: float
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtectionPlan: str
    PremiumTechSupport: str
    StreamingTV: str
    StreamingMovies: str
    StreamingMusic: str
    UnlimitedData: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharge: float
    TotalCharges: float
    TotalRefunds: float
    TotalExtraDataCharges: float
    TotalLongDistanceCharges: float
    TotalRevenue: float
    SatisfactionScore: int
    CLTV: float


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Inicialização: Carregar os modelos antes da API começar a aceitar requisições
    project_root = Path(__file__).resolve().parents[1]

    try:
        # 1. Carrega o artefato do sklearn
        pipeline_path = project_root / "models" / "preprocessor_pipeline.pkl"
        loaded_artifact = joblib.load(pipeline_path)

        # Se o arquivo salvo for o Pipeline completo, nós extraímos
        # apenas a etapa de pré-processamento (chamada "preprocess")
        if hasattr(loaded_artifact, "named_steps") and "preprocess" in loaded_artifact.named_steps:
            artifacts["preprocessor"] = loaded_artifact.named_steps["preprocess"]
        else:
            artifacts["preprocessor"] = loaded_artifact

        # 2. Descobre a dimensão de entrada após o processamento
        input_dim = len(artifacts["preprocessor"].get_feature_names_out())

        # 3. Inicializa a arquitetura e carrega os pesos do PyTorch
        model_path = project_root / "models" / "mlp_model.pth"
        model = TelcoMLP(input_dim=input_dim, hidden_dim=64)
        model.load_state_dict(torch.load(model_path, weights_only=True, map_location="cpu"))
        model.eval()  # Modo de inferência
        artifacts["model"] = model

        print("Modelos carregados com sucesso!")
        yield
    except Exception as e:
        print(f"Erro ao carregar os artefatos: {e}")
        raise e
    finally:
        # Limpeza ao desligar o serviço
        artifacts.clear()


# Instancia a aplicação FastAPI
app = FastAPI(
    title="Telco Churn Prediction API",
    description="API para inferência do modelo MLP PyTorch de previsão de Churn.",
    version="1.0.0",
    lifespan=lifespan
)


@app.get("/health", tags=["Health"])
async def health_check():
    """Endpoint para monitorar se a API está online."""
    return {"status": "ok", "message": "API de Churn operante."}


@app.post("/predict", tags=["Prediction"])
async def predict_churn(customer: CustomerData):
    """Endpoint de inferência em tempo real."""
    try:
        # 1. Converter payload validado pelo Pydantic para um DataFrame (mesmo formato do X_train)
        df_input = pd.DataFrame([customer.model_dump()])

        # 2. Aplicar as transformações do Pipeline do Scikit-Learn
        preprocessor = artifacts["preprocessor"]
        x_processed = preprocessor.transform(df_input)

        # 3. Converter para Tensor do PyTorch
        x_tensor = torch.tensor(x_processed, dtype=torch.float32)

        # 4. Fazer a inferência (Forward pass)
        model = artifacts["model"]
        with torch.no_grad():
            logits = model(x_tensor)
            probability = torch.sigmoid(logits).item()
            prediction = int(probability >= 0.5)

        # 5. Retornar JSON
        return {
            "churn_probability": round(probability, 4),
            "churn_prediction": prediction,
            "risk_level": "High" if prediction == 1 else "Low"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))