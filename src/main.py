import joblib
import torch
from fastapi import FastAPI
from contextlib import asynccontextmanager
from pathlib import Path

# Importando a classe do modelo e as rotas modulares
from src.train_mlp import TelcoMLP
from src.api.router import router


@asynccontextmanager
async def lifespan(app: FastAPI):
    project_root = Path(__file__).resolve().parents[1]

    try:
        # Carrega o artefato
        pipeline_path = project_root / "models" / "preprocessor_pipeline.pkl"
        loaded_artifact = joblib.load(pipeline_path)

        if hasattr(loaded_artifact, "named_steps") and "preprocess" in loaded_artifact.named_steps:
            preprocessor = loaded_artifact.named_steps["preprocess"]
        else:
            preprocessor = loaded_artifact

        input_dim = len(preprocessor.get_feature_names_out())

        # Inicializa a arquitetura e carrega os pesos
        model_path = project_root / "models" / "mlp_model.pth"
        model = TelcoMLP(input_dim=input_dim, hidden_dim=64)
        model.load_state_dict(torch.load(model_path, weights_only=True, map_location="cpu"))
        model.eval()

        # Salva no estado da aplicação
        app.state.preprocessor = preprocessor
        app.state.model = model

        print("Modelos carregados com sucesso!")
        yield
    except Exception as e:
        print(f"Erro ao carregar os artefatos: {e}")
        raise e
    finally:
        app.state.preprocessor = None
        app.state.model = None


app = FastAPI(
    title="Telco Churn Prediction API",
    description="API para inferência do modelo MLP PyTorch de previsão de Churn.",
    version="1.1.0",
    lifespan=lifespan
)

app.include_router(router)