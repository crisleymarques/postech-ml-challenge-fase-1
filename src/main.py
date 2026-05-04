import joblib
import torch
import time
import logging
from fastapi import FastAPI, Request
from contextlib import asynccontextmanager
from pathlib import Path

# Importando a classe do modelo e as rotas modulares
from src.train_mlp import TelcoMLP
from src.api.router import router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("telco_api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    project_root = Path(__file__).resolve().parents[1]

    try:
        logger.info("Iniciando o carregamento dos artefatos de ML...")

        # Carrega o artefato
        pipeline_path = project_root / "models" / "preprocessor_pipeline.pkl"
        loaded_artifact = joblib.load(pipeline_path)

        if (
            hasattr(loaded_artifact, "named_steps")
            and "preprocess" in loaded_artifact.named_steps
        ):
            preprocessor = loaded_artifact.named_steps["preprocess"]
        else:
            preprocessor = loaded_artifact

        input_dim = len(preprocessor.get_feature_names_out())

        # Inicializa a arquitetura e carrega os pesos
        model_path = project_root / "models" / "mlp_model.pth"
        model = TelcoMLP(input_dim=input_dim, hidden_dim=64)
        model.load_state_dict(
            torch.load(model_path, weights_only=True, map_location="cpu")
        )
        model.eval()

        # Salva no estado da aplicação
        app.state.preprocessor = preprocessor
        app.state.model = model

        logger.info("Modelos carregados com sucesso e prontos para inferência!")
        yield
    except Exception as e:
        logger.error(f"Erro fatal ao carregar os artefatos: {e}")
        raise e
    finally:
        logger.info("Encerrando a aplicação e limpando a memória...")
        app.state.preprocessor = None
        app.state.model = None


app = FastAPI(
    title="Telco Churn Prediction API",
    description="API para inferência do modelo MLP PyTorch de previsão de Churn.",
    version="1.1.0",
    lifespan=lifespan,
)


@app.middleware("http")
async def add_latency_logging(request: Request, call_next):
    start_time = time.time()

    # Executa a requisição
    response = await call_next(request)

    # Calcula a latência em milissegundos
    process_time_ms = (time.time() - start_time) * 1000

    # Log estruturado (chave=valor) facilitando o parsing por ferramentas de monitoramento
    logger.info(
        f"method={request.method} path={request.url.path} "
        f"status_code={response.status_code} latency_ms={process_time_ms:.2f}"
    )

    return response


app.include_router(router)
