import argparse
from typing import Tuple
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import mlflow
import mlflow.pytorch
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from src.config import DATA_DIR, MLFLOW_EXPERIMENT_NAME, MLFLOW_TRACKING_URI, RANDOM_SEED, TEST_SIZE
from src.data import load_telco_dataset, split_features_target
from src.features import build_preprocessor


class TelcoMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout_rate: float = 0.2):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, x):
        return self.network(x)


def prepare_data(batch_size: int) -> Tuple[DataLoader, torch.Tensor, pd.Series, int]:
    """Carrega os dados, aplica o pipeline do sklearn e converte para tensores."""
    torch.manual_seed(RANDOM_SEED)
    df = load_telco_dataset(DATA_DIR)
    x, y = split_features_target(df)

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_SEED
    )

    preprocessor = build_preprocessor(x_train)
    x_train_processed = preprocessor.fit_transform(x_train)
    x_test_processed = preprocessor.transform(x_test)

    # Convertendo para Tensores
    X_train_tensor = torch.tensor(x_train_processed, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
    X_test_tensor = torch.tensor(x_test_processed, dtype=torch.float32)

    # Criando o DataLoader para o treino
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    input_dim = X_train_tensor.shape[1]

    return train_loader, X_test_tensor, y_test, input_dim


def train_model(model: nn.Module, train_loader: DataLoader, criterion: nn.Module, optimizer: optim.Optimizer,
                epochs: int) -> None:
    """Executa o loop de treinamento do PyTorch."""
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}/{epochs} | Loss: {epoch_loss / len(train_loader):.4f}")


def evaluate_model(model: nn.Module, X_test_tensor: torch.Tensor, y_test: pd.Series) -> dict[str, float]:
    """Avalia o modelo treinado com os dados de teste e retorna as métricas."""
    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test_tensor)
        probabilities = torch.sigmoid(test_outputs).numpy()
        predictions = (probabilities >= 0.5).astype(int)

    metrics = {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions, zero_division=0),
        "recall": recall_score(y_test, predictions, zero_division=0),
        "f1": f1_score(y_test, predictions, zero_division=0),
        "roc_auc": roc_auc_score(y_test, probabilities)
    }
    return metrics


def run_training(
        hidden_dim: int,
        learning_rate: float,
        epochs: int,
        batch_size: int,
        dropout_rate: float
) -> dict[str, float]:
    """Função orquestradora: gerencia MLflow, pipeline de dados, treino e avaliação."""

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    # 1. Prepara os dados
    train_loader, X_test_tensor, y_test, input_dim = prepare_data(batch_size)

    # 2. Inicializa o modelo, loss e otimizador
    model = TelcoMLP(input_dim, hidden_dim, dropout_rate)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    with mlflow.start_run(run_name="pytorch-mlp"):
        mlflow.log_params({
            "hidden_dim": hidden_dim,
            "learning_rate": learning_rate,
            "epochs": epochs,
            "batch_size": batch_size,
            "dropout_rate": dropout_rate,
            "random_seed": RANDOM_SEED
        })

        # 3. Treina o modelo
        train_model(model, train_loader, criterion, optimizer, epochs)

        # 4. Avalia e salva
        metrics = evaluate_model(model, X_test_tensor, y_test)
        mlflow.log_metrics(metrics)
        input_example = X_test_tensor[:5].numpy()

        mlflow.pytorch.log_model(
            model,
            "model",
            serialization_format="pt2",
            input_example=input_example
        )

    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Treina a MLP do projeto de Churn com PyTorch.")
    parser.add_argument("--hidden-dim", type=int, default=64, help="Dimensão da camada oculta.")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate (taxa de aprendizado).")
    parser.add_argument("--epochs", type=int, default=50, help="Número de épocas.")
    parser.add_argument("--batch-size", type=int, default=32, help="Tamanho do batch.")
    parser.add_argument("--dropout", type=float, default=0.2, help="Taxa de dropout para evitar overfitting.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print("Iniciando treinamento da MLP...")

    metrics = run_training(
        hidden_dim=args.hidden_dim,
        learning_rate=args.lr,
        epochs=args.epochs,
        batch_size=args.batch_size,
        dropout_rate=args.dropout
    )

    print("\nResultados Finais no Conjunto de Teste:")
    for m, v in metrics.items():
        print(f"{m}: {v:.4f}")


if __name__ == "__main__":
    main()