# src/pytorch_mlp.py

import argparse
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


# 1 & 2. Definir Arquitetura da MLP e Função de Ativação
class TelcoMLP(nn.Module):
    """
        Um modelo de Perceptron Multicamadas (MLP) adaptado para tarefas de classificação.

        Esta classe implementa uma arquitetura simples de rede neural projetada para problemas de
        classificação binária. Consiste em múltiplas camadas totalmente conectadas (fully connected),
        funções de ativação ReLU e camadas de dropout para evitar o overfitting. A camada de saída
        é reduzida a uma única unidade para classificação binária, com os logits retornados para
        processamento posterior por uma função de ativação sigmoid, tipicamente incluída na função
        de perda (ex: BCEWithLogitsLoss).

        :ivar network: Um container sequencial de camadas incluindo camadas Linear, ReLU e Dropout,
            culminando em uma camada Linear final para classificação binária.
        :type network: torch.nn.Sequential
    """
    def __init__(self, input_dim: int, hidden_dim: int, dropout_rate: float = 0.2):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),  # Função de ativação
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, 1)  # Saída única para classificação binária
        )

    def forward(self, x):
        # A saída bruta (logits). A ativação Sigmoid será aplicada pela Loss Function (BCEWithLogitsLoss)
        return self.network(x)


def run_training(
        hidden_dim: int,
        learning_rate: float,
        epochs: int,
        batch_size: int,
        dropout_rate: float
) -> dict[str, float]:
    # Setup MLflow
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    # Preparação de Dados e Fixação de Seed
    torch.manual_seed(RANDOM_SEED)
    df = load_telco_dataset(DATA_DIR)
    x, y = split_features_target(df)

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_SEED
    )

    # Garantir compatibilidade com o pipeline de features
    preprocessor = build_preprocessor(x_train)
    x_train_processed = preprocessor.fit_transform(x_train)
    x_test_processed = preprocessor.transform(x_test)

    # Convertendo Pandas/Numpy para Tensores do PyTorch
    X_train_tensor = torch.tensor(x_train_processed, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
    X_test_tensor = torch.tensor(x_test_processed, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Inicialização do Modelo
    input_dim = X_train_tensor.shape[1]
    model = TelcoMLP(input_dim, hidden_dim, dropout_rate)

    # 3. Definir Loss Function (BCEWithLogitsLoss é mais estável numericamente que BCELoss)
    criterion = nn.BCEWithLogitsLoss()

    # 4. Definir Optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    with mlflow.start_run(run_name="pytorch-mlp"):
        # Log dos hiperparâmetros no MLflow (Checklist: Tornar configurável)
        mlflow.log_params({
            "hidden_dim": hidden_dim,
            "learning_rate": learning_rate,
            "epochs": epochs,
            "batch_size": batch_size,
            "dropout_rate": dropout_rate,
            "random_seed": RANDOM_SEED
        })

        # Loop de Treinamento
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

        # Avaliação
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

        mlflow.log_metrics(metrics)

        # Salvando o modelo treinado no formato PyTorch no MLflow
        mlflow.pytorch.log_model(model, "model")

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