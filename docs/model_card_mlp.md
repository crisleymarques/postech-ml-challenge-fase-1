# Model Card — MLP (PyTorch) — Telco Customer Churn

## Visão geral

- **Nome do modelo:** `pytorch-mlp` (MLP para churn)
- **Tipo:** classificação binária supervisionada
- **Frameworks:** PyTorch, scikit-learn (pré-processamento), MLflow (tracking)
- **Código-fonte do treino:** [train\_mlp.py](file:///c:/Users/abtav/OneDrive/FIAP/Trabalho_fase_1/postech-ml-challenge-fase-1/src/train_mlp.py)
- **Rotina de treino (loop + early stopping):** [mlp\_training.py](file:///c:/Users/abtav/OneDrive/FIAP/Trabalho_fase_1/postech-ml-challenge-fase-1/src/mlp_training.py)
- **Objetivo:** estimar a probabilidade de churn (cancelamento) por cliente para priorização de ações de retenção.

## Problema e definição do alvo

- **Entidade:** cliente (`CustomerID`)
- **Alvo (label):** `target` construído a partir de `ChurnValue` (1 se `ChurnValue > 0`, senão 0)
- **Referência:** [load\_telco\_dataset](file:///c:/Users/abtav/OneDrive/FIAP/Trabalho_fase_1/postech-ml-challenge-fase-1/src/data.py#L34-L51)

## Dados

### Fontes

- Os dados são carregados de arquivos Excel em `data/` e unidos por `CustomerID` e `ZipCode`.
- Arquivos esperados (config): [RAW\_DATA\_FILES](file:///c:/Users/abtav/OneDrive/FIAP/Trabalho_fase_1/postech-ml-challenge-fase-1/src/config.py#L27-L33)
- Merge e target: [load\_telco\_dataset](file:///c:/Users/abtav/OneDrive/FIAP/Trabalho_fase_1/postech-ml-challenge-fase-1/src/data.py#L34-L51)

### Features e prevenção de vazamento

- As features são obtidas removendo identificadores e colunas com potencial de vazamento, além do próprio alvo.
- IDs removidos: [ID\_COLUMNS](file:///c:/Users/abtav/OneDrive/FIAP/Trabalho_fase_1/postech-ml-challenge-fase-1/src/config.py#L35-L38)
- Colunas de vazamento removidas: [LEAKAGE\_COLUMNS](file:///c:/Users/abtav/OneDrive/FIAP/Trabalho_fase_1/postech-ml-challenge-fase-1/src/config.py#L40-L48)
- Split X/y: [split\_features\_target](file:///c:/Users/abtav/OneDrive/FIAP/Trabalho_fase_1/postech-ml-challenge-fase-1/src/data.py#L65-L72)

### Pré-processamento

O pré-processamento é aplicado via `ColumnTransformer` do scikit-learn:

- Numéricas/booleanas: imputação por mediana + padronização (StandardScaler)
- Categóricas: imputação por moda + one-hot encoding (`handle_unknown="ignore"`)
- Referência: [build\_preprocessor](file:///c:/Users/abtav/OneDrive/FIAP/Trabalho_fase_1/postech-ml-challenge-fase-1/src/features.py#L8-L34)

## Arquitetura do modelo

Rede MLP simples (fully connected) com dropout:

- Entrada: `input_dim` (número de features após o pré-processamento)
- Camadas: `Linear(input_dim → hidden_dim) → ReLU → Dropout → Linear(hidden_dim → hidden_dim/2) → ReLU → Dropout → Linear(hidden_dim/2 → 1)`
- Saída: **logit** (sem sigmoid na última camada)
- Loss: `BCEWithLogitsLoss` (sigmoid implícita na loss)
- Referência: [TelcoMLP](file:///c:/Users/abtav/OneDrive/FIAP/Trabalho_fase_1/postech-ml-challenge-fase-1/src/train_mlp.py#L40-L55)

## Processo de treino

### Divisão treino/validação/teste

- `test_size = 0.2` estratificado
- `val_size = 0.2` estratificado (aplicado sobre o conjunto de treino “full”)
- Proporções efetivas (do total): treino ≈ 64%, validação ≈ 16%, teste = 20%
- Referência: [prepare\_data](file:///c:/Users/abtav/OneDrive/FIAP/Trabalho_fase_1/postech-ml-challenge-fase-1/src/train_mlp.py#L57-L103)

### Otimização e hiperparâmetros padrão

Defaults do CLI (quando não há override):

- `hidden_dim=64`
- `lr=0.001`
- `epochs=100` (máximo; pode parar antes)
- `batch_size=32`
- `dropout=0.2`
- `monitor="val_loss"` (alternativa: `val_roc_auc`)
- Early stopping: `patience=10`, `min_delta=1e-4`
- Referências: [parse\_args](file:///c:/Users/abtav/OneDrive/FIAP/Trabalho_fase_1/postech-ml-challenge-fase-1/src/train_mlp.py#L216-L253), [config](file:///c:/Users/abtav/OneDrive/FIAP/Trabalho_fase_1/postech-ml-challenge-fase-1/src/config.py#L10-L19)

### Early stopping

- Interrompe o treino quando a métrica monitorada deixa de melhorar pelo número de épocas definido em `patience`.
- Por padrão, restaura os melhores pesos observados.
- Referência: [EarlyStopping](file:///c:/Users/abtav/OneDrive/FIAP/Trabalho_fase_1/postech-ml-challenge-fase-1/src/mlp_training.py#L17-L72)

## Inferência e saída

- O modelo produz logits; a probabilidade é `sigmoid(logit)`.
- Predição binária usa threshold fixo de `0.5`.
- Referência: [evaluate\_model](file:///c:/Users/abtav/OneDrive/FIAP/Trabalho_fase_1/postech-ml-challenge-fase-1/src/train_mlp.py#L105-L125)

Recomendação para uso real:

- Ajustar threshold com base em custos de FP/FN (retenção vs. incentivo) e priorização por ranking (top-k).

## Métricas de avaliação

O script reporta e registra as seguintes métricas no **holdout de teste**:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Referência: [evaluate\_model](file:///c:/Users/abtav/OneDrive/FIAP/Trabalho_fase_1/postech-ml-challenge-fase-1/src/train_mlp.py#L105-L125)

### Resultados (a partir do MLflow)

Os valores numéricos variam conforme a execução. As métricas e parâmetros ficam registradas em MLflow no experimento definido em:

- `MLFLOW_EXPERIMENT_NAME = "telco-churn-baselines"`
- `MLFLOW_TRACKING_URI = sqlite:///mlflow.db` (na raiz do projeto)
- Referência: [config](file:///c:/Users/abtav/OneDrive/FIAP/Trabalho_fase_1/postech-ml-challenge-fase-1/src/config.py#L17-L19)

Última execução registrada localmente (MLflow):

- **Run ID:** `1344daa131c84e0a806f6c2bdd03d9ad`
- **Early stopping:** `True` (melhor época: `4`; épocas treinadas: `14`)
- **Parâmetros:** `hidden_dim=64`, `lr=0.001`, `epochs=30`, `batch_size=32`, `dropout=0.2`, `val_size=0.2`, `monitor=val_loss`, `device=cpu`
- **Métricas (teste):** accuracy=0.9532, precision=0.9503, recall=0.8690, f1=0.9078, roc\_auc=0.9892
- **Métricas (validação, melhor/último log):** val\_loss=0.1723, val\_accuracy=0.9547, val\_f1=0.9131, val\_roc\_auc=0.9874

## Uso pretendido

- **Uso adequado:** priorizar clientes com maior risco para intervenções de retenção; análise agregada por segmentos; experimentos offline.
- **Uso não recomendado:** decisões automatizadas sem revisão humana e sem avaliação de custo/impacto; uso fora do contexto do dataset sem checagem de drift.

## Considerações de segurança, privacidade e ética

- O dataset contém atributos demográficos (ex.: idade, gênero). Recomenda-se:
  - Avaliar métricas por subgrupo (ex.: `Gender`, `SeniorCitizen`) e taxas de intervenção.
  - Definir políticas de uso para evitar tratamento discriminatório.
- Evitar expor identificadores pessoais em logs/artefatos.

## Limitações e riscos conhecidos

- **Representatividade:** dataset é um recorte histórico; pode não representar mudanças futuras (produto, preço, concorrência).
- **Threshold fixo:** `0.5` pode ser inadequado para o objetivo de negócio; ideal calibrar e otimizar para custo/benefício.
- **Validação:** o script usa holdout + validação interna; recomenda-se validação cruzada e avaliação temporal quando houver dado longitudinal.
- **Risco de drift:** mudanças no mix de contratos/meios de pagamento podem afetar performance.

## Reprodutibilidade

- Seeds usadas:
  - `RANDOM_SEED` em splits do scikit-learn e em `torch.manual_seed`.
  - `DataLoader` com `generator` fixo para o shuffle.
- Referência: [prepare\_data](file:///c:/Users/abtav/OneDrive/FIAP/Trabalho_fase_1/postech-ml-challenge-fase-1/src/train_mlp.py#L57-L103)

## Como treinar

Exemplo (ajuste parâmetros conforme necessário):

```bash
python -m src.train_mlp --hidden-dim 64 --lr 0.001 --epochs 100 --batch-size 32 --dropout 0.2 --monitor val_loss
```

## Artefatos gerados

No MLflow, a execução registra:

- Parâmetros do treino e hiperparâmetros
- Métricas por época (train/val) e métricas finais de teste
- Artefato `training/history.json` com curvas de treino
- Modelo PyTorch (`mlflow.pytorch.log_model`)

Referência: [run\_training](file:///c:/Users/abtav/OneDrive/FIAP/Trabalho_fase_1/postech-ml-challenge-fase-1/src/train_mlp.py#L128-L213)
