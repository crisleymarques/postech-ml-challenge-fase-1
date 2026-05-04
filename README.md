
# 🚀 Guia de Setup Inicial - Tech Challenge Fase 1

Este guia contém os passos necessários para configurar seu ambiente de desenvolvimento e garantir que todos estejamos usando as mesmas versões de bibliotecas e padrões de código.

---

## 📋 Pré-requisitos
Antes de começar, certifique-se de ter instalado:
* **Python 3.13+**
* **Git**
* **Make** (Para usuários Windows, instale via `winget install GnuWin32.Make` ou Chocolatey).

---

## 🛠️ Configuração do Ambiente

Siga os passos abaixo no seu terminal (preferencialmente Git Bash ou terminal do VS Code):

### 1. Clonar o Repositório
```bash
git clone <url-do-repositorio>
cd postech-ml-challenge-fase-1
```

### 2. Criar e Ativar o Ambiente Virtual (venv)
```bash
python -m venv .venv

# No Windows (Git Bash):
source .venv/Scripts/activate

# No Windows (PowerShell):
.\.venv\Scripts\Activate.ps1
```

### 3. Instalar Dependências e Ferramentas
Com a venv ativa, utilize o Makefile para automatizar a instalação:
```bash
make install
```
*Este comando atualizará o pip e instalará o projeto em modo editável (`-e .`), incluindo as bibliotecas de Data Science e as ferramentas de qualidade (Ruff, Pytest).*

---

## 🧪 Qualidade de Código e Testes

Para manter o repositório organizado, utilizamos o **Ruff** como linter e formatador.

* **Verificar erros de lint:** `make lint`
* **Formatar código automaticamente:** `make format` (Execute sempre antes de um commit!)
* **Rodar testes:** `make test`

### MLflow

Os scripts de treino usam o **tracking URI** definido em `src/config.py` (por padrão SQLite em `mlflow.db` na raiz do repositório). Para ver runs e métricas na UI, suba o MLflow apontando para o **mesmo** backend que o código (`make mlflow-ui`), e não espere que tudo apareça só em `mlruns/` se o experimento foi registrado no SQLite.

---

## 🌳 Fluxo de Trabalho (Git Flow)

**Nunca trabalhe diretamente na branch `main`**. Para cada tarefa do backlog, siga este fluxo:

1. Atualize sua main: `git pull origin main`
2. Crie uma nova branch: `git checkout -b feature/nome-da-tarefa`
3. Faça suas alterações e commits.
4. Envie para o GitHub: `git push origin feature/nome-da-tarefa`
5. Abra um **Pull Request** para revisão do grupo.

---

## 📂 Estrutura do Projeto
- `src/`: Código Python reutilizável e entrypoints de treino.
  - `src/models/`: Definições de modelos (MLP e baselines).
  - `src/training/`: Loops de treino, validação e early stopping.
  - `src/evaluation/`: Métricas e avaliação de modelos.
  - `src/tracking/`: Integrações de tracking, como MLflow.
- `notebooks/`: Análises exploratórias e prototipagem.
- `data/raw/`: Dados brutos de entrada.
- `data/processed/`: Dados tratados ou derivados.
- `models/`: Artefatos de modelos treinados.
- `tests/`: Testes unitários e de integração.
- `docs/`: Documentação extra e ML Canvas.
<<<<<<< HEAD
```
=======
>>>>>>> f8290b60dbc7640f2f4cc0fb4d3618e51dad89f7
