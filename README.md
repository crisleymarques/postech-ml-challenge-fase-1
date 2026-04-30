
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
- `src/`: Scripts Python com a lógica do projeto (loaders, trainers, etc).
- `notebooks/`: Análises exploratórias e prototipagem.
- `data/`: Armazenamento de datasets (arquivos grandes são ignorados pelo git).
- `models/`: Artefatos de modelos treinados.
- `tests/`: Testes unitários e de integração.
- `docs/`: Documentação extra e ML Canvas.
```
