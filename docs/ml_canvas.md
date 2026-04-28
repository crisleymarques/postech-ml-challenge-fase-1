# Machine Learning Canvas — Telco Customer Churn

Blocos e perguntas conforme **Machine Learning Canvas v1.2** (Louis Dorard, OWNML) — [ownml.co](https://www.ownml.co/).

---

## Prediction task

> What is the type of task?  
> Which entity are predictions made on?  
> What are the possible outcomes to predict?  
> When are outcomes observed?

- **Tipo de tarefa:** classificação **binária supervisionada** — estimar se o cliente vai churnar.
- **Entidade:** **cliente** da operadora (`CustomerID`), no instante do snapshot (uma linha = um cliente no corte temporal).
- **Desfechos:** `ChurnValue` ∈ {0, 1} (permaneceu vs cancelou), alinhado a `ChurnLabel` / `CustomerStatus` no notebook; em produção, expor também **probabilidade** para threshold e filas.
- **Momento da observação do rótulo:** no histórico, churn já ocorreu ou não no período do dataset; em produção, definir **janela** (ex.: cancelamento nos próximos *N* dias/meses após o snapshot de entrada).
- **EDA:** média de `ChurnValue` ≈ **0,265** (~26,5% positivos) — churn como classe **minoritária**; reforça **PR-AUC** além da AUC-ROC.

---

## Decisions

> How are predictions turned into **actionable recommendations or decisions** for the end-user? *(Mention parameters of the process / application for this.)*

- **Saída operacional:** lista priorizada da base ativa por **probabilidade de churn** (ou faixas / deciles de risco).
- **Parâmetros do processo:** threshold (top *k*% ou probabilidade mínima); tipo de ação (desconto, upgrade, contato humano, nenhuma ação); canal (CRM, discador, e-mail); teto de incentivo por cliente/período.
- **Integração:** fila de ação no **CRM** alimentada pela API de scoring (`/predict`, escopo FastAPI do desafio).
- **Trade-off econômico:** **falso positivo** → custo de campanha e irritação; **falso negativo** → perda de **MRR/receita** e custo de reposição do cliente.

---

## Value proposition

> Who is the **end beneficiary**, and what specific **pain points** are addressed?  
> How will the ML solution **integrate** with their workflow, and through which **user interfaces**?

- **Beneficiários:** diretoria e **retenção**; **marketing e call center**; **cliente final**.
- **Dores:** churn acelerado e receita incerta; fila de retenção sem priorização; excesso de contato de baixo valor.
- **Integração:** **CRM** ou motor de campanhas — scores em **batch** (ex.: recálculo diário) ou **consulta online** por `CustomerID`.
- **Interfaces:** no desafio, **FastAPI** (`/predict`, `/health`); em negócio, painéis ou apps já usados pela operadora.

---

## Data collection

> How is the **initial set** of entities and outcomes sourced (e.g., database extracts, API pulls, manual labeling)?  
> What strategies are in place to **update** data continuously while **controlling cost** and maintaining **freshness**?

- **Conjunto inicial:** arquivos **Excel** em `data/`, unificados no notebook (`merge` por `CustomerID` e `ZipCode`) — análogo a *extracts* de billing, CRM e OSS.
- **Atualização contínua (visão alvo):** **ETL agendado** a partir de DW ou APIs internas.
- **Custo e frescor:** processamento incremental ou amostragem onde couber; **freshness** alinhada ao SLA do armazém (ex.: scores diários).
- **Rótulo:** continua a exigir **esperar o desfecho** ou fechar janela temporal antes de rotular novos exemplos.

---

## Data sources

> Where can we get data on **entities** and **observed outcomes**? *(Mention internal and external database tables or API methods.)*

- `Telco_customer_churn_demographics.xlsx` — perfil do cliente (entidade).
- `Telco_customer_churn_location.xlsx` — localização (entidade).
- `Telco_customer_churn_services.xlsx` — contrato, serviços, cobrança, uso (entidade + comportamento).
- `Telco_customer_churn_population.xlsx` — população por CEP (contexto regional).
- `Telco_customer_churn_status.xlsx` — satisfação, **ChurnValue** / status (desfecho), CLTV, motivos.
- **Fora do merge atual da EDA:** `Telco_customer_churn.xlsx`, `Telco_customer_churn_old.xlsx` (disponíveis no repo, não usados na EDA atual).

---

## Impact simulation

> What are the **cost/gain** values for **(in)correct** decisions?  
> Which **data** is used to simulate **pre-deployment** impact?  
> What are the **criteria for deployment**?  
> Are there **fairness** constraints?

- **Custos / ganhos (decisões corretas e incorretas):**
  - Verdadeiro positivo: evita perda de receita e CLTV.
  - Falso positivo: custo de incentivo, operação, canibalização possível.
  - Verdadeiro negativo / falso negativo: quantificar em **planilha de negócio** (valores monetários hipotéticos calibráveis).
- **Dados para simulação pré-deploy:** matrizes de confusão; curvas ROC/PR; cenários de threshold sobre **CV estratificado** ou holdout; proxies de valor (**MonthlyCharge**, **TotalRevenue**, **CLTV**) com premissas explícitas.
- **Critérios de deploy:** superar **DummyClassifier** e **regressão logística** em **PR-AUC** e/ou **F1**; documentar **trade-off FP vs FN** no corte; registrar no **MLflow**; métricas offline adicionais do desafio: **AUC-ROC**, precisão/recall no threshold.
- **Fairness:** monitorar métricas e taxa de intervenção por segmentos (ex.: `SeniorCitizen`, região); alinhar política a compliance — dataset **fictício**.

---

## Making predictions

> Are predictions made in **batch** or in **real time**?  
> How **frequently**?  
> How much **time** is available for this *(including featurization and decisions)*?  
> Which **computational resources** are used?

- **Modo:** **batch** (ex.: recálculo noturno da base) e **tempo real** sob demanda por cliente (**FastAPI**).
- **Frequência:** online por requisição; batch conforme ciclo de negócio (ex.: diário).
- **Tempo (featurização + decisão):** pipeline **sklearn** + inferência MLP/baseline; meta de **baixa latência** na API (ver *Monitoring* / SLOs).
- **Recursos:** CPU para API e inferência; **GPU opcional** para treino da MLP; local ou nuvem (deploy nuvem opcional no desafio).
- **Fallback:** último score em cache ou regra simples (tenure + tipo de contrato); `/health` para roteamento.

---

## Building models

> How many **models** are needed in production?  
> When should they be **updated**?  
> How much **time** is available for this *(including featurization and analysis)*?  
> Which **computation resources** are used?

- **Quantidade em produção:** em geral **um modelo champion** por release (ex.: MLP escolhida); baselines só para **comparação offline**; modelos por segmento = evolução fora do MVP do desafio.
- **Atualização:** periódica (ex.: mensal) ou por **drift** / queda de métrica em holdout; experimentos no **MLflow**.
- **Janela de tempo:** treino e análise **offline** (minutos a horas), sem competir com SLA de inferência.
- **Stack e recursos:** **PyTorch** (MLP) + **Scikit-Learn** (pipelines, baselines); CPU/GPU conforme ambiente; **seeds fixas** (requisito do desafio).

---

## Features

> What **representations** are used for entities at prediction time?  
> What **aggregations or transformations** are applied to raw data sources?

- **Representação:** vetor **tabular** pós-merge (**53 colunas** no notebook) — demografia, localização, `Population`, serviços (`PhoneService`, `InternetService`, `InternetType`, add-ons, streaming), `TenureinMonths`, `Contract`, `PaymentMethod`, encargos/receita (`MonthlyCharge`, `TotalCharges`, `TotalRevenue`, …), `SatisfactionScore`.
- **Exclusões / vazamento:** remover **ID** de entrada; **não** usar `ChurnScore`, `ChurnCategory`, `ChurnReason`; evitar redundância com o alvo (`ChurnLabel` / `CustomerStatus`). **CLTV** só se conhecido **no momento da decisão** — documentar premissa.
- **Transformações:** limpeza de nomes, merges, **encoding** de categóricas, **imputação** (EDA: `Offer` ~55% ausentes, `InternetType` ~21,7%, motivos de churn ~73,5% onde aplicável), **normalização** por algoritmo; revisar `Quarter_x` / `Quarter_y` se redundantes.

---

## Monitoring

> Which **metrics and KPIs** are used to track the ML solution’s **impact** once deployed, both for **end-users** and for the **business**?  
> How **often** should they be reviewed?

- **Técnicos:** PR-AUC, recall no corte, taxa de FP; shadow vs produção quando existir.
- **Negócio:** churn por cohort, receita retida, custo por salvamento, ROI de campanha.
- **API:** disponibilidade, p95 de latência em `/predict`, taxas 5xx, uso de fallback.
- **Dados:** drift (`MonthlyCharge`, `Contract`, etc.), missing em produção.
- **SLOs (referência):** disponibilidade mensal ≥ **99,5%**; p95 em `/predict` **abaixo de 300 ms** (payload único, pós warm-up); alertas se PR-AUC recente cair abaixo do mínimo acordado.
- **Cadência de revisão:** dashboards **semanal**; comitê de modelo / retreino **mensal**; análises **ad hoc** após incidentes ou mudança de produto/pricing.

---

## Baselines e experimentação (Tech Challenge)

- Baselines: **DummyClassifier** e **regressão logística** (Scikit-Learn).
- Modelo central: **MLP (PyTorch)** com early stopping.
- Rastreio: **MLflow** (parâmetros, métricas, artefatos, versão de dados).
- Avaliação: **validação cruzada estratificada**.
