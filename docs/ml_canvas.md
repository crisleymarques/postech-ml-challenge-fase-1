# Machine Learning Canvas — Telco Customer Churn

Blocos e perguntas conforme **Machine Learning Canvas v1.2** (Louis Dorard, OWNML) — [ownml.co](https://www.ownml.co/).

---

## Prediction task

> What is the type of task?  
> Which entity are predictions made on?  
> What are the possible outcomes to predict?  
> When are outcomes observed?

- **Tipo de tarefa:** classificação **binária supervisionada** — prever se o cliente cancela o plano da operadora.
- **Entidade:** **cliente** (`CustomerID`).
- **Desfechos:** `ChurnValue` ∈ {0, 1}; em produção expor **probabilidade** (threshold / filas).
- **Momento do rótulo:** **Histórico** — churn já definido para o período congelado do dataset. **Produção** — contratar **janela de evento** (ex.: cancelamento nos próximos *N* dias/meses após o snapshot usado como entrada).

---

## Decisions

> How are predictions turned into **actionable recommendations or decisions** for the end-user? *(Mention parameters of the process / application for this.)*

- **Saída operacional:** base ativa priorizada por probabilidade ou faixa / décil de risco.
- **Parâmetros:** threshold (top *k*% ou *p* mínimo); tipo de ação (desconto, upgrade, contato humano, nenhuma); canal (CRM, discador, e-mail); teto de incentivo por cliente/período.
- **Integração:** fila de ação no **CRM** alimentada pela API de scoring (**detalhe em Making predictions**).
- **Economia FP/FN:** matriz de custos e cenários — **Impact simulation**.

---

## Value proposition

> Who is the **end beneficiary**, and what specific **pain points** are addressed?  
> How will the ML solution **integrate** with their workflow, and through which **user interfaces**?

- **Stakeholders:** diretoria (priorização / budget); **CRM e retenção** (priorização da fila); marketing e **call center** (execução de campanhas); cliente final (efeito das ações).
- **Dores:** churn acelerado e receita incerta; sem priorização da fila; contato massificado de baixo valor.
- **Métrica de negócio (ânchora):** **receita mensal esperada retida vs. custo de intervenção** (proxy MRR/CLTV − incentivo − operação), calibrável na planilha de custos de **Impact simulation**.
- **Integração:** **CRM** ou motor de campanhas — scores por **batch** (ex.: diário) ou **consulta sob demanda** por `CustomerID`.
- **Interfaces:** no desafio, **FastAPI**; endpoints e SLA em **Making predictions** / **Monitoring**; em negócio, dashboard ou aplicativos já usados pela operadora.

---

## Data collection

> How is the **initial set** of entities and outcomes sourced (e.g., database extracts, API pulls, manual labeling)?  
> What strategies are in place to **update** data continuously while **controlling cost** and maintaining **freshness**?

- **Conjunto inicial:** arquivos **Excel** em `data/`, merge por `CustomerID` e `ZipCode`.
- **Atualização (alvo):** **ETL agendado** a partir de DW ou APIs internas.
- **Custo e escala:** padrão **batch diário**; incremental quando o armazém suportar; amostragem só sob restrição forte de ingestão (fora do escopo local do projeto).
- **Freshness:** alinhada ao SLA do DW (gerar scores compatíveis com esse ciclo — ex.: diário).

---

## Data sources

> Where can we get data on **entities** and **observed outcomes**? *(Mention internal and external database tables or API methods.)*

- `Telco_customer_churn_demographics.xlsx` — perfil do cliente (entidade).
- `Telco_customer_churn_location.xlsx` — localização (entidade).
- `Telco_customer_churn_services.xlsx` — contrato, serviços, cobrança, uso (entidade + comportamento).
- `Telco_customer_churn_population.xlsx` — população por CEP (contexto regional).
- `Telco_customer_churn_status.xlsx` — satisfação, **ChurnValue** / status (desfecho), CLTV, motivos.
- **Fora do merge atual da EDA:** `Telco_customer_churn.xlsx`, `Telco_customer_churn_old.xlsx` (no repo; não usados na EDA atual).

---

## Impact simulation

> What are the **cost/gain** values for **(in)correct** decisions?  
> Which **data** is used to simulate **pre-deployment** impact?  
> What are the **criteria for deployment**?  
> Are there **fairness** constraints?

- **Matriz econômica (TP / FP / TN / FN):** valores monetários e sensibilidade em **planilha** com premissas explícitas; FP — incentivo e operação; FN — **MRR/CLTV** e reposição; TP/TN conforme cenário.
- **Simulação pré-deploy:** confusion matrix; ROC/PR; thresholds sobre **CV estratificada** ou holdout; proxies (**MonthlyCharge**, **TotalRevenue**, **CLTV**) com hipótese explícita.
- **Critérios de champion / deploy:** MLP deve superar **DummyClassifier** e **regressão logística** em **PR-AUC** e/ou **F1**; documentar corte FP vs FN; **MLflow** com runs versionados; desafio exige também **AUC-ROC**, precisão/recall no corte (**Baselines**).
- **Fairness:** desempenho e taxa de intervenção por segmento (ex.: `SeniorCitizen`, região).

---

## Making predictions

> Are predictions made in **batch** or in **real time**?  
> How **frequently**?  
> How much **time** is available for this *(including featurization and decisions)*?  
> Which **computational resources** are used?

- **Modo:** **batch** (ex.: recálculo noturno da base) e **tempo real** por requisição (**FastAPI**).
- **Frequência:** online a cada chamada; batch conforme negócio (ex.: diário).
- **Latência:** pipeline **sklearn** + inferência — meta em **Monitoring** / SLOs.
- **Endpoints (desafio):** `/predict`, `/health`; **fallback** para cache ou regra simples (tenure + contrato); `/health` para roteamento.
- **Recursos:** CPU na API e inferência; **GPU opcional** no treino MLP; local ou nuvem (opcional).

---

## Building models

> How many **models** are needed in production?  
> When should they be **updated**?  
> How much **time** is available for this *(including featurization and analysis)*?  
> Which **computation resources** are used?

- **Produção:** **um modelo champion** por release; baselines só **offline**.
- **Atualização:** cadência ou **drift** / queda métrica em holdout; **MLflow**.
- **Janela de treino/análise:** offline (minutos a horas); não disputa SLA de inferência.
- **Stack:** **PyTorch** (MLP) + **Scikit-learn** (pipelines, baselines); **seeds fixas**.

---

## Features

> What **representations** are used for entities at prediction time?  
> What **aggregations or transformations** are applied to raw data sources?

- **Representação:** vetor **tabular** pós-merge (**53 colunas** no notebook) — demografia, localização, `Population`, serviços (`PhoneService`, `InternetService`, `InternetType`, add-ons, streaming), `TenureinMonths`, `Contract`, `PaymentMethod`, encargos/receita (`MonthlyCharge`, `TotalCharges`, `TotalRevenue`, …), `SatisfactionScore`.
- **Exclusões / vazamento:** não usar `CustomerID` como feature; evitar `ChurnScore`, `ChurnCategory`, `ChurnReason`; reduzir redundância com o alvo (`ChurnLabel` / `CustomerStatus`). **CLTV:** incluir **só** se for observável **no mesmo instante** da decisão simulada; caso contrário **excluir** ou documentar vazamento temporal no notebook.
- **Transformações:** limpeza, merges, **encoding**, **imputação**, **normalização** conforme o estimador.

---

## Monitoring

> Which **metrics and KPIs** are used to track the ML solution’s **impact** once deployed, both for **end-users** and for the **business**?  
> How **often** should they be reviewed?

- **Técnicos:** PR-AUC, recall no corte, taxa FP; shadow vs produção quando houver.
- **Negócio:** churn por cohort, receita retida, custo por salvamento, ROI de campanha.
- **API:** disponibilidade, **p95** latência em `/predict`, 5xx, uso de fallback.
- **Dados:** drift (ex.: `MonthlyCharge`, `Contract`), missing em produção.
- **SLOs iniciais:** disponibilidade mensal ≥ **99,5%**; **p95** `/predict` < **300 ms** (request único, warm-up feito); alerta se PR-AUC recente < piso acordado.
- **Revisão:** dashboards **semanais**; comitê / retreino **mensal**; **ad hoc** após incidente ou mudança de produto/pricing.

---

## Baselines e experimentação (Tech Challenge)

- **Baselines:** **DummyClassifier** e **regressão logística** (Scikit-learn). **Modelo central:** **MLP (PyTorch)** com early stopping. **Rastreio:** **MLflow** (parâmetros, métricas, artefatos, versão de dados). **Avaliação:** **CV estratificada**.
- **Comparação (≥ 4 métricas, desafio):** **AUC-ROC**, **PR-AUC**, **F1** no threshold escolhido, **precision** e **recall** (ou taxa de FP) no mesmo corte.
- **Critérios de deploy e trade-off de custo:** ver **Impact simulation**.
