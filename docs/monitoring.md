## 📈 Plano de Monitoramento de ML — Previsão de Churn

Este documento detalha a estratégia de monitoramento para o modelo de previsão de churn da Telco, cobrindo a integridade da infraestrutura e o impacto financeiro das predições.

### 1. Pilares de Monitoramento

O monitoramento foi divido em quatro camadas críticas para garantir que o modelo permaneça confiável e viável.

#### 1.1 Saúde da API (Service Metrics)
* **Latência (p95):** Tempo de resposta do endpoint `/predict`&rarr; Target: < 200ms.
* **Taxa de Erro HTTP:** Monitoramento de respostas tipo 400 (erros de contrato) e tipo 500 (erros internos do servidor).
* **Disponibilidade (Uptime):** Percentual de tempo que o serviço FastAPI está operacional.

#### 1.2 Qualidade de Dados e Drift (Data Metrics)
* **Data Drift:** Cálculo do *Population Stability Index* (PSI) para monitorar mudanças no perfil dos clientes (ex: aumento súbito na idade média ou nos gastos mensais).
* **Integridade de Schema:** Volume de requisições rejeitadas por tipos de dados inválidos via Pydantic.
* **Validação de Nulos:** Monitoramento de colunas obrigatórias que começam a chegar com valores vazios.

#### 1.3 Performance do Modelo (Model Metrics)
* **PR-AUC (Métrica Primária):** Monitoramento da área sob a curva Precision-Recall, ideal para o cenário de classes desbalanceadas.
* **Recall (Ground Truth):** Assim que o status real do cliente é confirmado (30 dias após a predição), calculamos quantos cancelamentos o modelo conseguiu de fato capturar.
* **Score Distribution:** Histograma das probabilidades geradas pela rede neural para detectar desvios de confiança do modelo.

#### 1.4 Impacto de Negócio (Business Metrics)
* **ROI de Retenção:** Retorno sobre o Investimento &rarr; Lucro gerado pelas ações de retenção bem-sucedidas vs. o custo fixo de \$50 por intervenção.
* **Expected Value Per Customer:** Acompanhamento da média de valor esperado que o modelo está salvando para a empresa.

---

### 2. Política de Alertas

Configuração de gatilhos para notificação imediata dos engenheiros de ML e stakeholders.

| Nível | Métrica | Gatilho (Trigger) |
| :--- | :--- | :--- | 
| **Crítico** | Taxa de Erro 5xx | > 2% em uma janela de 5 min | 
| **Crítico** | Latência p95 | > 1000ms (1 segundo) | 
| **Aviso** | Data Drift (PSI) | > 0.20 em features financeiras | 
| **Aviso** | Recall do Modelo | Queda de 10% em relação ao treino | 

---

### 3. Playbook de Resposta (Incidências)

Procedimentos operacionais para resposta diante de problemas detectados.

#### Cenário A: Falha na API ou Alta Latência
1. Verificar logs no terminal da aplicação.
2. Checar recursos de CPU e Memória.
3. **Ação:** Reiniciar containers ou realizar Rollback para a última versão estável via MLflow Model Registry.

#### Cenário B: Detecção de Data Drift Severo
1. Comparar as distribuições atuais com as estatísticas do arquivo `telco_churn_model_ready_manifest.json`.
2. Avaliar mudanças de mercado ou novos comportamentos de consumo.
3. **Ação:** Iniciar pipeline de retreino utilizando os dados coletados nos últimos 30 dias.

#### Cenário C: Queda de Performance Técnica
1. Validar se o Ground Truth (dados reais) está sendo coletado corretamente.
2. Analisar se o limiar de decisão (threshold) precisa de recalibração baseada em novos custos de negócio.
3. **Ação:** Atualizar o threshold no roteador da API ou treinar arquitetura MLP alternativa.

---

### 4. Governança
* **Versionamento:** Todos os logs de predição devem conter o ID da Run do MLflow que gerou a predição.
* **Auditabilidade:** Relatórios mensais de $PSI$ e $Recall$ são armazenados no diretório `outputs/monitoring/`.