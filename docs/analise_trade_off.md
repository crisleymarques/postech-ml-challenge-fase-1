## Análise de Impacto de Negócio e Trade-off de Custos (Churn)

Esta análise expõe o desempenho técnico obtido pelo modelo de *Multi-Layer Perceptron* com *PyTorch* em métricas de valor financeiro, fundamentando as decisões de threshold e monitoramento para a operação de retenção da operadora.

### 1. Avaliação do Trade-off: Falsos Positivos vs. Falsos Negativos

No contexto de cancelamento de assinaturas (Churn), os erros de classificação possuem custos assimétricos:

* **Falso Negativo (maior impacto):** O modelo prevê que o cliente permanecerá ativo, mas ele cancela o serviço.
    * **Impacto:** Perda total do *Customer Lifetime Value* (CLTV), perda de receita mensal recorrente e necessidade de novo investimento em marketing (CAC) para repor o cliente.
* **Falso Positivo (menor impacto):** O modelo prevê que o cliente vai sair, mas ele não tinha intenção de cancelar.
    * **Impacto:** Custo operacional de uma tentativa de retenção desnecessária e concessão de incentivos/descontos (estimado em $50.00 por ação) para um cliente que já ficaria na base.

Observa-se portante maior importância da métrica de Recall, em relação à de Precisão.

### 2. Impacto no Negócio Estudado

Considerando as premissas estabelecidas na fase de experimentação:
* **Custo de Intervenção:** $50.00
* **Receita Mensal Média (Proxy):** ~$70.00
* **Taxa de Sucesso na Retenção:** 25%

O impacto financeiro de um Falso Negativo é aproximadamente 5 a 10 vezes superior ao custo de um Falso Positivo, dependendo do tempo que o cliente permaneceria na base (LTV). Portanto, a prioridade absoluta do negócio é a otimização do Recall (Revocação).

### 3. Estratégia de Threshold (Limiar de Decisão)

O modelo MLP obteve um **Recall de 0.8770** com o threshold padrão de `0.5`. 

**Decisão de Engenharia:**
Para otimizar o ROI (Retorno sobre Investimento), recomendamos uma **estratégia de threshold agressiva**:
* **Ajuste Sugerido:** Reduzir o threshold operacional de `0.5` para **`0.4` ou `0.35`**.
* **Justificativa:** Ao baixar o limiar, a API sinalizará mais clientes como "Alto Risco". Isso pode aumente o número de falsos positivos (descontos ineficientes), mas o ganho financeiro ao evitar *churns* reais passem despercebidos compensa o desperdício operacional.

### 4. Registro da Decisão Final

* **Modelo Champion:** *Multi-Layer Perceptron*, devido ao seu ROC-AUC superior (0.9837), demonstrando maior capacidade de separação entre as classes e estabilidade comparado aos baselines.
* **Métrica de Sucesso de Negócio (KPI):** Redução da Taxa de Churn Mensal em detrimento de uma margem controlada de perda em Precisão.
* **Diretriz de MLOps:** O monitoramento em produção deve focar na "Matriz de Custos Esperados". Caso o custo de aquisição de novos clientes (CAC) aumente no mercado, o threshold deve ser reduzido ainda mais para proteger a base atual de clientes.