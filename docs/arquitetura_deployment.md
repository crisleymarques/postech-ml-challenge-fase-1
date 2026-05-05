## Documentação de Arquitetura de Deploy (Issue #36)

**Estratégia Escolhida:** Arquitetura Híbrida (Processamento em Lote via API de Inferência).

### 1. Justificativa de Negócio: O Caráter Preventivo da Retenção
Diferente de sistemas de detecção de fraude ou recomendações em e-commerce, a predição de **Churn** em Telecomunicações possui um caráter eminentemente preventivo e estratégico. As ações de mitigação (campanhas de fidelização, concessão de benefícios ou contato direto do CRM) não são desencadeadas por eventos instantâneos, mas sim executadas regularmente com base em um planejamento de médio prazo.

Dessa forma, o **Deploy em Batch (Lote)** é a solução mais aderente à realidade operacional do negócio:
* **Periodicidade:** A base de clientes é processada de forma cíclica (ex: semanal ou mensal).
* **Antecedência:** O modelo identifica o risco de evasão com antecedência suficiente para que as equipes de marketing e vendas planejem o contato.
* **Eficiência de Recursos:** Permite o processamento de grandes volumes de dados em horários de menor carga no sistema.

### 2. Decisão Técnica: Por que uma API para suportar o Batch?
Embora a necessidade de negócio seja focada em processos regulares (Batch), a implementação técnica foi feita através de uma **API REST (FastAPI)**. Esta decisão foi tomada para garantir a conformidade com os requisitos de engenharia do projeto e oferecer os seguintes benefícios:

* **Desacoplamento e Modularidade:** Ao encapsular a Rede Neural (MLP) em um serviço separado, garantimos que o orquestrador de processos (ex: Airflow ou cron) precise apenas enviar os dados via JSON para obter os scores, sem a necessidade de gerenciar dependências de PyTorch ou modelos pesados internamente.
* **Consistência de Inferência:** O mesmo endpoint `/predict` utilizado para gerar a lista de churn semanal pode ser consultado por outros sistemas internos de forma independente, garantindo que o score do cliente seja sempre calculado pela mesma versão do modelo.
* **Escalabilidade Futura:** Caso a operadora decida implementar uma camada de retenção ativa (ex: oferta automática durante uma chamada), a infraestrutura de tempo real já estará pronta e validada.

### 3. Componentes da Arquitetura
* **Framework:** `FastAPI` para a exposição dos endpoints.
* **Validação:** `Pydantic` para garantir o contrato de dados entre o processo batch e o modelo.
* **Endpoints Implementados:**
    * `/predict`: Recebe os atributos do cliente e retorna a probabilidade de churn.
    * `/health`: Check de integridade do serviço de inferência.