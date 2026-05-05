### 🧪 Arquitetura da Suíte de Testes (MLOps & CI/CD)

A garantia de qualidade deste projeto vai além da simples verificação de código. Nossa suíte de testes (com 14 cenários automatizados via **pytest**) foi desenhada sob a ótica de **MLOps**, validando três pilares fundamentais: a integridade dos dados, a consistência do treinamento e a confiabilidade da API de inferência.

A execução contínua desta suíte protege o sistema contra *data leakage*, quebras de contrato de dados e degradação silenciosa.

### 1. Testes de Dados e Versionamento
Antes de treinar qualquer modelo, garantimos que a base de dados é sólida e rastreável.
* **Prevenção de Data Leakage:** Valida se colunas que contêm informações do futuro ou diretamente ligadas à variável alvo (como `Churn Score`, `Customer Status`) são rigorosamente removidas da matriz de features (`x`) antes do treinamento.
* **Rastreabilidade (Hashing):** Verifica se o carregamento de dados exige e valida o manifesto (`_manifest.json`) com o hash **SHA-256**. O teste força uma falha caso os dados tenham sido alterados sem o devido rastreio, garantindo reprodutibilidade estrita.
* **Estabilidade de Versão:** Testa se a geração do manifesto produz a mesma versão para os mesmos arquivos e se o versionamento muda adequadamente quando a fonte de dados crua é alterada.

### 2. Testes de Treinamento e Métricas
Garante que as rotinas de machine learning operam conforme as boas práticas estatísticas.
* **Isolamento de Cross-Validation (CV):** Testa rigorosamente a função de treinamento para garantir que os *scalers* e modelos vejam apenas os dados de treino durante as dobras do CV. Isso impede que o modelo \"roube\" informações do conjunto de validação.
* **Geração de Métricas Base:** Verifica se as pipelines de pré-processamento e modelos treinam corretamente e conseguem extrair todas as métricas obrigatórias (`accuracy`, `precision`, `recall`, `f1`, `roc_auc`).
* **Training Smoke Test:** Valida o fluxo ponta a ponta do treinamento, garantindo que o acionamento do script constrói o pipeline e avalia os modelos sem quebrar a execução.

### 3. Testes de Inferência e Contrato da API (`test_api.py`)
Esta etapa simula a interação do usuário final (ou de sistemas externos) com o modelo já treinado. Para garantir a estabilidade do serviço em produção, aplicamos três conceitos fundamentais da Engenharia de Software:

* **1. Smoke Test (Teste de Fumaça - Endpoint `/health`):** O nome vem da engenharia de hardware (se você ligar o aparelho e não sair fumaça, o teste básico passou). Em software, é um teste rápido para confirmar se as funções vitais do sistema estão operacionais.
  * **No projeto:** Verifica se a aplicação FastAPI consegue inicializar corretamente, se a porta de comunicação responde e, crucialmente, se os artefatos pesados de Machine Learning (o `.pkl` do Scikit-Learn e o `.pth` do PyTorch) foram carregados com sucesso na memória, retornando o status HTTP `200 OK`.

* **2. Schema Test (Teste de Contrato/Validação):**
  * **O que é:** É a barreira de segurança da API. Garante o princípio do *\"garbage in, garbage out\"* (lixo entra, lixo sai), atestando que o sistema rejeita requisições com dados malformados antes que eles alcancem o motor de inferência, evitando sobrecarga ou travamentos.
  * **No projeto:** Simula o envio de um payload com tipos incorretos (ex: enviar uma *String* onde a rede neural espera um número *Inteiro*). O teste valida se o esquema rígido do `Pydantic` intercepta a falha e devolve graciosamente o erro padrão de contrato: HTTP `422 Unprocessable Entity`.

* **3. API Test (Teste de Integração End-to-End - Endpoint `/predict`):**
  * **O que é:** Avalia o fluxo completo de uma transação de sucesso, certificando-se de que a comunicação entre o cliente, a interface web e o serviço de dados interno funciona em perfeita harmonia.
  * **No projeto:** Um payload JSON completo, imitando o cadastro real de um cliente da Telco, é enviado à API. O teste acompanha o dado entrando pela rota, sendo limpo pelo pipeline de pré-processamento, convertido em tensores matemáticos (`torch.Tensor`) e mastigado pela MLP. Por fim, o teste atesta se a API retorna o HTTP `200 OK` acompanhado dos cálculos finais (`churn_probability` e `risk_level`).

### 4. Testes de Setup
* **Environment Sanity:** Checagem de base para atestar que o ecossistema virtual e as dependências Python estão ativas antes de executar testes complexos.

---
**Como executar a suíte localmente (CI/CD Ready):** Rode no terminal o `make test`