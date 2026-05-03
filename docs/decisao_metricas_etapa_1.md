# Decisão de Métricas - Etapa 1

## Contexto do Dataset

O problema é classificação binária de churn em telecom: `target = 1` representa cliente que cancelou e `target = 0` representa cliente que permaneceu. A classe positiva é minoritária, com aproximadamente 26,5% dos registros. Isso torna a acurácia insuficiente como critério principal, porque um modelo conservador pode acertar muitos clientes que não cancelam sem capturar bem os clientes em risco.

O uso esperado do modelo também não é apenas classificar, mas priorizar clientes para ações de retenção. Nesse cenário, a qualidade do ranking de risco e o trade-off entre falsos positivos e falsos negativos são mais importantes que o percentual bruto de acertos.

## Métrica Primária: PR-AUC

`PR-AUC` ou `average_precision` é a métrica primária porque avalia diretamente a relação entre precisão e recall da classe positiva. Isso é adequado para churn, já que a classe de interesse é menor e operacionalmente mais valiosa: clientes que podem cancelar.

No dataset Telco, um bom `PR-AUC` indica que o modelo consegue ordenar clientes churn acima de clientes não churn ao longo de diferentes thresholds. Isso combina com o uso real em campanhas, onde normalmente se aciona um subconjunto priorizado por score.

## Métricas Complementares

`ROC-AUC` mede a separabilidade global entre as classes. Ela é útil para comparar modelos de forma geral, mas pode parecer otimista em datasets desbalanceados, por isso não deve ser a única métrica.

`Recall` mede quantos clientes churn reais são capturados. É crítico porque falso negativo significa deixar de abordar um cliente que pode cancelar.

`Precision` mede quantos clientes marcados como churn realmente cancelam. É importante porque falso positivo consome verba de campanha, desconto, atendimento ou contato comercial em cliente que talvez não cancelasse.

`F1` resume precision e recall em um único valor no threshold escolhido. Ele ajuda a comparar cortes operacionais, mas não substitui `PR-AUC`, porque depende de um threshold específico.

A matriz de confusão é mantida para explicar os erros em contagens absolutas: verdadeiros positivos, falsos positivos, falsos negativos e verdadeiros negativos.

## Métrica de Negócio

A métrica de negócio é o valor esperado por threshold. Ela traduz a matriz de confusão para impacto operacional:

- verdadeiro positivo: cliente churn identificado e acionado, com ganho esperado de receita retida menos custo de intervenção;
- falso positivo: cliente acionado sem necessidade, gerando custo de intervenção;
- falso negativo: cliente churn não acionado, gerando perda evitável de receita;
- verdadeiro negativo: cliente corretamente não acionado, sem custo incremental.

Como o dataset não traz uma campanha real com taxa de salvamento observada, as premissas são simuladas e explícitas no notebook 02: custo de intervenção, meses de receita potencialmente retida e taxa de sucesso da retenção. `MonthlyCharge` é usado como proxy observável de valor econômico. `CLTV` não entra como feature no dataset final por risco temporal, já que pode carregar informação agregada posterior ao evento de churn.

## Porque Acurácia Não Decide

`Accuracy` é registrada apenas como sanidade. Com desbalanceamento, um modelo que favorece a classe majoritária pode ter acurácia razoável e ainda assim ser ruim para retenção. Para este dataset, a decisão deve priorizar `PR-AUC`, recall/precision no threshold e valor esperado da ação.
