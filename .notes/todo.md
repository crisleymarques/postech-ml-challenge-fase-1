- Comparação com ensembles está fraca. Existe RandomForest, que cobre árvore/ensemble básico. Mas o PDF fala em “MLP e ensembles”; eu adicionaria pelo menos GradientBoostingClassifier, HistGradientBoostingClassifier, XGBoost ou LightGBM para não ficar dependente de um único ensemble.

Conclusão final precisa conter resultados reais renderizados. A seção existe, mas sem output. Precisa mostrar claramente melhor modelo, métricas finais, threshold escolhido, custo esperado e comparação contra baseline.

Validação cruzada está parcial. Ela roda só LogReg e MLP-best. Não é necessariamente errado, mas para a comparação da Etapa 2 eu incluiria também RandomForest/ensemble no CV ou justificaria explicitamente o custo computacional.


A análise de custo usa premissas simuladas. Isso é aceitável, mas precisa declarar melhor a origem/justificativa de COST_FP=50, COST_FN=500, COST_TP=-200. Sem isso, parece arbitrário.