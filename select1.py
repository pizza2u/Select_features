import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from feature_engine.selection import DropConstantFeatures, RecursiveFeatureElimination

# 1. Gerando Dataset Exemplo com 120 variáveis
# Criamos um cenário com muito ruído para testar a seleção
X, y = make_classification(
    n_samples=2000, 
    n_features=120,    # Mais de 100 variáveis
    n_informative=15,  # Apenas 15 são realmente boas
    n_redundant=20,    # 20 são redundantes
    random_state=42
)

feature_names = [f'feat_{i:03d}' for i in range(120)]
df = pd.DataFrame(X, columns=feature_names)
y = pd.Series(y)

# 2. Divisão Treino e Teste
X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.3, random_state=42)

# 3. FILTRO 1: Remover alta concentração (>98% no mesmo valor)
# Isso já limpa variáveis sem potencial nenhum
constant_filter = DropConstantFeatures(tol=0.98)
X_train_cleaned = constant_filter.fit_transform(X_train)
X_test_cleaned = constant_filter.transform(X_test)

# 4. AVALIAÇÃO DE POTENCIAL: Modelo LightGBM para Ranking
# Usamos o 'gain' para medir quanto cada variável contribui para o KS/AUC
lgb_model = lgb.LGBMClassifier(
    n_estimators=100,
    max_depth=4,
    learning_rate=0.05,
    importance_type='gain', # Foco no ganho de separação
    random_state=42,
    verbosity=-1
)

# 5. SELEÇÃO RECURSIVA (Otimizando o potencial de grupo)
# Ele vai testar o potencial das variáveis e cortar as que não ajudam no KS (AUC)
rfe = RecursiveFeatureElimination(
    estimator=lgb_model,
    scoring="roc_auc",
    threshold=0.001, # Critério: ganho mínimo de 0.1% no AUC/KS
    cv=3
)

rfe.fit(X_train_cleaned, y_train)

# 6. EXIBINDO O POTENCIAL DAS VARIÁVEIS (Ranking)
# Aqui você vê o potencial real de cada uma que restou
importances = rfe.performance_drifts_ # Queda de performance ao remover cada variável
ranking = pd.Series(importances).sort_values(ascending=False)

print("\n--- TOP 10 VARIÁVEIS POR POTENCIAL (Ganho de AUC) ---")
print(ranking.head(10))

# 7. Filtrando os dados finais
X_train_final = rfe.transform(X_train_cleaned)
X_test_final = rfe.transform(X_test_cleaned)

print(f"\nVariáveis iniciais: {len(feature_names)}")
print(f"Variáveis finais após análise de potencial: {X_train_final.shape[1]}")
