
# Estudo de Seleção de Variáveis e Otimização de KS

Este estudo implementa um pipeline de **Feature Selection** robusto, projetado para cenários com alta dimensionalidade (100+ variáveis). O foco principal é a eliminação de ruído estatístico para evitar o overfitting e maximizar o ganho de **KS (Kolmogorov-Smirnov)** em bases de teste.

---

## 🎯 Objetivos do Estudo
* **Redução de Dimensionalidade:** Filtrar mais de 100 variáveis iniciais para manter apenas as mais preditivas.
* **Prevenção de Overfitting:** Utilizar Validação Cruzada (CV) para garantir que a importância das variáveis não seja fruto de acaso.
* **Maximização de Ganho:** Identificar as variáveis com maior "Performance Drift" (impacto direto no AUC/KS).

---

## 🛠️ Metodologia (Pipeline de Seleção)

O processo de seleção foi estruturado em três camadas críticas:

### 1. Filtro de Alta Concentração
Variáveis com concentração superior a **98%** em um único valor são descartadas. Variáveis quase constantes não possuem poder discriminante e geram instabilidade no modelo.


### 2. Tratamento de Multicolinearidade
Identificação de variáveis com correlação de Pearson superior a **0.85**. O seletor escolhe a melhor variável de cada grupo correlacionado baseando-se na maior variância, mantendo a integridade da informação sem redundância.

### 3. Recursive Feature Elimination (RFE)
Utilizando o **LightGBM** com `importance_type='gain'`, o algoritmo remove recursivamente as variáveis menos impactantes.
* **Critério de Permanência:** Ganho mínimo de **0.1%** no AUC por variável.
* **Validação:** Cross-validation com 3 folds para assegurar estabilidade.

---

## 📊 Relatório de Potencial
O script gera um ranking de **Performance Drift**. Esta métrica indica exatamente quanto o poder de separação do modelo (AUC/KS) cai ao removermos uma variável específica.



---

## 💻 Como Utilizar

1. **Instalação das dependências:**
```bash
pip install pandas lightgbm feature-engine scikit-learn
