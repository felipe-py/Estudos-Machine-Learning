#Revisar curva Roc

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import numpy as np

def plotar_curva_roc(y_true, y_scores, title='Curva ROC'):
    """
    Esta função calcula e plota a Curva ROC.
    
    Parâmetros:
    y_true: Rótulos verdadeiros da classificação (ex: [0, 1, 1, 0]).
    y_scores: Pontuações de probabilidade da classe positiva, retornadas pelo modelo.
    title: Título do gráfico.
    """
    
    # 1. Calcular os pontos da curva (TFP, TVP)
    # A função roc_curve faz todo o trabalho de testar os limiares que fizemos manualmente.
    tfp, tvp, limiares = roc_curve(y_true, y_scores)
    
    # 2. Calcular a Área Sob a Curva (AUC)
    # A AUC é uma métrica que resume a curva em um único número (quanto maior, melhor).
    roc_auc = auc(tfp, tvp)
    
    # 3. Criar o gráfico
    plt.figure(figsize=(8, 6))
    
    # Plotar a Curva ROC do nosso modelo
    plt.plot(tfp, tvp, color='darkorange', lw=2, 
             label=f'Curva ROC (AUC = {roc_auc:.2f})')
    
    # Plotar a linha de referência (classificador aleatório)
    # Um modelo sem capacidade de distinção ficaria nesta linha.
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
             label='Classificador Aleatório')
    
    # Configurações do gráfico
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Taxa de Falsos Positivos (TFP)', fontsize=12)
    plt.ylabel('Taxa de Verdadeiros Positivos (TVP)', fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.show()

# --- Exemplo de Como Usar a Função ---

# Vamos usar exatamente os mesmos dados do nosso exemplo manual anterior.
# Assim você pode ver o código gerando o gráfico que discutimos.

# Rótulos verdadeiros (1 = SPAM, 0 = Não SPAM)
y_true = np.array([1, 0, 1, 1, 0, 0, 1, 0])

# Pontuações de probabilidade que o modelo deu para cada e-mail ser SPAM
y_scores = np.array([0.90, 0.20, 0.80, 0.70, 0.60, 0.10, 0.30, 0.40])

# Chamar a função para gerar e exibir o gráfico
plotar_curva_roc(y_true, y_scores, title='Curva ROC para o Modelo de SPAM')