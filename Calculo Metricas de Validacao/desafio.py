import numpy as np

# --- 1. Definição dos Valores da Matriz de Confusão ---
# Com base no nosso cenário de classificação de e-mails (SPAM vs Não SPAM).

# Classe Positiva: SPAM
# Classe Negativa: Não SPAM

VP = 40  # Verdadeiros Positivos: O modelo previu SPAM e estava correto.
VN = 145 # Verdadeiros Negativos: O modelo previu NÃO SPAM e estava correto.
FP = 5   # Falsos Positivos: O modelo previu SPAM, mas o e-mail era normal. (Erro Tipo I)
FN = 10  # Falsos Negativos: O modelo previu NÃO SPAM, mas o e-mail era SPAM. (Erro Tipo II)

# --- 2. Criação da Matriz de Confusão ---
# A convenção mais comum é:
# Linha 0: Real Negativo | Linha 1: Real Positivo
# Coluna 0: Previsto Negativo | Coluna 1: Previsto Positivo
#
#           [ [VN, FP],
#             [FN, VP] ]

matriz_confusao = np.array([
    [VN, FP],
    [FN, VP]
])


# --- 3. Exibição dos Resultados ---
# Imprime a matriz formatada e os valores individuais para facilitar os cálculos.

print("--- Matriz de Confusão Gerada ---")
print("Legenda: Linhas = Valores Reais, Colunas = Valores Previstos\n")
print("\t\tPrevisto: NÃO SPAM\tPrevisto: SPAM")
print(f"Real: NÃO SPAM\t[ {matriz_confusao[0,0]:^18} {matriz_confusao[0,1]:^18} ]")
print(f"Real: SPAM\t[ {matriz_confusao[1,0]:^18} {matriz_confusao[1,1]:^18} ]\n")


print("--- Valores para Utilização no Exercício ---")
print(f"Verdadeiros Positivos (VP): {VP}")
print(f"Verdadeiros Negativos (VN): {VN}")
print(f"Falsos Positivos (FP): {FP}")
print(f"Falsos Negativos (FN): {FN}")
print(f"Total de elementos (N): {VP + VN + FP + FN}")


# Este tópico busca encontrar a capacidade que o modelo tem de obter todos os casos positivos, minimizando os negativos. Métrica de cobertura
# Neste contexto responderia quantos e-mails que eram realmente spam (ao todo) o modelo conseguiu identificar
def calculo_sensibilidade(vp, fn):
    return vp / (vp + fn)


# Oposto da sensibilidade, mede a capacidade que o modelo teve de identificar corretamente os casos negativos, minimizando os falsos positivos.
# Neste contexto responderia quantos e-mails que não eram spam (ao todo) o modelo conseguiu identificar 
def calculo_especificidade(vn, fp):
    return vn / (fp + vn)


# Neste contexto responderia do total de previsões feitas, qual seria a taxa de acerto
def calculo_acuracia(vp, vn, fp, fn):
    return (vp + vn) / (vp + vn + fp + fn)


# De todos os e-mails que o modelo classificou como span, quantos realmente eram span?
# Foca na qualidade das previsões positivas, minimiza os falsos positivos
def calculo_precisao(vp, fp):
    return vp / (vp + fp)


# Equilíbrio entre precisão e sensibilidade, utilizada quanto o custo de um Falso Positivo e de um Falso Negativo são igualmente importantes
def calculo_fEscore(vp, fn, fp):
    p = calculo_precisao(vp, fp)
    s = calculo_sensibilidade(vp, fn)
    return (2 * (p * s)) / (p + s)

print("\n\n=== RESULTADO DAS MÉTRICAS DE VALIDAÇÃO ===\n\n")
print(f"Sensibilidade: {calculo_sensibilidade(VP, FN):.4f}\n")
print(f"Especificidade: {calculo_especificidade(VN, FP):.4f}\n")
print(f"Acurácia: {calculo_acuracia(VP, VN, FP, FN):.4f}\n")
print(f"Precisão: {calculo_precisao(VP, FP):.4f}\n")
print(f"F-score: {calculo_fEscore(VP, FN, FP):.4f}\n")