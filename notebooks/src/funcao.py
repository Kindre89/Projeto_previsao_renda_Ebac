import pandas as pd
import numpy as np
import statsmodels.api as sm

from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd



def avaliar_modelo_r2(alphas, X_train, y_train, X_test, y_test, mod_pre=Ridge, max_iter=600):
    """
    Avalia o score R² para modelos Lasso, Ridge ou ElasticNet com diferentes valores de alpha.

    Parâmetros:
    - alphas (list): Lista de valores de alpha para testar.
    - X_train (DataFrame/array): Features de treinamento.
    - y_train (Series/array): Target de treinamento.
    - X_test (DataFrame/array): Features de teste.
    - y_test (Series/array): Target de teste.
    - mod_pre (class): Classe do modelo a ser usado (Lasso, Ridge ou ElasticNet). O padrão é Lasso.
    - max_iter (int): Número máximo de iterações para o solver. O padrão é 1000.

    Retorna:
    - resultados (list): Lista de scores R² para cada alpha.
    """
    resultados = [] 
    
    for alpha in alphas:
        modelo = mod_pre(alpha=alpha, max_iter=max_iter)
        modelo.fit(X_train, y_train)
        y_pred = modelo.predict(X_test)
        resultados.append(r2_score(y_test, y_pred))
    
    return resultados

def best_estimator(alphas, r2):
    """
    Avalia qual é o melhor alpha e r2 do modelo.

    Parâmetros:
    - alphas (list): Lista de valores alphas.
    - r2  (list): Lista de valores de r2 do modelo.

    Retorna:
    - retorna uma tupla com o valor de alpha e o valor de r2 mais alto.
    """
    
    return list(zip(alphas, r2))[r2.index(max(r2))]



def stepwise_selection(X, y, initial_list=[], threshold_in=0.01, threshold_out=0.05, verbose=True):
    """
    Realiza a seleção de variáveis usando Stepwise Selection (combinação de Forward e Backward Selection).
    
    Parâmetros:
    - X: DataFrame com as variáveis independentes.
    - y: Série ou array com a variável dependente.
    - initial_list: Lista de variáveis iniciais a serem incluídas no modelo (opcional).
    - threshold_in: Limite de p-valor para adicionar uma variável (padrão: 0.01).
    - threshold_out: Limite de p-valor para remover uma variável (padrão: 0.05).
    - verbose: Se True, exibe mensagens sobre as variáveis adicionadas/removidas (padrão: True).
    
    Retorna:
    - included: Lista das variáveis selecionadas.
    - model: Modelo de regressão linear ajustado com as variáveis selecionadas.
    - mse: Erro Quadrático Médio na base de teste.
    - r2: Coeficiente de Determinação (R²) na base de teste.
    """
    included = list(initial_list)  # Lista de variáveis incluídas no modelo
    while True:
        changed = False  # Flag para verificar se houve mudança no passo atual

        # Passo Forward: Adicionar variáveis
        excluded = list(set(X.columns) - set(included))  # Variáveis ainda não incluídas
        new_pval = pd.Series(index=excluded)  # Armazenar p-valores das variáveis candidatas

        for new_column in excluded:
            # Ajusta o modelo com a variável candidata
            model = sm.OLS(y, sm.add_constant(pd.DataFrame(X[included + [new_column]]))).fit()
            new_pval[new_column] = model.pvalues[new_column]  # P-valor da variável candidata

        best_pval = new_pval.min()  # Menor p-valor entre as variáveis candidatas
        if best_pval < threshold_in:
            best_feature = new_pval.idxmin()  # Variável com o menor p-valor
            included.append(best_feature)  # Adiciona a variável ao modelo
            changed = True
            if verbose:
                print(f'Add  {best_feature:30} with p-value {best_pval:.6f}')

        # Passo Backward: Remover variáveis
        model = sm.OLS(y, sm.add_constant(pd.DataFrame(X[included]))).fit()
        pvalues = model.pvalues.iloc[1:]  # P-valores das variáveis no modelo (exceto intercepto)
        worst_pval = pvalues.max()  # Maior p-valor entre as variáveis no modelo

        if worst_pval > threshold_out:
            worst_feature = pvalues.idxmax()  # Variável com o maior p-valor
            included.remove(worst_feature)  # Remove a variável do modelo
            changed = True
            if verbose:
                print(f'Drop {worst_feature:30} with p-value {worst_pval:.6f}')

        # Se não houve mudança no passo atual, encerra o loop
        if not changed:
            break

    # Dividir os dados em treino e teste com as variáveis selecionadas
    X_melhores_colunas = X[included]
    X_train, X_test, y_train, y_test = train_test_split(X_melhores_colunas, y, test_size=0.25, random_state=42)

    # Ajustar o modelo de regressão linear
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Fazer previsões na base de teste
    y_pred = model.predict(X_test)

    # Avaliar o modelo
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    if verbose:
        print(f"\nMelhor modelo Stepwise - R² na base de teste: {r2:.4f}")
        print(f"Erro Quadrático Médio (MSE): {mse:.4f}")
        print("Variáveis selecionadas:", included)

    return included, model, mse, r2





    




