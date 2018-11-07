
# Atividade 4

## Equipe:

* Felipe Getúlio Laranjeira do Nascimento
* Lucas Pereira Reis

## Importação das bibliotecas


```python
import pandas as pd
import numpy as np
import combination as comb
from math import *
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neural_network import MLPClassifier

from sklearn.metrics import accuracy_score, confusion_matrix
```

## Realizando a leitura do *dataset*


```python
df = pd.read_csv('seeds_dataset.txt', sep='\t', header=None)
df.columns = ["area","perimeter","compactness",
              "length_kernel","width_kernel","asymmetry_coefficient",
              "length_kernel_groove","variety"]
print(f'Tamanho do dataset: {len(df)}')
```

    Tamanho do dataset: 210


# Analisando o *Dataset*

### 1. Histograma do atributo alvo


```python
target_name = "variety"
g = sns.catplot(x=target_name, data=df, kind="count", palette="muted", height=4.5, aspect=1.0)
g.set_xticklabels(['Kama', 'Rosa', 'Canadian'])
g.set_axis_labels("Classe", "Frequência")
plt.title('Quantidade de amostras para cada classe')
plt.show()
```


![png](resolucao-atividade-4_files/resolucao-atividade-4_8_0.png)


Conforme pode ser visto no histograma, as classes do atributo alvo são balanceadas, pois o *dataset* possui um total de 210 amostras e cada classe possui 70 amostras.

### 2. Heatmap da correlação de Pearson dos atributos do dataset


```python
plt.figure(figsize = (10,7))
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

corr = df.corr()

sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns,vmin=-1, vmax=1,linewidths=.5, cmap = "RdBu_r",annot=True)
plt.title('Heatmap da correlação de Pearson')
plt.show()
```


![png](resolucao-atividade-4_files/resolucao-atividade-4_11_0.png)


O *heatmap* acima exibe a correlação de Pearson entre todos os atributos do *dataset*, como pode ser visto o atributo `length_kernel_groove` possui uma correlação de aproximadamente igual a 0 com o atributo alvo `variety`. Assim, optou-se por remover o atributo do *dataset*.


```python
df.drop(['length_kernel_groove'],axis=1,inplace=True)
```

# Pré-Processamento de Dados

Antes de começar a utilizar o `GridSearchCV`, precisa-se primeiramente preparar o *dataset* e alguns atributos para serem utilizados na busca.


```python
target = df.variety
df.drop(['variety'],axis=1,inplace=True)
```


```python
n_i, n_o = len(df.columns), 3
```


```python
X_train, X_test, Y_train, Y_test = train_test_split(df,target,test_size=0.3)
```


```python
def geometric_pyramid(alpha):
    return alpha*sqrt(n_i*n_o)
```

#### Observação:
Para um `alpha = 2` ou `alpha = 3` o algoritmo para gerar o subconjunto fica bastante pesado. Talvez seja necessário diminuir o valor de `n_i` retirando colunas desnecessárias ou utilizar PCA. Coloquei `alpha = 1` apenas para testar


```python
alpha = [0.5,1]
n_h = [ceil(geometric_pyramid(a)) for a in alpha]
hidden_layer_sizes = []

for n in n_h:
    subsets = comb.partitions(n)
    hidden_layer_sizes = hidden_layer_sizes + subsets
```

# Paramêtros/Hiperparamêtros para a busca em grade

Na célula seguinte temos os paramêtros a serem passados para as redes neurais. Para o hiperparamêtro *solver*, optou-se por utilizar apenas o `lbfgs` pois o *dataset* desta atividade possui apenas 210 amostras, caracterizando-o como um *dataset* pequeno. Assim, o *solver* `lbfgs` será mais eficiente para o problema.


```python
params = {
    'activation': ['identity', 'logistic','tanh','relu'],
    'hidden_layer_sizes': hidden_layer_sizes,
    'solver': ['lbfgs']
}
```

# Projetando Redes Neurais através da busca em grade

A acurácia foi selecionada como a métrica de desempenho a ser utilizada para as redes neurais do `GridSearchCV`, e o método de validação cruzada escolhido é o *k-fold* com *k* = 3.


```python
gs = GridSearchCV(MLPClassifier(), params, cv=3, scoring='accuracy', return_train_score=1)
```


```python
X,y = df,target
```


```python
gs.fit(X,y);
```


```python
pd.DataFrame(gs.cv_results_).drop('params', 1).sort_values(by='rank_test_score').head()
```

Acima temos o *DataFrame* das Redes Neurais projetadas com o `GridSearchCV` ordenadas pelo campo `rank_test_score`, ou seja, está sendo exibido as cinco melhores redes neurais criadas. É possível retornar apenas a melhor rede neural através do campo `best_estimator_`, conforme abaixo.

# Avaliando a melhor Rede Neural


```python
best_model = gs.best_estimator_
```


```python
Y_pred = best_model.predict(X_test)
```


```python
accuracy_score(Y_test,Y_pred)
```

# Exportando para *Markdown*


```python
!jupyter nbconvert Atividade_4.ipynb --to markdown --output resolucao-atividade-4.md
```


```python

```