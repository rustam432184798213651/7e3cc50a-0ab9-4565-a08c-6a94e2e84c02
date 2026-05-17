# Источники формул из раздела «Используемые модели машинного обучения»

Файл фиксирует, откуда взяты математические записи для моделей из главы отчета `Используемые модели машинного обучения`. Формулы в отчете записаны в едином обозначении под задачу бинарной классификации удовлетворенности пассажиров, поэтому местами это не дословное копирование, а нормализованная запись стандартных формул из источников.

## Logistic Regression

Формулы в отчете:

```tex
P(y=1 \mid x)=\sigma(w^{T}x+b)=\frac{1}{1+\exp(-(w^{T}x+b))}
```

```tex
\hat{y}=\mathbb{I}\{P(y=1\mid x)\geq t\}
```

```tex
L(w,b)=-\frac{1}{n}\sum_{i=1}^{n}\left[y_i\log p_i+(1-y_i)\log(1-p_i)\right]+\lambda\|w\|_2^2
```

Источник:

- Scikit-learn User Guide, раздел `1.1.11 Logistic regression`, подраздел `1.1.11.1 Binary Case`:  
  https://scikit-learn.org/stable/modules/linear_model.html#binary-case

Точная часть источника:

- В разделе `Logistic regression` сказано, что вероятности исходов моделируются логистической функцией, а модель является линейной моделью классификации.
- В подразделе `Binary Case` дана формула вероятности положительного класса:

```tex
\hat{p}(X_i)=\operatorname{expit}(X_iw+w_0)=\frac{1}{1+\exp(-X_iw-w_0)}
```

- Там же дана функция потерь для бинарной логистической регрессии с регуляризацией:

```tex
\min_w \frac{1}{S}\sum_{i=1}^{n}s_i
\left(-y_i\log(\hat{p}(X_i))-(1-y_i)\log(1-\hat{p}(X_i))\right)
+\frac{r(w)}{SC}
```

- В таблице регуляризаций того же подраздела указано, что для L2-регуляризации используется член:

```tex
\frac{1}{2}\|w\|_2^2
```

Как это перенесено в отчет:

- `X_iw+w_0` записано как `w^Tx+b`.
- `\hat{p}(X_i)` записано как `P(y=1|x)`.
- Регуляризация `r(w)/(SC)` упрощена до `\lambda\|w\|_2^2`, чтобы не привязывать теоретическую запись к параметру `C` из конкретной реализации.
- Пороговое правило классификации взято из того же описания scikit-learn: числовой выход логистической регрессии является вероятностью и используется как классификатор через применение порога, по умолчанию `0.5`.

## KNN

Формулы в отчете:

```tex
\rho_2(x,z)=\sqrt{\sum_{j=1}^{d}(x_j-z_j)^2}
```

```tex
\rho_1(x,z)=\sum_{j=1}^{d}|x_j-z_j|
```

```tex
\hat{y}=\arg\max_{c}\sum_{i\in N_k(x)}\mathbb{I}\{y_i=c\}
```

```tex
\hat{y}=\arg\max_{c}\sum_{i\in N_k(x)}\frac{\mathbb{I}\{y_i=c\}}{\rho(x,x_i)+\varepsilon}
```

Источники:

- Scikit-learn User Guide, раздел `1.6 Nearest Neighbors`, подраздел `1.6.2 Nearest Neighbors Classification`:  
  https://scikit-learn.org/stable/modules/neighbors.html#nearest-neighbors-classification
- Scikit-learn API, `KNeighborsClassifier`:  
  https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
- Scikit-learn API, `euclidean_distances`:  
  https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.euclidean_distances.html
- Scikit-learn API, `manhattan_distances`:  
  https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.manhattan_distances.html

Точная часть источника:

- В `Nearest Neighbors Classification` указано, что классификация вычисляется через majority vote ближайших соседей: объект получает класс, наиболее часто представленный среди ближайших соседей.
- В `KNeighborsClassifier` параметр `weights='uniform'` описан как равные веса всех соседей, а `weights='distance'` как веса, пропорциональные обратному расстоянию.
- В `euclidean_distances` приведена евклидова дистанция между двумя векторами через скалярные произведения:

```text
dist(x, y) = sqrt(dot(x, x) - 2 * dot(x, y) + dot(y, y))
```

Это эквивалентно записи `sqrt(sum_j (x_j-y_j)^2)`.

- В `manhattan_distances` указано, что функция вычисляет `L1 distances` между векторами, что соответствует `sum_j |x_j-y_j|`.

Как это перенесено в отчет:

- Majority vote из scikit-learn записан через `argmax` по классам и сумму индикаторов.
- `weights='distance'` записан как вес `1/(\rho(x,x_i)+\varepsilon)`. Добавка `\varepsilon` нужна только для математической устойчивости записи при нулевом расстоянии.
- Евклидова и манхэттенская метрики записаны в покомпонентном виде, потому что так они проще читаются в теоретическом разделе.

## Decision Tree

Формулы в отчете:

```tex
G(S)=1-\sum_{c=1}^{C}p_c^2
```

```tex
\Delta G=G(S)-\frac{|S_L|}{|S|}G(S_L)-\frac{|S_R|}{|S|}G(S_R)
```

Источник:

- Scikit-learn User Guide, раздел `1.10 Decision Trees`, подразделы `1.10.7 Mathematical formulation` и `1.10.7.1 Classification criteria`:  
  https://scikit-learn.org/stable/modules/tree.html#mathematical-formulation  
  https://scikit-learn.org/stable/modules/tree.html#classification-criteria

Точная часть источника:

- В `Mathematical formulation` кандидатное разбиение узла задается как `theta=(j,t_m)`, где `j` - признак, а `t_m` - порог.
- Качество разбиения задается через взвешенную impurity дочерних узлов:

```tex
G(Q_m,\theta)=
\frac{n_m^{left}}{n_m}H(Q_m^{left}(\theta))+
\frac{n_m^{right}}{n_m}H(Q_m^{right}(\theta))
```

- Затем выбирается разбиение, минимизирующее impurity:

```tex
\theta^*=\operatorname{argmin}_{\theta}G(Q_m,\theta)
```

- В `Classification criteria` критерий Джини задан как:

```tex
H(Q_m)=\sum_k p_{mk}(1-p_{mk})
```

Эта формула эквивалентна `1-\sum_k p_{mk}^2`, потому что `\sum_k p_{mk}=1`.

Как это перенесено в отчет:

- `H(Q_m)` из scikit-learn записано как `G(S)`, чтобы явно обозначить Gini impurity.
- В отчете дана не сама взвешенная impurity дочерних узлов, а уменьшение impurity: impurity родителя минус взвешенная impurity левого и правого дочерних узлов. Это та же логика выбора split, только записанная как максимизация выигрыша, а не минимизация остаточной impurity.

## Random Forest

Формулы в отчете:

```tex
\hat{y}=\arg\max_{c}\sum_{b=1}^{B}\mathbb{I}\{h_b(x)=c\}
```

```tex
\hat{P}(y=c\mid x)=\frac{1}{B}\sum_{b=1}^{B}\mathbb{I}\{h_b(x)=c\}
```

Источник:

- Breiman L. Random Forests, технический отчет Berkeley, раздел `1.1 Introduction`, `Definition 1.1`:  
  https://www.stat.berkeley.edu/~breiman/random-forests.pdf

Точная часть источника:

- В начале раздела `1.1 Introduction` описана идея ансамбля деревьев, которые голосуют за наиболее популярный класс.
- В `Definition 1.1` Random Forest определяется как коллекция tree-structured classifiers `{h(x, Θ_k), k=1,...}`, где `{Θ_k}` - независимые одинаково распределенные случайные векторы, и каждое дерево отдает один голос за наиболее популярный класс на входе `x`.
- В разделе `2.1 Random Forests Converge` используется функция margin:

```tex
mg(X,Y)=av_k I(h_k(X)=Y)-max_{j \ne Y} av_k I(h_k(X)=j)
```

Здесь `av_k I(...)` - средняя доля голосов деревьев.

Как это перенесено в отчет:

- Голосование деревьев из `Definition 1.1` записано через `argmax` по классам и сумму индикаторов.
- Средняя доля голосов `av_k I(...)` записана как оценка вероятности класса `1/B * sum_b I(h_b(x)=c)`.
- Обозначение `h_b(x)` в отчете соответствует отдельному дереву `h(x, Θ_k)` у Breiman.

## XGBoost

Формулы в отчете:

```tex
\hat{y}_i^{(t)}=\sum_{k=1}^{t} f_k(x_i), \quad f_k \in \mathcal{F}
```

```tex
\mathcal{L}^{(t)}=\sum_{i=1}^{n}l(y_i,\hat{y}_i^{(t-1)}+f_t(x_i))+\Omega(f_t)
```

```tex
\Omega(f)=\gamma T+\frac{1}{2}\lambda\sum_{j=1}^{T}w_j^2+\alpha\sum_{j=1}^{T}|w_j|
```

Источник:

- Chen T., Guestrin C. XGBoost: A Scalable Tree Boosting System, KDD 2016 / arXiv:1603.02754, разделы `2.1 Regularized Learning Objective` и `2.2 Gradient Tree Boosting`:  
  https://arxiv.org/abs/1603.02754  
  PDF: https://arxiv.org/pdf/1603.02754

Точная часть источника:

- В разделе `2.1 Regularized Learning Objective` дана аддитивная модель деревьев:

```tex
\hat{y}_i=\phi(x_i)=\sum_{k=1}^{K} f_k(x_i), \quad f_k \in \mathcal{F}
```

- Там же определена регуляризованная objective-функция:

```tex
L(\phi)=\sum_i l(\hat{y}_i,y_i)+\sum_k \Omega(f_k)
```

- В той же части дана регуляризация дерева:

```tex
\Omega(f)=\gamma T+\frac{1}{2}\lambda\|w\|^2
```

- В разделе `2.2 Gradient Tree Boosting` описано пошаговое добавление нового дерева:

```tex
L^{(t)}=\sum_{i=1}^{n} l(y_i,\hat{y}_i^{(t-1)}+f_t(x_i))+\Omega(f_t)
```

Как это перенесено в отчет:

- Формула аддитивной композиции взята из `Eq. (1)` статьи и записана для шага `t`.
- Формула `L^{(t)}` взята из раздела `2.2 Gradient Tree Boosting`.
- Член `\frac{1}{2}\lambda\sum_j w_j^2` является покомпонентной записью `\frac{1}{2}\lambda\|w\|^2`.
- Дополнительный L1-член `\alpha\sum_j |w_j|` соответствует параметру `reg_alpha` в современной реализации XGBoost. В исходной статье KDD 2016 явно записан L2-член; L1-регуляризация в отчете добавлена, потому что в работе подбирался параметр `reg_alpha` и нужно было показать обе регуляризации, фактически используемые в эксперименте.

Дополнительная ссылка на документацию реализации:

- XGBoost Parameters, раздел `Parameters for Tree Booster`, параметры `reg_alpha` и `reg_lambda`:  
  https://xgboost.readthedocs.io/en/stable/parameter.html

## Neural Networks

Формулы в отчете:

```tex
h_1 = \mathrm{ReLU}(W_1x+b_1)
```

```tex
h_2 = \mathrm{ReLU}(W_2h_1+b_2)
```

```tex
z = W_3h_2+b_3
```

```tex
P(y=c\mid x)=\frac{\exp(z_c)}{\sum_{k}\exp(z_k)}
```

```tex
L=-\frac{1}{n}\sum_{i=1}^{n}\sum_{c}y_{ic}\log \hat{p}_{ic}
```

```tex
\tilde{h}=m\odot h, \quad m_j\sim \mathrm{Bernoulli}(1-p)
```

Источники:

- PyTorch `Linear`:  
  https://docs.pytorch.org/docs/2.12/generated/torch.nn.Linear.html
- PyTorch `ReLU`:  
  https://docs.pytorch.org/docs/2.12/generated/torch.nn.ReLU.html
- PyTorch `Softmax`:  
  https://docs.pytorch.org/docs/2.12/generated/torch.nn.Softmax.html
- PyTorch `CrossEntropyLoss`:  
  https://docs.pytorch.org/docs/2.12/generated/torch.nn.CrossEntropyLoss.html
- PyTorch `Dropout`:  
  https://docs.pytorch.org/docs/2.12/generated/torch.nn.Dropout.html
- PyTorch `AdamW`, параметр `weight_decay`:  
  https://docs.pytorch.org/docs/2.12/generated/torch.optim.AdamW.html

Точная часть источника:

- В `Linear` указано, что слой применяет аффинное линейное преобразование:

```tex
y=xA^T+b
```

- В `ReLU` указана функция:

```tex
\mathrm{ReLU}(x)=\max(0,x)
```

- В `Softmax` указана нормализация:

```tex
\mathrm{Softmax}(x_i)=\frac{\exp(x_i)}{\sum_j \exp(x_j)}
```

- В `CrossEntropyLoss` для задачи с `C` классами дана функция потерь через logits и целевой класс:

```tex
l_n=-w_{y_n}\log\frac{\exp(x_{n,y_n})}{\sum_{c=1}^{C}\exp(x_{n,c})}
```

- Там же указано, что этот случай эквивалентен применению `LogSoftmax` к входу и затем `NLLLoss`.
- В `Dropout` указано, что во время обучения некоторые элементы входного тензора зануляются с вероятностью `p`, элементы выбираются независимо и сэмплируются из распределения Бернулли.
- В `AdamW` указано, что optimizer реализует AdamW, где `weight_decay` не накапливается в momentum и variance, а параметр `weight_decay` является коэффициентом weight decay.

Как это перенесено в отчет:

- Два скрытых слоя записаны как последовательность `Linear + ReLU`: `W_1x+b_1`, `W_2h_1+b_2`.
- Выходной слой записан как еще одно линейное преобразование `W_3h_2+b_3`.
- Вероятности классов записаны через `Softmax`.
- Cross-entropy записана в one-hot форме `-\sum_c y_{ic}\log \hat{p}_{ic}`; это эквивалентно формуле PyTorch для target class indices после применения softmax/log-softmax.
- Dropout записан через маску `m`, где элементы маски имеют распределение Бернулли. В отчете используется `m_j ~ Bernoulli(1-p)`, потому что `p` в PyTorch - вероятность зануления, а вероятность сохранения активации равна `1-p`.
- Weight decay в отчете описан как L2-штраф к весам; в реализации он задавался параметром optimizer, что соответствует документации `AdamW`.

## Сводка соответствий

| Модель | Формула в отчете | Основной источник | Часть источника |
|---|---|---|---|
| Logistic Regression | sigmoid, threshold, log-loss + L2 | scikit-learn User Guide | `1.1.11 Logistic regression`, `1.1.11.1 Binary Case` |
| KNN | Euclidean/L1 distance, majority vote, distance weighting | scikit-learn User Guide/API | `1.6.2 Nearest Neighbors Classification`, `KNeighborsClassifier`, `euclidean_distances`, `manhattan_distances` |
| Decision Tree | Gini impurity, impurity decrease | scikit-learn User Guide | `1.10.7 Mathematical formulation`, `1.10.7.1 Classification criteria` |
| Random Forest | majority vote, average vote probability | Breiman, Random Forests | `1.1 Introduction`, `Definition 1.1`, `2.1 Random Forests Converge` |
| XGBoost | additive trees, regularized objective, tree penalty | Chen & Guestrin, XGBoost | `2.1 Regularized Learning Objective`, `2.2 Gradient Tree Boosting` |
| Neural Network | Linear, ReLU, Softmax, CrossEntropy, Dropout, weight decay | PyTorch docs | `Linear`, `ReLU`, `Softmax`, `CrossEntropyLoss`, `Dropout`, `AdamW` |
