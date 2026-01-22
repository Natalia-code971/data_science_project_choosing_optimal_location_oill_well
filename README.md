# Выбор локации для скважины
Допустим, вы работаете в добывающей компании «ГлавРосГосНефть». Нужно решить, где бурить новую скважину.

Вам предоставлены пробы нефти в трёх регионах: в каждом 100 000 месторождений, где измерили качество нефти и объём её запасов. Постройте модель машинного обучения, которая поможет определить регион, где добыча принесёт наибольшую прибыль. Проанализируйте возможную прибыль и риски техникой *Bootstrap.*

Шаги для выбора локации:

- В избранном регионе ищут месторождения, для каждого определяют значения признаков;
- Строят модель и оценивают объём запасов;
- Выбирают месторождения с самым высокими оценками значений. Количество месторождений зависит от бюджета компании и стоимости разработки одной скважины;
- Прибыль равна суммарной прибыли отобранных месторождений.

```
python

!pip install -U scikit-learn -q

!pip install --upgrade scikit-learn lightgbm -q
```
# Загрузка библиотек

```
python

import re

import math

import pandas as pd

import sklearn

import seaborn as sns

import matplotlib.pyplot as plt

import numpy as np

from scipy import stats as st

from scipy.stats import loguniform

from sklearn.preprocessing import (OneHotEncoder,
                                   StandardScaler,
                                   PolynomialFeatures,
                                   MinMaxScaler,
                                   FunctionTransformer)

from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline

from sklearn.linear_model import Ridge

from sklearn.impute import SimpleImputer

from sklearn.model_selection import RandomizedSearchCV

from sklearn.linear_model import LinearRegression

from sklearn.metrics import (mean_absolute_error,
                             mean_squared_error,
                             r2_score,
                             precision_score,
                             confusion_matrix)

from sklearn.model_selection import train_test_split
```
