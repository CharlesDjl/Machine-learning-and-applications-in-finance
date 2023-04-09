import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import plotly.express as ex
import plotly.graph_objs as go
import plotly.figure_factory as ff
import plotly.offline as pyo
from plotly.subplots import make_subplots

from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score as f1
from sklearn.metrics import confusion_matrix
from imblearn.over_sampling import SMOTE
import scikitplot as skplt

pyo.init_notebook_mode()
sns.set_style('darkgrid')
plt.rc('figure',figsize=(18,9))

data = pd.read_csv('../data/BankChurners.csv')
data = data[data.columns[:-2]] ## Naive Bayes class of the last cols
print(data.head(3))
