from keras_metrics import sparse_categorical_precision, sparse_categorical_recall, sparse_categorical_f1_score
from keras.models import load_model
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
import pandas as pd
import numpy as np
from math import sqrt
from utils import *


def read_data(file='test.txt', type='test'):
    # if 'Cu' in file:
    #     sep = '\t'
    # if '锰' in file:
    #     sep = ' '
    assert 'test' in type
    # with open(file, 'r') as f:
    #     text = f.readlines()
    #     pass

    data = pd.read_table(file, sep='\t', header=None, encoding='utf8')
    feature_set = data.iloc[:, :-1]
    feature_set = feature_set.values.reshape((-1, 279, 1))
    label_set = data.iloc[:, -1]
    label_set = LabelEncoder().fit_transform(label_set)
    # test_size = 0.5
    # random_state = 42
    # if 'Cu' in file:
    #     feature_set = feature_set.reshape((-1, 2, 10))
    #     test_size = 0.5
    #     random_state = 42
    # if '锰' in file:
    #     feature_set = feature_set.reshape((-1, 5, 14))
    #     test_size = 0.5
    #     random_state = 42

    return feature_set, label_set


feature_set, label_set = read_data()

model = load_model('cnn_model.h5', custom_objects={"sparse_categorical_precision": sparse_categorical_precision(),
                                                    "sparse_categorical_recall": sparse_categorical_recall(),
                                                    "sparse_categorical_f1_score": sparse_categorical_f1_score()})
score = model.evaluate(feature_set, label_set, verbose=1)
proba = model.predict_proba(feature_set)[:, 1]
pred = model.predict_classes(feature_set)
con_mat = confusion_matrix(label_set, pred)
auc = plot_roc(label_set, proba)
result = score + [con_mat[0, 0] / np.sum(con_mat, axis=1)[0],
                  con_mat[1, 1] / np.sum(con_mat, axis=1)[1],
                  np.linalg.det(con_mat) / sqrt(np.sum(con_mat, axis=1).cumprod()[-1] * np.sum(con_mat, axis=0).cumprod()[-1]),
                  auc]
print(*list(zip(["loss", "accuracy", "F1", "Precision", "Recall", "Sensitivity", "Specificity", "MCC", "AUC"], result)),
      sep='\n')
print(con_mat)
