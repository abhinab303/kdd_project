import numpy as np
import pandas as pd

from simpletransformers.language_representation import RepresentationModel
from simpletransformers.config.model_args import ModelArgs

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import precision_score, f1_score, recall_score

from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform

from sklearn import svm

from pathlib import Path

import pdb

import time

l = 50

data_dir = "/home/aa7514/PycharmProjects/kdd_project/data/"
fig_dir = "/home/aa7514/PycharmProjects/kdd_project/plots/"

model_args = ModelArgs(max_seq_length=100)
emb_model = RepresentationModel(
        model_type="bert",
        model_name="google/bert_uncased_L-2_H-128_A-2",
        use_cuda=True,
        # args= model_args
    )

train_file = f"{data_dir}{l}/train.csv"
test_file = f"{data_dir}{l}/test.csv"

train_df = pd.read_csv(train_file)
test_df = pd.read_csv(test_file)

api_dataframe = pd.concat([train_df, test_df], axis=0)
api_dataframe.reset_index(inplace=True, drop=True)

training_data = api_dataframe.iloc[0:len(train_df)]
testing_data = api_dataframe.iloc[len(train_df):]

op_file_path = f"/home/aa7514/PycharmProjects/kdd_project/files/emb_tiny{l}.npy"
op_file = Path(op_file_path)
if op_file.exists():
    word_vectors = np.load(op_file_path)
else:
    word_vectors = emb_model.encode_sentences(api_dataframe['ServiceDescription'],
                                              combine_strategy="mean")
    with open(op_file_path, 'wb') as f:
        np.save(f, word_vectors)
#

# pass
# pdb.set_trace()

print("train data dim: ",
      word_vectors[training_data["ServiceDescription"].index.values.tolist()].shape)
print("all data dim: ", word_vectors.shape)

x_train = word_vectors[training_data["ServiceDescription"].index.values.tolist()]
label_encoder = LabelEncoder()
values = np.array(training_data.ServiceClassification)
y_train = label_encoder.fit_transform(values)

x_test = word_vectors[testing_data["ServiceDescription"].index.values.tolist()]
label_encoder = LabelEncoder()
values = np.array(testing_data.ServiceClassification)
y_test = label_encoder.fit_transform(values)

print("label count: ", len(np.unique(y_train)))

# Hyper Params:
# max_iter = [100, 150, 200, 250, 300, 350, 400, 450, 500]

max_iter = [1] + list(range(50,1050,50))
class_weight = ["balanced", None]
c_param = [1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

result_dict = {
    'max_iter': [],
    'class_weight': [],
    'c': [],
    'Train': [],
    'Test': [],
    'F1 score': [],
    'Precision': [],
    'Recall': [],
}
# max_iter_list = []

print("training...")
start_time = time.time()
svm_p = svm.LinearSVC(C=1, class_weight="balanced", max_iter=1000)
distributions = dict(C=uniform(loc=0, scale=4), penalty=['l2', 'l1'],
                     class_weight=["balanced", None], max_iter=max_iter)
clf = RandomizedSearchCV(svm_p, distributions, random_state=0, n_jobs=-1, scoring="f1_weighted")
search = clf.fit(x_train, y_train)
print("best params: ", search.best_params_)


# besr params:  {'C': 0.28414423279154777, 'class_weight': None, 'penalty': 'l2'}

# random search with max iter (accuracy):
# 'C': 0.08087358976130288, 'class_weight': None, 'max_iter': 250, 'penalty': 'l2'

# random search with max iter (F1 score):
#