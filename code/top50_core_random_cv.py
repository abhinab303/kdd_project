import numpy as np
import pandas as pd

from simpletransformers.language_representation import RepresentationModel
from simpletransformers.config.model_args import ModelArgs

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import precision_score, f1_score, recall_score

from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform
from sklearn.model_selection import train_test_split


from sklearn import svm

from pathlib import Path

import pdb

import time

l = 50
sampling = "under"

if l == 50:
    max_iter = [550]
    class_weight = [None]
    c_param = [0.8595229470023127]
else:
    max_iter = [550]
    class_weight = [None]
    c_param = [0.8595229470023127]

data_dir = "/home/aa7514/PycharmProjects/kdd_project/data/"
fig_dir = "/home/aa7514/PycharmProjects/kdd_project/plots/"

model_args = ModelArgs(max_seq_length=100)
emb_model = RepresentationModel(
        model_type="bert",
        model_name="bert-base-uncased",
        # model_name="google/bert_uncased_L-2_H-128_A-2",
        use_cuda=True,
        # args= model_args
    )

train_file = f"{data_dir}{l}/train.csv"
test_file = f"{data_dir}{l}/test.csv"

train_df = pd.read_csv(train_file)
test_df = pd.read_csv(test_file)

api_dataframe_org = pd.concat([train_df, test_df], axis=0)
api_dataframe_org.reset_index(inplace=True, drop=True)

sorted_labels = api_dataframe_org.groupby('ServiceClassification')['ServiceClassification'].count().sort_values(
    ascending=False)

if sampling == "under":
    min_freq = sorted_labels.values[-1]
    api_dataframe = pd.DataFrame({'ServiceName': pd.Series(dtype='object'),
                                  'ServiceDescription': pd.Series(dtype='object'),
                                  'ServiceClassification': pd.Series(dtype='object'),
                                  # 'y': pd.Series(dtype='object')
                                  })
    for y in sorted_labels.index:
        y_indices = api_dataframe_org[api_dataframe_org.ServiceClassification == y].index
        y_chosen = np.random.choice(y_indices, min_freq, replace=False)
        y_samples = api_dataframe_org.loc[y_chosen]
        api_dataframe = pd.concat([api_dataframe, y_samples], axis=0)

    api_dataframe.reset_index(inplace=True, drop=True)

    label_encoder = LabelEncoder()
    values = np.array(api_dataframe.ServiceClassification)
    api_dataframe['y'] = label_encoder.fit_transform(values)

    training_data, testing_data = train_test_split(api_dataframe, test_size=0.2, random_state=0,
                                   stratify=api_dataframe[['ServiceClassification']])

elif sampling == "over":
    min_freq = sorted_labels.values[0]
    api_dataframe = pd.DataFrame({'ServiceName': pd.Series(dtype='object'),
                                  'ServiceDescription': pd.Series(dtype='object'),
                                  'ServiceClassification': pd.Series(dtype='object'),
                                  # 'y': pd.Series(dtype='object')
                                  })
    for (i, y) in enumerate(sorted_labels.index):
        y_indices = api_dataframe_org[api_dataframe_org.ServiceClassification == y].index
        if i == 0:
            y_chosen = np.random.choice(y_indices, min_freq, replace=False)
        else:
            # pdb.set_trace()
            y_chosen = np.random.choice(y_indices, min_freq, replace=True)
        y_samples = api_dataframe_org.loc[y_chosen]
        api_dataframe = pd.concat([api_dataframe, y_samples], axis=0)

    api_dataframe.reset_index(inplace=True, drop=True)

    label_encoder = LabelEncoder()
    values = np.array(api_dataframe.ServiceClassification)
    api_dataframe['y'] = label_encoder.fit_transform(values)

    training_data, testing_data = train_test_split(api_dataframe, test_size=0.2, random_state=0,
                                   stratify=api_dataframe[['ServiceClassification']])

elif sampling == "both":
    min_freq = int(sorted_labels.mean())
    api_dataframe = pd.DataFrame({'ServiceName': pd.Series(dtype='object'),
                                  'ServiceDescription': pd.Series(dtype='object'),
                                  'ServiceClassification': pd.Series(dtype='object'),
                                  # 'y': pd.Series(dtype='object')
                                  })
    for (i, y) in enumerate(sorted_labels.index):
        y_indices = api_dataframe_org[api_dataframe_org.ServiceClassification == y].index
        if sorted_labels[i] >= min_freq:
            y_chosen = np.random.choice(y_indices, min_freq, replace=False)
        else:
            # pdb.set_trace()
            y_chosen = np.random.choice(y_indices, min_freq, replace=True)
        y_samples = api_dataframe_org.loc[y_chosen]
        api_dataframe = pd.concat([api_dataframe, y_samples], axis=0)

    api_dataframe.reset_index(inplace=True, drop=True)

    label_encoder = LabelEncoder()
    values = np.array(api_dataframe.ServiceClassification)
    api_dataframe['y'] = label_encoder.fit_transform(values)

    training_data, testing_data = train_test_split(api_dataframe, test_size=0.2, random_state=0,
                                   stratify=api_dataframe[['ServiceClassification']])

else:
    api_dataframe = api_dataframe_org

    label_encoder = LabelEncoder()
    values = np.array(api_dataframe.ServiceClassification)
    api_dataframe['y'] = label_encoder.fit_transform(values)

    training_data = api_dataframe.iloc[0:len(train_df)]
    testing_data = api_dataframe.iloc[len(train_df):]

sorted_labels_after = api_dataframe.groupby('ServiceClassification')['ServiceClassification'].count().sort_values(
    ascending=False)
print(sorted_labels_after)
word_vectors = emb_model.encode_sentences(api_dataframe['ServiceDescription'], combine_strategy="mean")

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

# max_iter = list(range(300,1300,100))
max_iter = [1000]
class_weight = [None]
# c_param = [1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

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
distributions = dict(C=uniform(loc=0, scale=2), penalty=['l2'],
                     class_weight=["balanced", None], max_iter=max_iter)
clf = RandomizedSearchCV(svm_p, distributions, random_state=0, n_jobs=-1, scoring="f1_weighted", n_iter=10)
search = clf.fit(x_train, y_train)
print("best params: ", search.best_params_)


# US, 768d, 1000e
# best params:  {'C': 0.14207211639577388, 'class_weight': None, 'max_iter': 1000, 'penalty': 'l2'}


