import numpy as np
import pandas as pd

from simpletransformers.language_representation import RepresentationModel
from simpletransformers.config.model_args import ModelArgs

from sklearn.preprocessing import LabelEncoder

from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import precision_score, f1_score, recall_score
from sklearn.model_selection import train_test_split

l = 50
sampling = "under"

if l == 50:
    max_iter = [1000]
    class_weight = ['balanced']
    c_param = [0.7668830376515554]
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

# ## Get data embedding:

train_file = f"{data_dir}{l}/train.csv"
test_file = f"{data_dir}{l}/test.csv"

train_df = pd.read_csv(train_file)
test_df = pd.read_csv(test_file)

api_dataframe = pd.concat([train_df, test_df], axis=0)
api_dataframe.reset_index(inplace=True, drop=True)

# sorted_labels = api_dataframe.groupby('ServiceClassification')['ServiceClassification'].count().sort_values(
#     ascending=False)

label_encoder = LabelEncoder()
values = np.array(api_dataframe.ServiceClassification)
api_dataframe['y'] = label_encoder.fit_transform(values)

training_data = api_dataframe.iloc[0:len(train_df)]
testing_data = api_dataframe.iloc[len(train_df):]

sorted_labels = training_data.groupby('ServiceClassification')['ServiceClassification'].count().sort_values(
    ascending=False)

if sampling == "under":
    min_freq = sorted_labels.values[-1]
    t_df = pd.DataFrame({'ServiceName': pd.Series(dtype='object'),
                          'ServiceDescription': pd.Series(dtype='object'),
                          'ServiceClassification': pd.Series(dtype='object'),
                          # 'y': pd.Series(dtype='object')
                          })
    for y in sorted_labels.index:
        y_indices = training_data[training_data.ServiceClassification == y].index
        y_chosen = np.random.choice(y_indices, min_freq, replace=False)
        y_samples = training_data.loc[y_chosen]
        t_df = pd.concat([t_df, y_samples], axis=0)

    # api_dataframe.reset_index(inplace=True, drop=True)

    # label_encoder = LabelEncoder()
    # values = np.array(api_dataframe.ServiceClassification)
    # api_dataframe['y'] = label_encoder.fit_transform(values)
    #
    # training_data, testing_data = train_test_split(api_dataframe, test_size=0.2, random_state=0,
    #                                stratify=api_dataframe[['ServiceClassification']])

    training_data = t_df

elif sampling == "over":
    min_freq = sorted_labels.values[0]
    t_df = pd.DataFrame({'ServiceName': pd.Series(dtype='object'),
                                  'ServiceDescription': pd.Series(dtype='object'),
                                  'ServiceClassification': pd.Series(dtype='object'),
                                  # 'y': pd.Series(dtype='object')
                                  })
    for (i, y) in enumerate(sorted_labels.index):
        y_indices = training_data[training_data.ServiceClassification == y].index
        if i == 0:
            y_chosen = np.random.choice(y_indices, min_freq, replace=False)
        else:
            # pdb.set_trace()
            y_chosen = np.random.choice(y_indices, min_freq, replace=True)
        y_samples = training_data.loc[y_chosen]
        t_df = pd.concat([t_df, y_samples], axis=0)

    # api_dataframe.reset_index(inplace=True, drop=True)

    # label_encoder = LabelEncoder()
    # values = np.array(api_dataframe.ServiceClassification)
    # api_dataframe['y'] = label_encoder.fit_transform(values)
    #
    # training_data, testing_data = train_test_split(api_dataframe, test_size=0.2, random_state=0,
    #                                stratify=api_dataframe[['ServiceClassification']])

    training_data = t_df

elif sampling == "both":
    min_freq = int(sorted_labels.mean())
    t_df = pd.DataFrame({'ServiceName': pd.Series(dtype='object'),
                                  'ServiceDescription': pd.Series(dtype='object'),
                                  'ServiceClassification': pd.Series(dtype='object'),
                                  # 'y': pd.Series(dtype='object')
                                  })
    for (i, y) in enumerate(sorted_labels.index):
        y_indices = training_data[training_data.ServiceClassification == y].index
        if sorted_labels[i] >= min_freq:
            y_chosen = np.random.choice(y_indices, min_freq, replace=False)
        else:
            # pdb.set_trace()
            y_chosen = np.random.choice(y_indices, min_freq, replace=True)
        y_samples = training_data.loc[y_chosen]
        t_df = pd.concat([t_df, y_samples], axis=0)

    # api_dataframe.reset_index(inplace=True, drop=True)
    #
    # label_encoder = LabelEncoder()
    # values = np.array(api_dataframe.ServiceClassification)
    # api_dataframe['y'] = label_encoder.fit_transform(values)
    #
    # training_data, testing_data = train_test_split(api_dataframe, test_size=0.2, random_state=0,
    #                                stratify=api_dataframe[['ServiceClassification']])

    training_data = t_df

else:
    pass

sorted_labels_after = training_data.groupby('ServiceClassification')['ServiceClassification'].count().sort_values(
    ascending=False)
print(sorted_labels_after)

word_vectors = emb_model.encode_sentences(api_dataframe['ServiceDescription'],
                                          combine_strategy="mean")
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

result_dict = {
    'Model':[],
    'F Score': [],
    'Precision': [],
    'Recall': [],
}

def eval_models(clf, model_name):
    clf.fit(x_train, y_train)

    pdn = clf.predict(x_test)

    f_score = f1_score(y_test, pdn, average='weighted')
    precision = precision_score(y_test, pdn, average='weighted')
    recall = recall_score(y_test, pdn, average='weighted')

    print("Train: ", clf.score(x_train, y_train))
    print("Test: ", clf.score(x_test, y_test))
    print("F P R: ", f_score, precision, recall)

    result_dict['Model'].append(str(model_name))
    result_dict['F Score'].append(str(f_score))
    result_dict['Precision'].append(str(precision))
    result_dict['Recall'].append(str(recall))

print("AdaBoost: ")
eval_models(AdaBoostClassifier(base_estimator=DecisionTreeClassifier(random_state=0),
                               n_estimators=10, random_state=0))
print("LR: ")
eval_models(LogisticRegression(random_state=0))
print("NN: ")
eval_models(MLPClassifier(random_state=1, max_iter=300))
print("RF: ")
eval_models(RandomForestClassifier(
    # n_estimators=10,
                                   # max_depth=2,
                                   random_state=0))

# from sklearn.tree import DecisionTreeClassifier
# eval_models(DecisionTreeClassifier(random_state=0))

csv_df = pd.DataFrame.from_dict(result_dict)
# pdb.set_trace()
csv_file_path = f"/home/aa7514/PycharmProjects/kdd_project/plots/final/top{l}_candidate_{sampling}_score.csv"
csv_df.to_csv(csv_file_path, index=False)

