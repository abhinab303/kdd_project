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

l = 50

data_dir = "/home/aa7514/PycharmProjects/kdd_project/data/"
fig_dir = "/home/aa7514/PycharmProjects/kdd_project/plots/"

model_args = ModelArgs(max_seq_length=100)
emb_model = RepresentationModel(
        model_type="bert",
        # model_name="bert-base-uncased",
        model_name="google/bert_uncased_L-2_H-128_A-2",
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

training_data = api_dataframe.iloc[0:len(train_df)]
testing_data = api_dataframe.iloc[len(train_df):]

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


def eval_models(clf):
    clf.fit(x_train, y_train)

    pdn = clf.predict(x_test)

    f_score = f1_score(y_test, pdn, average='weighted')
    precision = precision_score(y_test, pdn, average='weighted')
    recall = recall_score(y_test, pdn, average='weighted')

    print("Train: ", clf.score(x_train, y_train))
    print("Test: ", clf.score(x_test, y_test))
    print("F P R: ", f_score, precision, recall)

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

