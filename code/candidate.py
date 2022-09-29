import numpy as np
import pandas as pd

from simpletransformers.language_representation import RepresentationModel
from simpletransformers.config.model_args import ModelArgs

from sklearn.preprocessing import LabelEncoder

from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier

l = 50

data_dir = "/content/drive/Shareddrives/ISTE780project/data/"
fig_dir = "/content/drive/Shareddrives/ISTE780project/plots/"

model_args = ModelArgs(max_seq_length=100)
emb_model = RepresentationModel(
        model_type="bert",
        model_name="bert-base-uncased",
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
    print("Train: ", clf.score(x_train, y_train))
    print("Test: ", clf.score(x_test, y_test))


eval_models(AdaBoostClassifier(n_estimators=100, random_state=0))
eval_models(LogisticRegression(random_state=0))
eval_models(MLPClassifier(random_state=1, max_iter=300))
eval_models(RandomForestClassifier(max_depth=2, random_state=0))



