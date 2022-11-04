import numpy as np
import pandas as pd

from simpletransformers.language_representation import RepresentationModel
from simpletransformers.config.model_args import ModelArgs

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import precision_score, f1_score, recall_score
from sklearn.model_selection import StratifiedKFold

import matplotlib.pyplot as plt

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
        model_name="bert-base-uncased",
        use_cuda=True,
        # args= model_args
    )

train_file = f"{data_dir}{l}/train.csv"
test_file = f"{data_dir}{l}/test.csv"

train_df = pd.read_csv(train_file)
test_df = pd.read_csv(test_file)

api_dataframe = pd.concat([train_df, test_df], axis=0)
api_dataframe.reset_index(inplace=True, drop=True)

# training_data = api_dataframe.iloc[0:len(train_df)]
# testing_data = api_dataframe.iloc[len(train_df):]

op_file_path = f"/home/aa7514/PycharmProjects/kdd_project/files/emb{l}.npy"
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

# get the stratified indices:
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=4)
print(skf.get_n_splits(api_dataframe["ServiceDescription"], api_dataframe["ServiceClassification"]))
fold = 1

max_iter = [250, 300, 400, 350, 500]
class_weight = [None, ]
c_param = [1]

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

for itr in max_iter:
    for cpm in c_param:
        for wt in class_weight:
            f_score = 0
            train_acc = 0
            test_acc = 0
            precision = 0
            recall = 0
            start_time = time.time()
            print("training...")
            for train_index, test_index in skf.split(api_dataframe["ServiceDescription"], api_dataframe["ServiceClassification"]):
                training_data = api_dataframe[api_dataframe.index.isin(train_index)]
                testing_data = api_dataframe[api_dataframe.index.isin(test_index)]

                # Trainlabelcount = training_data['ServiceClassification'].value_counts()
                # trainP = Trainlabelcount / Trainlabelcount.sum()
                # Testlabelcount = testing_data['ServiceClassification'].value_counts()
                # Testlabelcount = Testlabelcount[Trainlabelcount.index]
                # TestP = Testlabelcount / Testlabelcount.sum()
                # # comparedf = pd.DataFrame({'Training Set': trainP, 'Test Set': TestP})
                # comparedf = pd.DataFrame({'Training Set': Trainlabelcount, 'Test Set': Testlabelcount})
                # comparedf.plot(kind='bar', figsize=(25, 15), fontsize=15)
                # plt.savefig(f'/home/aa7514/PycharmProjects/kdd_project/plots/s_k_fold_50_{fold}_count.pdf', format='pdf', dpi=300)
                # fold += 1

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

                # clf = svm.SVC(C=1, gamma="scale", class_weight="balanced", max_iter=250)
                clf = svm.LinearSVC(C=cpm, class_weight=wt, max_iter=itr)
                clf.fit(x_train, y_train)
                pdn = clf.predict(x_test)
                train_acc += clf.score(x_train, y_train)
                test_acc += clf.score(x_test, y_test)
                f_score += f1_score(y_test, pdn, average='weighted')
                precision += precision_score(y_test, pdn, average='weighted')
                recall += recall_score(y_test, pdn, average='weighted')

            train_acc /= 5
            test_acc /= 5
            f_score /= 5
            precision /= 5
            recall /= 5

            result_dict['max_iter'].append(str(itr))
            result_dict['class_weight'].append(str(wt))
            result_dict['c'].append(str(cpm))
            result_dict['Train'].append(str(train_acc))
            result_dict['Test'].append(str(test_acc))
            result_dict['F1 score'].append(str(f_score))
            result_dict['Precision'].append(str(precision))
            result_dict['Recall'].append(str(recall))

            print("Train: ", train_acc)
            print("Test: ", test_acc)
            print("F1 score: ", f_score)
            # print("Precision: ", precision)
            # print("Recall: ", recall)

            csv_df = pd.DataFrame.from_dict(result_dict)
            # pdb.set_trace()
            csv_file_path = "/home/aa7514/PycharmProjects/kdd_project/plots/result_50_k_fold.csv"
            csv_df.to_csv(csv_file_path, index=False)
            print("Time taken: ", time.time() - start_time)
