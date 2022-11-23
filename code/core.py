import numpy as np
import pandas as pd

from simpletransformers.language_representation import RepresentationModel
from simpletransformers.config.model_args import ModelArgs

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import precision_score, f1_score, recall_score

from sklearn import svm

from pathlib import Path

import pdb

import time
import matplotlib.pyplot as plt

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

train_file = f"{data_dir}{l}/train.csv"
test_file = f"{data_dir}{l}/test.csv"

train_df = pd.read_csv(train_file)
test_df = pd.read_csv(test_file)

api_dataframe = pd.concat([train_df, test_df], axis=0)
api_dataframe.reset_index(inplace=True, drop=True)

sorted_labels = api_dataframe.groupby('ServiceClassification')['ServiceClassification'].count().sort_values(
    ascending=False)


label_encoder = LabelEncoder()
values = np.array(api_dataframe.ServiceClassification)
api_dataframe['y'] = label_encoder.fit_transform(values)

training_data = api_dataframe.iloc[0:len(train_df)]
testing_data = api_dataframe.iloc[len(train_df):]

op_file_path = f"/home/aa7514/PycharmProjects/kdd_project/files/emb_tiny{l}.npy"
# op_file_path = f"/home/aa7514/PycharmProjects/kdd_project/files/emb{l}.npy"
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
# label_encoder = LabelEncoder()
# values = np.array(training_data.ServiceClassification)
# y_train = label_encoder.fit_transform(values)
y_train = training_data['y']

x_test = word_vectors[testing_data["ServiceDescription"].index.values.tolist()]
# label_encoder = LabelEncoder()
# values = np.array(testing_data.ServiceClassification)
# y_test = label_encoder.fit_transform(values)
y_test = testing_data['y']

# pdb.set_trace()

print("label count: ", len(np.unique(y_train)))

# Hyper Params:
# max_iter = [100, 150, 200, 250, 300, 350, 400, 450, 500]
# max_iter = [200]
max_iter = [550]
# class_weight = ["balanced", None]
class_weight = [None]
# c_param = [1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
# c_param = [0.1537017058909389]
c_param = [0.8595229470023127]

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

for itr in max_iter:
    for cpm in c_param:
        for wt in class_weight:
            # clf = svm.SVC(C=1, gamma="scale", class_weight="balanced", max_iter=250)
            print("training...")
            start_time = time.time()
            clf = svm.LinearSVC(C=cpm, class_weight=wt, max_iter=itr)
            classifier = clf.fit(x_train, y_train)
            pdn = clf.predict(x_test)
            train_acc = clf.score(x_train, y_train)
            test_acc = clf.score(x_test, y_test)
            f_score = f1_score(y_test, pdn, average='weighted')
            precision = precision_score(y_test, pdn, average='weighted')
            recall = recall_score(y_test, pdn, average='weighted')
            # print("Train: ", train_acc)
            # print("Test: ", test_acc)
            # print("F1 score: ", f1_score)
            # print("Precision: ", precision)
            # print("Recall: ", recall)
            result_dict['max_iter'].append(str(itr))
            result_dict['class_weight'].append(str(wt))
            result_dict['c'].append(str(cpm))
            result_dict['Train'].append(str(train_acc))
            result_dict['Test'].append(str(test_acc))
            result_dict['F1 score'].append(str(f_score))
            result_dict['Precision'].append(str(precision))
            result_dict['Recall'].append(str(recall))

            csv_df = pd.DataFrame.from_dict(result_dict)
            # pdb.set_trace()
            csv_file_path = "/home/aa7514/PycharmProjects/kdd_project/plots/result_50_tiny_random_search.csv"
            csv_df.to_csv(csv_file_path, index=False)
            print("Time taken: ", time.time() - start_time)
            f1_scores = f1_score(y_test, pdn, average=None)
            print("F scores: ", list(f1_scores), "Len: ", len(f1_scores))
            print("F score Weighted: ", str(f_score))



pass

per_class_score = []
score_dict = {
    'Class': [],
    'Score': []
}

conf_label_order = []
for y in sorted_labels.index:
    label_idx = list(label_encoder.classes_).index(y)
    conf_label_order.append(label_idx)
    per_class_score.append((y, list(f1_scores)[label_idx]))
    print(y, list(f1_scores)[label_idx])
    score_dict['Class'].append(str(y))
    score_dict['Score'].append(str(list(f1_scores)[label_idx]))

score_df = pd.DataFrame.from_dict(score_dict)
csv_file_path2 = "/home/aa7514/PycharmProjects/kdd_project/plots/class_score_50_tiny_random_search.csv"
score_df.to_csv(csv_file_path2, index=False)

# print(per_class_score)

# pdb.set_trace()

# without max iter in search:
#                  Train             Test                 F1 score             Precision          Recall
# grid search:   0.54731029866979	0.4887043189368770	0.45534338985173300	0.4597240653136050	0.4887043189368770
# random search: 0.5480632477202380	0.48903654485049800	0.45632014255216900	0.459341798724965	0.48903654485049800

# with max iter in search:
# grid search:
# random search:

# random search with max iter (F1 score)/50 parameter samples:
# max_iter,class_weight,c,Train,Test,F1 score,Precision,Recall
# 700,None,1.1576243717888044
# 0.5619509746507153,0.4840531561461794,0.4572721214195841,0.4594292414071669,0.4840531561461794

# random search with max iter (F1 score)/100 parameter samples:
# 550,None, 0.8595229470023127,
# 0.5597757885049778,0.4870431893687708,0.4589428652068902,0.4637838280118621,0.4870431893687708

from sklearn.metrics import ConfusionMatrixDisplay

np.set_printoptions(precision=2)
# Plot non-normalized confusion matrix
titles_options = [
    ("Confusion matrix, without normalization", None),
    ("Normalized confusion matrix", "true"),
]
for title, normalize in titles_options:
    disp = ConfusionMatrixDisplay.from_estimator(
        classifier,
        x_test,
        y_test,
        # display_labels=label_encoder.classes_,
        display_labels=[""]*50,
        include_values=False,
        cmap=plt.cm.Blues,
        normalize=normalize,
        labels = conf_label_order
    )
    disp.ax_.set_title(title)

    print(title)
    print(disp.confusion_matrix)

plt.show()
plt.savefig('/home/aa7514/PycharmProjects/kdd_project/plots/conf_ordered_50_128.pdf', bbox_inches='tight')

pdb.set_trace()
