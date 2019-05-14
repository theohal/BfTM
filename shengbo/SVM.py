import data_loader
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectFdr, SelectKBest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)


# dataset = pd.read_csv('merge_features.csv')
# dataset = dataset.drop(columns='Sample')

dataset = data_loader.load_dataset()
target = data_loader.load_target()

# y = dataset['Subgroup']
# X = dataset.drop(columns='Sample')
X = dataset
y = target

classes = np.unique(y)
classes = classes.tolist()

X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=0)

classifier = LogisticRegression(solver='liblinear', C=1, multi_class='auto')

features_name = ['17_35076296_35282086','17_41062669_41447005',
    '22_37683612_37705310','1_149369522_149394958','6_135286400_135498058',
    '12_64727853_66012212','3_196908262_196937230','17_39280577_40847517',
    '16_69763173_70593293','6_64633346_64705894','16_68710277_68751390',
    '12_70462057_70560603','22_41293579_41298170','16_30561643_31835555',
    '17_41458670_41494331','9_42913008_43066618','15_32447707_32511475',
    '16_67314736_68371045','5_150224239_152555267','12_85450052_85962613',
    '12_75396411_75693696','8_42123757_42175380','16_69391442_69397102',
    '11_83733321_85064586','6_32732764_32738443','16_68880321_69381055',
    '12_85986743_87258350','17_41983407_42049540','20_194755_763411',
    '23_3229_45135','15_32605086_32654561','23_66527817_67559319',
    '1_199087690_199111062','6_40747924_42826227','8_36512939_36612488',
    '15_44193179_48393597','16_70601469_71161060','1_72489681_72620747',
    '23_131385521_132949963','11_78137964_78627898']
features_name[:10]

iterations = []
scores = []
i = 0
j = 20
while j <= len(features_name):
    X_train_limited = X_train[features_name[i:j]]
    classifier.fit(X_train_limited, y_train)
    X_test_limited = X_test[features_name[i:j]]
    Y_pred = classifier.predict(X_test_limited)
    lr_report = classification_report(y_pred=Y_pred, y_true=y_test, labels=classes)

    score = classifier.score(X_test_limited, y_test)
    iterations.append(str(i) + '-' + str(j))
    scores.append(score)

    i = i + 20
    j = j + 20

adding_features = pd.DataFrame(data={'Selected features': iterations, 'Tesing scores': scores})


X_train_limited = X_train[features_name[0:20]]
X_test_limited = X_test[features_name[0:20]]
classifier.fit(X_train_limited, y_train)
Y_pred = classifier.predict(X_test_limited)
lr_report = classification_report(y_pred=Y_pred, y_true=y_test, labels=classes)
print(lr_report)

X_train_limited = X_train[features_name[20:40]]
X_test_limited = X_test[features_name[20:40]]
classifier.fit(X_train_limited, y_train)
Y_pred = classifier.predict(X_test_limited)
lr_report = classification_report(y_pred=Y_pred, y_true=y_test, labels=classes)
print(lr_report)


X_train_limited = X_train[features_name[20:30]]
X_test_limited = X_test[features_name[20:30]]
classifier.fit(X_train_limited, y_train)
Y_pred = classifier.predict(X_test_limited)
lr_report = classification_report(y_pred=Y_pred, y_true=y_test, labels=classes)
print(lr_report)













X_test_limited = X_test[features_name[10:20]]

X.shape

X_new = SelectKBest(k=100).fit(X_train, y_train)

features_list = X.columns
scores = X_new.scores_
pairs = zip(features_list[1:], scores)

k_best_features = pd.DataFrame(list(pairs), columns=['feature', 'score'])
k_best_features = k_best_features.sort_values('score', ascending=False)

k_best_features['feature'][2140]

selectedFeatures = pd.DataFrame({'feature': X['17_35286565_35336158'], 'Subgroup': y})
selectedFeatures.to_csv('selected_features.csv', index=False)

X_new.scores_
X_new.get_support([0])
