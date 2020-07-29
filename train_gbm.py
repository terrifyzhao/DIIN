from model import DIIN
from fm import FM
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import Callback
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import *
from lightgbm.sklearn import LGBMClassifier
from sklearn.metrics import accuracy_score

df = pd.read_csv('data.csv')
df = df.sample(frac=1)

tf.config.experimental_run_functions_eagerly(True)


def age_process(age):
    if age <= 10:
        return 0
    elif 10 < age <= 18:
        return 1
    elif 18 < age < 30:
        return 2
    elif 30 < age < 50:
        return 3
    else:
        return 4


def gender_process(gender):
    if gender == 'M':
        return 0
    elif gender == 'F':
        return 1


def statistic_genres(genres):
    all_genres = []
    for g in genres:
        all_genres.extend(g.split('|'))

    all_genres = list(set(all_genres))
    dic = {}
    for g in all_genres:
        dic[g] = len(dic)
    return dic


def statistic_zip(zip):
    all_zip = []
    for z in zip:
        z = str(z)
        all_zip.append(int(z.split('-')[0]))

    all_zip = list(set(all_zip))
    dic = {}
    for z in all_zip:
        dic[z] = len(dic)
    return dic


genres_dic = statistic_genres(df['genres'].values)
zip_dic = statistic_zip(df['zip'].values)


def genres_process(genres):
    g = genres.split('|')[0]
    return genres_dic[g]


def zip_process(zip):
    z = str(zip)
    z = int(z.split('-')[0])
    return zip_dic[z]


df['gender'] = df['gender'].apply(gender_process)
df['age'] = df['age'].apply(age_process)
df['genres'] = df['genres'].apply(genres_process)
df['zip'] = df['zip'].apply(zip_process)

y = df['label'].values
df.drop(columns=['user_id', 'movie_id', 'rating', 'timestamp', 'title', 'label'], axis=1, inplace=True)
x = df.values

length = int(len(x) * 0.9)
x_train = x[0:length]
y_train = y[0:length]
x_test = x[length:]
y_test = y[length:]

model = LGBMClassifier(n_estimators=1200)
model.fit(x_train, y_train)
prediction = model.predict(x_test)
acc = accuracy_score(y_test, prediction)
print(acc)
