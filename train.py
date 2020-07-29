from model import DIIN
from fm import FM
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import Callback
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import *

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

print(len(df['zip'].unique()))

y = df['label'].values
df.drop(columns=['user_id', 'movie_id', 'rating', 'timestamp', 'title', 'label'], axis=1, inplace=True)
x = df.values

fm = FM(field_name=['gender', 'age', 'occupation', 'zip', 'genres'], fields_count=[2, 5, 21, 18, 3359],
        embedding_size=16)

x_input = Input(shape=(5,), batch_size=1024, dtype='int32')
output = fm(x_input)
model = Model(inputs=x_input, outputs=output)

model.compile(loss='sparse_categorical_crossentropy',
              optimizer=Adam(1e-2),
              metrics=['accuracy'])


class Evaluate(Callback):
    def __init__(self):
        super().__init__()
        self.lowest = 1e10

    def on_epoch_end(self, epoch, logs=None):
        # 保存最优
        if logs['loss'] <= self.lowest:
            self.lowest = logs['loss']
            model.save_weights('best_model.weights')
            print('save model success')


call_back = Evaluate()

model.fit(x,
          y,
          batch_size=1024,
          validation_split=0.1,
          epochs=20)
