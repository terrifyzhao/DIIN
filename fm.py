import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model


class FM(Model):
    def __init__(self, field_name, fields_count, embedding_size, *args, **kwargs):
        """
        DIIN的构造方法
        :param n_field: 有多少个field
        :param fields_count: 每个field的数量
        :param embedding_size: 每个field的embedding size
        """
        super().__init__(*args, **kwargs)
        self.field_name = field_name
        self.n_field = len(field_name)
        self.fields_count = fields_count
        self.embedding_size = embedding_size

        self.embedding_layer = {}
        for i in range(self.n_field):
            self.embedding_layer[field_name[i]] = Embedding(self.fields_count[i], self.embedding_size)

        self.deep_fc1 = Dense(64, activation='relu')
        self.deep_fc2 = Dense(64, activation='relu')

        self.cross_layer1 = Dense(1)
        self.cross_layer2 = Dense(1)

        self.out = Dense(2, activation='sigmoid')

    def call(self, inputs, training=None, mask=None):
        item = inputs

        # [batch_size, 48]
        item_embedding = self.item_process(item)

        # deep
        deep_embedding = self.deep_fc1(item_embedding)
        deep_embedding = self.deep_fc2(deep_embedding)

        # cross
        expand_item_embedding = tf.expand_dims(item_embedding, axis=2)
        cross_embedding = tf.matmul(expand_item_embedding, tf.transpose(expand_item_embedding, perm=[0, 2, 1]))
        cross_embedding = self.cross_layer1(cross_embedding) + expand_item_embedding

        cross_embedding = self.cross_layer2(
            tf.matmul(cross_embedding, tf.transpose(cross_embedding, perm=[0, 2, 1]))) + cross_embedding

        # fm
        fm_embedding = []
        for i in range(self.n_field):
            for j in range(i + 1, self.n_field):
                fm_res = tf.reduce_sum(
                    item_embedding[:, i * self.embedding_size:(i + 1) * self.embedding_size] *
                    item_embedding[:, j * self.embedding_size:(j + 1) * self.embedding_size],
                    axis=1,
                    keepdims=True)
                fm_embedding.append(fm_res)
        fm_embedding = tf.concat(fm_embedding, axis=1)

        # 合并 embedding
        merge_embedding = tf.concat([tf.squeeze(cross_embedding), fm_embedding, deep_embedding, item_embedding], axis=1)
        # merge_embedding = tf.concat([item_embedding, fm_embedding], axis=1)
        # return merge_embedding
        return self.out(merge_embedding)

    def item_process(self, item):
        embedding_0 = self.embedding_layer['gender'](item[:, 0])
        embedding_1 = self.embedding_layer['age'](item[:, 1])
        embedding_2 = self.embedding_layer['occupation'](item[:, 2])
        embedding_3 = self.embedding_layer['genres'](item[:, 3])
        embedding_4 = self.embedding_layer['zip'](item[:, 4])
        embedding = tf.concat([embedding_0, embedding_1, embedding_2, embedding_3, embedding_4], axis=1)
        return embedding
