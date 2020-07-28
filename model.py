import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model


class DIIN(Model):
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
        self.filed = fields_count
        self.embedding_size = embedding_size

        self.embedding_layer = {}
        for i in range(self.n_field):
            self.embedding_layer[field_name[i]] = Embedding(self.filed[i], self.embedding_size)

        self.deep_fc1 = Dense(300, activation='relu')
        self.deep_fc2 = Dense(300, activation='relu')
        self.deep_fc3 = Dense(300, activation='relu')

        self.cross_layer1 = Dense(300)
        self.cross_layer2 = Dense(300)
        self.cross_layer3 = Dense(300)

        self.out = Dense(1, activation='sigmoid')

    def call(self, inputs, training=None, mask=None):
        item, user, origin_items = inputs

        item_embedding = self.item_process(item)
        user_embedding = self.user_process(item)
        item_user_embedding = tf.concat([item_embedding, user_embedding], axis=1)

        # deep
        deep_embedding = self.deep_fc1(item_user_embedding)
        deep_embedding = self.deep_fc2(deep_embedding)
        deep_embedding = self.deep_fc3(deep_embedding)

        # cross
        cross_embedding = tf.matmul(tf.transpose(item_user_embedding), item_user_embedding)
        cross_embedding = self.cross_layer1(cross_embedding) + cross_embedding
        cross_embedding = self.cross_layer2(cross_embedding) + cross_embedding
        cross_embedding = self.cross_layer3(cross_embedding) + cross_embedding

        # fm
        fm_embedding = []
        for i in range(self.n_filed):
            for j in range(i, self.n_filed):
                fm_embedding.append(tf.reduce_sum(item_user_embedding[i] * item_user_embedding[j]))
        fm_embedding = tf.concat(fm_embedding, axis=1)

        # 合并 embedding
        item_user_embedding = tf.concat([deep_embedding, cross_embedding, fm_embedding, item_user_embedding], axis=1)

        # 12个item只做简单的embedding
        origin_items_embedding = []
        for item in origin_items:
            origin_items_embedding.append(self.item_process(item))
        origin_items_embedding = tf.concat(origin_items_embedding, axis=1)

        # dot attention
        origin_items_embedding = tf.transpose(origin_items_embedding)
        weight = tf.nn.softmax(tf.matmul(origin_items_embedding, item_user_embedding))
        attention_embedding = tf.matmul(origin_items_embedding, weight)

        embedding = tf.concat([attention_embedding, item_user_embedding], axis=1)
        return self.out(embedding)

    def item_process(self, item):
        embedding_0 = self.embedding_layer['0'](item[:, 0])
        embedding_1 = self.embedding_layer['1'](item[:, 1])
        embedding = tf.concat([embedding_0, embedding_1], axis=1)
        embedding = tf.concat([embedding, item[:, 2]], axis=1)
        return embedding

    def user_process(self, user):
        embedding_0 = self.embedding_layer['0'](user[:, 0])
        embedding_1 = self.embedding_layer['1'](user[:, 1])
        embedding = tf.concat([embedding_0, embedding_1], axis=1)
        embedding = tf.concat([embedding, user[:, 2]], axis=1)
        return embedding
