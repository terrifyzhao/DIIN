from model import DIIN
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import Callback

x = []
y = []
model = DIIN(field_name=['name', 'sex'], fields_count=['100', '2'], embedding_size=32)

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
          batch_size=512,
          validation_split=0.1,
          epochs=10,
          callbacks=[call_back])
