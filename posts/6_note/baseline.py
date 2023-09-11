import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Input, LSTM, Dense
from tensorflow.keras import Model
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt


# 데이터 불러오기
inputs = pd.read_csv('./train_input.csv')
outputs = pd.read_csv('./train_output.csv')


# nan 제거  -- 베이스라인이므로 간단한 처리를 위해 nan 항목 보간 없이 학습
inputs = inputs.dropna(axis=1)


# 주차 정보 수치 변환
inputs['주차'] = [int(i.replace('주차', "")) for i in inputs['주차']]


# scaler
input_scaler = MinMaxScaler()
output_scaler = MinMaxScaler()


# scaling
input_sc = input_scaler.fit_transform(inputs.iloc[:,3:].to_numpy())
output_sc = output_scaler.fit_transform(outputs.iloc[:,3:].to_numpy())


# 입력 시계열화
input_ts = []
for i in outputs['Sample_no']:
    sample = input_sc[inputs['Sample_no'] == i]
    if len(sample < 7):
        sample = np.append(np.zeros((7-len(sample), sample.shape[-1])), sample,
                           axis=0)
    sample = np.expand_dims(sample, axis=0)
    input_ts.append(sample)
input_ts = np.concatenate(input_ts, axis=0)


# 셋 분리
train_x, val_x, train_y, val_y = train_test_split(input_ts, output_sc, test_size=0.2,
                                                  shuffle=True, random_state=0)


# 모델 정의
def create_model():
    x = Input(shape=[7, 9])
    l1 = LSTM(64)(x)
    out = Dense(3, activation='tanh')(l1)
    return Model(inputs=x, outputs=out)

model = create_model()
model.summary()
checkpointer = ModelCheckpoint(monitor='val_loss', filepath='baseline.h5',
                               verbose=1, save_best_only=True, save_weights_only=True)
model.compile(loss='mse', optimizer=Adam(lr=0.001), metrics=['mse'])


# 학습
hist = model.fit(train_x, train_y, batch_size=32, epochs=50, validation_data=(val_x, val_y), callbacks=[checkpointer])


# loss 히스토리 확인
fig, loss_ax = plt.subplots()
loss_ax.plot(hist.history['loss'], 'r', label='loss')
loss_ax.plot(hist.history['val_loss'], 'g', label='val_loss')
loss_ax.set_xlabel('epoch')
loss_ax.set_ylabel('loss')
loss_ax.legend()
plt.title('Training loss - Validation loss plot')
plt.show()


# 저장된 가중치 불러오기
model.load_weights('baseline.h5')


# 테스트셋 전처리 및 추론
test_inputs = pd.read_csv('./test_input.csv')
output_sample = pd.read_csv('./answer_sample.csv')

test_inputs = test_inputs[inputs.columns]
test_inputs['주차'] = [int(i.replace('주차', "")) for i in test_inputs['주차']]
test_input_sc = input_scaler.transform(test_inputs.iloc[:,3:].to_numpy())

test_input_ts = []
for i in output_sample['Sample_no']:
    sample = test_input_sc[test_inputs['Sample_no'] == i]
    if len(sample < 7):
        sample = np.append(np.zeros((7-len(sample), sample.shape[-1])), sample,
                           axis=0)
    sample = np.expand_dims(sample, axis=0)
    test_input_ts.append(sample)
test_input_ts = np.concatenate(test_input_ts, axis=0)

prediction = model.predict(test_input_ts)

prediction = output_scaler.inverse_transform(prediction)
output_sample[['생장길이', '줄기직경', '개화군']] = prediction


# 제출할 추론 결과 저장
output_sample.to_csv('prediction.csv', index=False)


