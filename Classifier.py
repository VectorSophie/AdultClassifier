#필요 라이브러리 호출및 데이터 불러오기
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

data = pd.read_excel('/content/final_adult_content.xlsx')
#데이터 전처리
data.info()
data['Description'] = data['Description'].str.strip()

data[data['Description'].str.contains('[^a-zA-Z0-9\s]')].values[:10]
print(data.isnull ().sum())
print(data['Description'].duplicated().sum())
data.drop_duplicates(subset=['Description'], inplace=True)

data['Category'].value_counts ()
data['Category'] = data['Category'].replace(['Non_Adult','Adult'],[0,1])

x = data.Description.to_list()
y = data.Category.to_list()
xtrain, xtest, ytrain, ytest = train_test_split(x, y, stratify = y, test_size = 0.2, random_state = 28)
xtrain[:2]
ytrain[:2]
#bert모델, 토나이저 불러오기
from transformers import AutoConfig, BertTokenizerFast, TFBertForSequenceClassification
bert_model = 'bert-base-uncased'
tokenizer = BertTokenizerFast.from_pretrained(bert_model)

trainencode = tokenizer(xtrain, truncation=True, padding=True, )
testencode = tokenizer(xtest, truncation=True, padding=True)

train_dataset = tf.data.Dataset.from_tensor_slices((dict(trainencode), ytrain))
train_dataset = train_dataset.shuffle(1000).batch(16).cache().prefetch(tf.data.experimental.AUTOTUNE)

test_dataset = tf.data.Dataset.from_tensor_slices((dict(testencode), ytest))
test_dataset = test_dataset.batch(16).cache().prefetch(tf.data.experimental.AUTOTUNE)

config = AutoConfig.from_pretrained(bert_model)
config
#모델링
model = TFBertForSequenceClassification.from_pretrained(bert_model, num_labels=2, from_pt=True)
from transformers import TFBertForSequenceClassification

optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5)
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True)
model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
model.fit(train_dataset, epochs=1, batch_size=16, validation_data=test_dataset)
model.summary()
#테스트데이터랑 맞춰보기
pred = model.predict(test_dataset)
predtable = pd.DataFrame(np.argmax(pred.logits, axis=1), columns=['predict'])
predtable['true'] = ytest
predtable

