
#https://www.tensorflow.org/tutorials/structured_data/imbalanced_data#optional_set_the_correct_initial_bias
from google.colab import files
uploaded = files.upload()


#that was for plot scattler lot matrix
from pandas.plotting import scatter_matrix
from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow import keras
import os
import tempfile
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

file = tf.keras.utils
raw_df = pd.read_csv('animo_507.csv')

raw_df.hist()

#raw_df.plot(kind= ' density ' , subplots=True, layout=(3,3), sharex=False)
#raw_df.plot(kind= ' box ' , subplots=True, layout=(3,3), sharex=False, sharey=False)
names = ['O1','O2','class']
scatter_matrix(raw_df)
plt.show()

raw_df[['O1','O2','class']].describe()
#plt.plot('O1')
plt.plot(raw_df['O1'])
#plt.plot(raw_df['O2'])

plt.plot(raw_df['O2'])

#los valores neg, pos, son valores unitario como
#se observa a continuacion.

neg, pos = np.bincount(raw_df['class'])
total = neg + pos
print('ejemplo:\n    Total: {}\n    Positivo: {} ({:.2f}% del total)\n'.format(
    total, pos, 100 * pos / total))
print('Negativo: {} '.format(neg))

cleaned_df = raw_df.copy()
plt.plot(cleaned_df)
#[10]===================================
copy_data = raw_df.copy()
#we will separete o1 and o2 and try take out the peak.
# inside our data we hve some noise, we need to take out for 
# a best classification.  
array_one = np.array(copy_data.pop('O1'))
array_two = np.array(copy_data.pop('O2'))

for lon in range(len(array_two)):
   count = np.argwhere(array_two > 4500)

numpycount = np.array(count)
index_ins = numpycount.copy()


raw_df = raw_df.drop(raw_df.index[index_ins])

plt.plot(raw_df['O2'])

#[11]=========================================

# Use a utility from sklearn to split and shuffle our dataset.
# sklearn.model_selection.train_test_split(*arrays, **options)[source]
print('cleaned_df')
print(cleaned_df)
# we will change the test_zize of 0.2 to 0.4 that for 
#the distribution of own data
train_df, test_df = train_test_split(cleaned_df, test_size=0.2)
print('train_df')
print(train_df)
print('test_df')
print(test_df)
train_df, val_df = train_test_split(train_df, test_size=0.2)
print('train_df')
print(train_df)
print('val_df')
print(val_df)
#print('test')
#print(test_df)
train_labels = np.array(train_df.pop('class'))
print('train_labels')
print(train_labels)
#print(bool_train_labels)
bool_train_labels = train_labels != 0
val_labels = np.array(val_df.pop('class'))
print('val_labels')
print(val_labels)
print('val_df.pop')
print(val_df)
test_labels = np.array(test_df.pop('class'))
#numpy.array(object, dtype=None, copy=True, order='K', subok=False, ndmin=0)
train_features = np.array(train_df)
print('train_features')
print(train_features)
val_features = np.array(val_df)
print('val_features')
print(val_features)
test_features = np.array(test_df)
print('test_features')
print(test_features)

scaler = StandardScaler()
print('train_features')
print(train_features)
train_features = scaler.fit_transform(train_features)
print('train_features')
print(train_features)
val_features = scaler.transform(val_features)
test_features = scaler.transform(test_features)
print('vl_features')
print(val_features)
train_features = np.clip(train_features, -5,5)
val_features = np.clip(val_features, -5, 5)
print('val_features')
print(val_features)
test_features = np.clip(test_features, -5, 5)
print('Training labels shape:', train_labels.shape)
print('Validation labels shape:', val_labels.shape)
print('Test labels shape:', test_labels.shape)
print('Training features shape:', train_features.shape)
print('Validation features shape:', val_features.shape)
print('Test features shape:', test_features.shape)

pos_df = pd.DataFrame(train_features[ bool_train_labels], columns = train_df.columns)
neg_df = pd.DataFrame(train_features[~bool_train_labels], columns = train_df.columns)
sns.jointplot(pos_df['O1'], pos_df['O2'],
              kind='hex', xlim = (-2,2), ylim = (-2,2))
plt.suptitle("Positive distribution")
sns.jointplot(neg_df['O1'], neg_df['O2'],
              kind='hex', xlim = (-2,2), ylim = (-2,2))
_ = plt.suptitle("Negative distribution")

#___________________
METRICS = [
      keras.metrics.TruePositives(name='tp'),
      keras.metrics.FalsePositives(name='fp'),
      keras.metrics.TrueNegatives(name='tn'),
      keras.metrics.FalseNegatives(name='fn'), 
      keras.metrics.BinaryAccuracy(name='accuracy'),
      keras.metrics.Precision(name='precision'),
      keras.metrics.Recall(name='recall'),
      keras.metrics.AUC(name='auc'),
]

def make_model(metrics = METRICS, output_bias=None):
  if output_bias is not None:
    output_bias = tf.keras.initializers.Constant(output_bias)
  model = keras.Sequential([
      keras.layers.Dense(
          16, activation='relu',
          input_shape=(train_features.shape[-1],)),
      keras.layers.Dropout(0.5),
      keras.layers.Dense(1, activation='sigmoid',
                         bias_initializer=output_bias),
  ])

  model.compile(
      optimizer=keras.optimizers.Adam(lr=1e-3),
      loss=keras.losses.BinaryCrossentropy(),
      metrics=metrics)

  return model
  #___________________
  EPOCHS = 100
BATCH_SIZE = 2048
#BATCH_SIZE = 3096
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_auc', 
    verbose=1,
    patience=10,
    mode='max',
    restore_best_weights=True)
#__________________
model = make_model()
model.summary()
#_________________________
model.predict(train_features[:20])
#_______________________
results = model.evaluate(train_features, train_labels, batch_size=BATCH_SIZE, verbose=0)
print("Loss: {:0.4f}".format(results[0]))
#__________________
initial_bias = np.log([pos/neg])
initial_bias
#____________________
model = make_model(output_bias = initial_bias)
model.predict(train_features[:10])
#____________________
results = model.evaluate(train_features, train_labels, batch_size=BATCH_SIZE, verbose=0)
print("Loss: {:0.4f}".format(results[0]))
#________________________
initial_weights = os.path.join(tempfile.mkdtemp(),'initial_weights')
model.save_weights(initial_weights)
#plot_metrics(initial_weights)
#_____________________-
model = make_model()
model.load_weights(initial_weights)
model.layers[-1].bias.assign([0.0])
zero_bias_history = model.fit(
    train_features,
    train_labels,
    batch_size=BATCH_SIZE,
    epochs=20,
    validation_data=(val_features, val_labels), 
    verbose=0)
#____________________________
model = make_model()
model.load_weights(initial_weights)
careful_bias_history = model.fit(
    train_features,
    train_labels,
    batch_size=BATCH_SIZE,
    epochs=20,
    validation_data=(val_features, val_labels), 
    verbose=0)
#______________________________
def plot_loss(history, label, n):
  # Use a log scale to show the wide range of values.
  plt.semilogy(history.epoch,  history.history['loss'],
               color=colors[n], label='Train '+label)
  plt.semilogy(history.epoch,  history.history['val_loss'],
          color=colors[n], label='Val '+label,
          linestyle="--")
  plt.xlabel('Epoch')
  plt.ylabel('Loss')
  
  plt.legend()
  #______________________________
  plot_loss(zero_bias_history, "Zero Bias", 0)
plot_loss(careful_bias_history, "Careful Bias", 1)
#___________________________
plt.plot(val_features)
#_____________________________-
model = make_model()
model.load_weights(initial_weights)
baseline_history = model.fit(
    train_features,
    train_labels,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    callbacks = [early_stopping],
    validation_data=(val_features, val_labels))
#________________________________
