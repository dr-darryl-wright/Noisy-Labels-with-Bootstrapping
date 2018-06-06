import numpy as np
import matplotlib.pyplot as plt

from keras.layers import Input, Dense, Lambda
from keras.models import Model
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint
from keras import backend as K
from keras import regularizers

from keras.datasets import mnist
from keras.utils import np_utils

def noisify_mnist():
  # load the data
  (x_train, y_train), (x_test, y_test) = mnist.load_data()
  x_train = np.reshape(x_train, (x_train.shape[0], 784))
  x_test = np.reshape(x_test, (x_test.shape[0], 784))
  
  # add label noise
  np.random.seed(0)
  noise_mapping = np.array([i for i in range(10)])
  np.random.shuffle(noise_mapping)
  noise_levels = [0.48 for i in range(10)]
  print(noise_mapping)
  print(np.round(noise_levels, 3))
  map = np.zeros((10,10))
  for i in range(10):
    map[i,noise_mapping[i]] += 1
  
  y_train_noisy = np.zeros(y_train.shape)
  for i in range(y_train.shape[0]):
    if np.random.rand() <= noise_levels[y_train[i]]:
      #print('%d -> %d' % (y_train[i], noise_mapping[y_train[i]]))
      y_train_noisy[i] += noise_mapping[y_train[i]]
    else:
      #print('%d -> %d' % (y_train[i], y_train[i]))
      y_train_noisy[i] += y_train[i]
  return x_train, y_train, y_train_noisy, x_test, y_test, map, noise_mapping

def train_baseline_model(x_train, y_train, validation_data):
  # build the model for pre-training
  inputs = Input(shape=(784,))
  x = Dense(500, activation='relu', \
    kernel_regularizer=regularizers.l2(0.0001), \
    bias_regularizer=regularizers.l2(0.0001))(inputs)
  x = Dense(300, activation='relu', \
    kernel_regularizer=regularizers.l2(0.0001), \
    bias_regularizer=regularizers.l2(0.0001))(x)
  q = Dense(10, activation='relu', name='q',\
    kernel_regularizer=regularizers.l2(0.0001), \
    bias_regularizer=regularizers.l2(0.0001))(x)

  base_model = Model(inputs, q)
  
  sgd = SGD(lr=0.01)
  base_model.compile(loss='categorical_crossentropy', \
    optimizer=sgd, metrics=['acc'])

  # train the model on the noisy labels
  callbacks = [ModelCheckpoint('best_pre-training.h5', save_best_only=True, \
    save_weights_only=True)]

  try:
    base_model.load_weights('best_pre-training.h5')
  except FileNotFoundError:
    base_model.fit(x_train, np_utils.to_categorical(y_train), \
      validation_data=validation_data, \
      callbacks=callbacks, batch_size=256, epochs=500)
  
  return base_model
  
def train_model_mnist_recon_loss():

  x_train, y_train, y_train_noisy, x_test, y_test, map, _ = \
    noisify_mnist()

  m = 50000
  
  base_model = train_baseline_model(x_train[:m], y_train_noisy[:m], \
    (x_train[m:], y_train[m:]))

  print(base_model.evaluate(x_test, np_utils.to_categorical(y_test)))
  
  # build the consistency model
  inputs = Input(shape=(784,))
  q = base_model(inputs)
  
  t_layer = Dense(10, activation='softmax', name='t', \
    kernel_regularizer=regularizers.l2(0.0001), \
    kernel_initializer='identity', \
    use_bias=False)
   
  #recon_layer = Dense(784, activation='relu', name='recon', \
  #  kernel_regularizer=regularizers.l2(0.0001), \
  #  bias_regularizer=regularizers.l2(0.0001))

  #recon_layer = Lambda(lambda x:x+0, name='recon')

  #t_layer.build(input_shape=q._keras_shape)
  #recon_layer.build(input_shape=q._keras_shape)
  
  #recon_layer.kernel = t_layer.kernel
  #recon_layer.bias   = t_layer.bias
  #recon_layer._trainable_weights = []
  #recon_layer._trainable_weights.append(recon_layer.kernel)
  #recon_layer._trainable_weights.append(recon_layer.bias)

  #print(recon_layer.weights == t_layer.weights)

  t = t_layer(q)
  #recon = recon_layer(q)
  """
  x = Dense(10, activation='softmax', \
    kernel_regularizer=regularizers.l2(0.0001), \
    bias_regularizer=regularizers.l2(0.0001))(q)
  x = Dense(300, activation='relu', \
    kernel_regularizer=regularizers.l2(0.0001), \
    bias_regularizer=regularizers.l2(0.0001))(x)
  x = Dense(500, activation='relu', \
    kernel_regularizer=regularizers.l2(0.0001), \
    bias_regularizer=regularizers.l2(0.0001))(x)
  """
  recon = Dense(784, activation='relu', name='recon', \
    kernel_regularizer=regularizers.l2(0.0001), \
    bias_regularizer=regularizers.l2(0.0001))(q)
  
  model = Model(inputs, [t, recon])
  
  sgd = SGD(lr=0.01)
  model.compile(optimizer=sgd, metrics=['acc'], \
    loss={'t': 'categorical_crossentropy', 'recon': 'mse'},\
    loss_weights={'t': 1., 'recon': 0.005})
  """
  model.compile(optimizer='adam', metrics=['acc'], \
    loss={'t': 'categorical_crossentropy', 'recon': 'mse'},\
    loss_weights={'t': 1., 'recon': 0.005})
  """
  model.summary()
  callbacks = [ModelCheckpoint('best_training.h5', save_best_only=True, \
    save_weights_only=True)]

  """
  model.fit(x_train[:50000], [np_utils.to_categorical(y_train_noisy[:50000]), x_train[:50000]], \
    validation_data=(x_train[50000:], [np_utils.to_categorical(y_train[50000:]), x_train[50000:]]),\
    callbacks=callbacks, batch_size=256, epochs=500)
  """
  model.load_weights('best_training.h5')
  print(base_model.evaluate(x_test, np_utils.to_categorical(y_test)))
  print(base_model.evaluate(x_train[50000:], np_utils.to_categorical(y_train[50000:])))

  W = model.get_layer('t').get_weights()[0]

  fig = plt.figure()
  ax1 = fig.add_subplot(1,2,1)
  ax1.imshow(map, cmap='gray')
  ax2 = fig.add_subplot(1,2,2)
  ax2.imshow(W, cmap='gray')
  plt.show()

train_model_mnist_recon_loss()
