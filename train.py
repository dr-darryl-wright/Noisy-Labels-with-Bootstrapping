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

np.random.seed(0)

def noisify_mnist(noise_fraction, noise_mapping=None):
  # load the data
  (x_train, y_train), (x_test, y_test) = mnist.load_data()
  x_train = np.reshape(x_train, (x_train.shape[0], 784))
  x_test = np.reshape(x_test, (x_test.shape[0], 784))
  
  # add label noise
  if noise_mapping is None:
    noise_mapping = np.array([i for i in range(10)])
    np.random.shuffle(noise_mapping)
  noise_fractions = [noise_fraction for i in range(10)]
  
  print(noise_mapping)
  print(np.round(noise_fractions, 3))
  
  map = np.zeros((10,10))
  for i in range(10):
    map[i,noise_mapping[i].astype('int32')] += 1
  
  y_train_noisy = np.zeros(y_train.shape)
  for i in range(y_train.shape[0]):
    if np.random.rand() <= noise_fractions[y_train[i]]:
      #print('%d -> %d' % (y_train[i], noise_mapping[y_train[i]]))
      y_train_noisy[i] += noise_mapping[y_train[i]]
    else:
      #print('%d -> %d' % (y_train[i], y_train[i]))
      y_train_noisy[i] += y_train[i]
  return x_train, y_train, y_train_noisy, x_test, y_test, map, noise_mapping

def evaluate_noise_grid(model_getter, \
                        noise_grid=[x for x in np.arange(0.3,0.5,0.02)]):

  m = 50000 # validation split
  accs = []
  noise_mapping = None
  for noise_fraction in noise_grid:
    x_train, y_train, y_train_noisy, x_test, y_test, map, noise_mapping = \
      noisify_mnist(noise_fraction, noise_mapping)

    validation_data = (x_train[m:], np_utils.to_categorical(y_train[m:]))
    x_train = x_train[:m]
    y_train_noisy = y_train_noisy[:m]
    
    model, callbacks, trained, model_name = model_getter(noise_fraction)
    
    print(trained)
    if not trained:
      model.fit(x_train[:m], \
                np_utils.to_categorical(y_train_noisy[:m]), \
                validation_data=validation_data, \
                callbacks=callbacks, \
                batch_size=256, \
                epochs=500)

    weights_file = './%s/%s_noise_fraction_%.2lf.h5' \
                 % (model_name, model_name, noise_fraction)

    model.load_weights(weights_file)
    
    accs.append(model.evaluate(x_test, np_utils.to_categorical(y_test)))
    
  return noise_grid, accs

def baseline_model_getter(noise_fraction):

  # build the model for pre-training
  inputs = Input(shape=(784,))
  x = Dense(500, activation='relu', \
    kernel_regularizer=regularizers.l2(0.0001), \
    bias_regularizer=regularizers.l2(0.0001))(inputs)
  x = Dense(300, activation='relu', \
    kernel_regularizer=regularizers.l2(0.0001), \
    bias_regularizer=regularizers.l2(0.0001))(x)
  q = Dense(10, activation='softmax', name='q',\
    kernel_regularizer=regularizers.l2(0.0001), \
    bias_regularizer=regularizers.l2(0.0001))(x)

  model = Model(inputs, q)
  weights_file = \
    './baseline_model/baseline_model_noise_fraction_%.2lf.h5'%(noise_fraction)
  callbacks = None
  trained = False
  try:
    model.load_weights(weights_file)
    trained = True
  except OSError:
    callbacks = [ModelCheckpoint(weights_file, \
                                 save_best_only=True, \
                                 save_weights_only=True)]
  sgd = SGD(lr=0.01)
  model.compile(loss='categorical_crossentropy', \
                optimizer=sgd, \
                metrics=['acc'])
                  
  return model, callbacks, trained, 'baseline_model'

def plot_results(noise_grid, accs_list, model_names, colours):
  fig = plt.figure()
  ax1 = fig.add_subplot(1,3,1)
  for i,accs in enumerate(accs_list):
    ax1.plot(noise_grid, accs, '-', color=colours[i])
    ax1.plot(noise_grid, accs, 'o', color=colours[i], label=model_names[i])
  ax1.set_title('MNIST with random fixed label noise')
  ax1.set_ylabel('Classification accuracy (%)')
  ax1.set_xlabel('Noise fraction')
  plt.legend(loc='lower left')
  plt.axes().set_aspect('equal', 'datalim')
  plt.savefig('replicated_results.png')

def train_model_mnist_recon_loss():

  x_train, y_train, y_train_noisy, x_test, y_test, map, _ = \
    noisify_mnist(0.48)

  m = 50000 # validation split
  
  base_model = train_baseline_model(x_train[:m], y_train_noisy[:m], \
    validation_data=(x_train[m:], y_train[m:]))

  print(base_model.evaluate(x_test, np_utils.to_categorical(y_test)))
  
  # build the consistency model
  inputs = Input(shape=(784,))
  q = base_model(inputs)
  
  t_layer = Dense(10, activation='softmax', name='t', \
    kernel_regularizer=regularizers.l2(0.0001), \
    kernel_initializer='identity', \
    use_bias=False)

  t = t_layer(q)

  recon = Dense(784, activation='relu', name='recon', \
    kernel_regularizer=regularizers.l2(0.0001), \
    bias_regularizer=regularizers.l2(0.0001))(q)
  
  model = Model(inputs, [t, recon])
  
  sgd = SGD(lr=0.01)
  model.compile(optimizer=sgd, metrics=['acc'], \
    loss={'t': 'categorical_crossentropy', 'recon': 'mse'},\
    loss_weights={'t': 1., 'recon': 0.005}) # beta = 0.005

  model.summary()
  callbacks = [ModelCheckpoint('best_training.h5', save_best_only=True, \
    save_weights_only=True)]

  try:
    model.load_weights('best_training.h5')
  except OSError:
    model.fit(x_train[:m], [np_utils.to_categorical(y_train_noisy[:m]), x_train[:m]], \
      validation_data=(x_train[m:], [np_utils.to_categorical(y_train[m:]), x_train[m:]]),\
      callbacks=callbacks, batch_size=256, epochs=500)

  print(base_model.evaluate(x_test, np_utils.to_categorical(y_test)))
  print(base_model.evaluate(x_train[m:], np_utils.to_categorical(y_train[m:])))

  W = model.get_layer('t').get_weights()[0]

  fig = plt.figure()
  ax1 = fig.add_subplot(1,2,1)
  ax1.imshow(map, cmap='gray')
  ax2 = fig.add_subplot(1,2,2)
  ax2.imshow(W, cmap='gray')
  plt.show()

def main():
  
  noise_grid, accs = evaluate_noise_grid(baseline_model_getter)

  plot_results(noise_grid, [accs], ['baseline'], ['r'])

if __name__ == '__main__':
  main()
