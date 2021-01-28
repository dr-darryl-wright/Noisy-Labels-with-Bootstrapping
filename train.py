import numpy as np
import matplotlib.pyplot as plt

from matplotlib.ticker import FormatStrFormatter

from keras.layers import Input, Dense, Lambda
from keras.models import Model
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras import backend as K
from keras import regularizers

from keras.datasets import mnist
from keras.utils import np_utils

def get_noise_mapping(seed=1):
  np.random.seed(seed)
  noise_mapping = np.array([i for i in range(10)])
  np.random.shuffle(noise_mapping)
  return noise_mapping

def noisify_labels(y, noise_fractions, noise_mapping, seed=1):
  
  np.random.seed(seed)
  y_noisy = np.zeros(y.shape)
  noisy_sum = 0
  clean_sum = 0
  for i in range(y.shape[0]):
    if np.random.rand() <= noise_fractions[y[i]]:
      #print('%d -> %d' % (y_train[i], noise_mapping[y_train[i]]))
      y_noisy[i] += noise_mapping[y[i]]
      noisy_sum += 1
    else:
      #print('%d -> %d' % (y_train[i], y_train[i]))
      y_noisy[i] += y[i]
      clean_sum += 1
  #print(noisy_sum, clean_sum, noisy_sum/float(y.shape[0]), clean_sum/float(y.shape[0]))
  return y_noisy

def noisify_mnist(noise_fraction):
  # load the data
  (x_train, y_train), (x_test, y_test) = mnist.load_data()

  x_train = np.reshape(x_train, (x_train.shape[0], 784)) / 255.
  x_test = np.reshape(x_test, (x_test.shape[0], 784)) / 255.
  
  noise_mapping = get_noise_mapping()
  noise_fractions = [noise_fraction for i in range(10)]
  
  print()
  print(noise_mapping)
  print(np.round(noise_fractions, 3))
  
  map = np.zeros((10,10))
  for i in range(10):
    map[i,noise_mapping[i].astype('int32')] += 1
  
  y_train_noisy = noisify_labels(y_train, noise_fractions, noise_mapping)
  return x_train, y_train, y_train_noisy, x_test, y_test, map, noise_mapping

def evaluate_noise_grid(model_getter, noise_grid, train=False, n_trials=5):

  accs = []
  noise_mapping = None
  for noise_fraction in noise_grid:
    x_train, y_train, y_train_noisy, x_test, y_test, map, noise_mapping = \
      noisify_mnist(noise_fraction)
    
    #validation_data = (x_test, np_utils.to_categorical(y_test))
    validation_data = (x_train[:10000], np_utils.to_categorical(y_train_noisy[:10000]))
    x_train = x_train[10000:]
    y_train = y_train[10000:]
    y_train_noisy = y_train_noisy[10000:]
    if train:
      best_acc = 0.0
      for i in range(1,n_trials+1):
        model, model_name = model_getter()
        weights_file = \
          './baseline_model/trials/%d/best_baseline_model_noise_fraction_%.2lf_trial_%d.h5'%(i, noise_fraction, i)
        callbacks = [ModelCheckpoint(weights_file, 'val_acc', save_best_only=True), \
                     EarlyStopping('val_acc', mode='auto', patience=5)]
        model.fit(x_train, \
                  np_utils.to_categorical(y_train_noisy), \
                  validation_data=validation_data, \
                  callbacks=callbacks, \
                  batch_size=32, \
                  epochs=100)
        model.load_weights(weights_file)
        acc = model.evaluate(x_test, np_utils.to_categorical(y_test))[1]
        if  acc > best_acc:
          print('\naccuracy: %.3lf -> %.3lf'%(best_acc, acc))
          weights_file = \
            './baseline_model/best_baseline_model_noise_fraction_%.2lf.h5'%(noise_fraction)
          best_acc = acc
          model.save(weights_file)
    weights_file = \
      './baseline_model/best_baseline_model_noise_fraction_%.2lf.h5'%(noise_fraction)
    model, model_name = model_getter(weights_file)
    
    acc = model.evaluate(x_test, np_utils.to_categorical(y_test))[1]
    accs.append(acc)
  return noise_grid, accs

def baseline_model_getter(weights_file=None):

  # build the model for pre-training
  inputs = Input(shape=(784,))
  x = Dense(500, activation='relu', \
    bias_regularizer=regularizers.l2(0.0001), \
    kernel_regularizer=regularizers.l2(0.0001))(inputs)
  x = Dense(300, activation='relu', \
    bias_regularizer=regularizers.l2(0.0001), \
    kernel_regularizer=regularizers.l2(0.0001))(x)
  x = Dense(10, activation='relu', \
    bias_regularizer=regularizers.l2(0.0001), \
    kernel_regularizer=regularizers.l2(0.0001))(x)
  q = Dense(10, activation='softmax', name='q', \
    bias_regularizer=regularizers.l2(0.0001), \
    kernel_regularizer=regularizers.l2(0.0001))(x)

  model = Model(inputs, q)

  optimizer = SGD(lr=0.1)
  model.compile(loss='categorical_crossentropy', \
                optimizer=optimizer, \
                metrics=['acc'])

  if weights_file:
    model.load_weights(weights_file)

  return model, 'baseline_model'

def bootstrap_recon_model_getter(noise_fraction):

  baseline_model, _, trained, _ = baseline_model_getter(noise_fraction)

  try:
    assert(trained == True)
  except AssertionError:
    exit('Baseline models must be trained first.')

  # build the consistency model
  inputs = Input(shape=(784,))
  q = baseline_model(inputs)
  
  t_layer = Dense(10, activation='softmax', name='t', \
    kernel_regularizer=regularizers.l2(0.0001), \
    kernel_initializer='identity', \
    use_bias=False)

  t = t_layer(q)

  recon = Dense(784, activation='relu', name='recon', \
    kernel_regularizer=regularizers.l2(0.0001), \
    bias_regularizer=regularizers.l2(0.0001))(q)
  
  model = Model(inputs, [t, recon])

  weights_file = \
    './bootstrap_recon_model/bootstrap_recon_noise_fraction_%.2lf.h5' \
    %(noise_fraction)

  callbacks = None
  trained = False
  try:
    model.load_weights(weights_file)
    trained = True
  except OSError:
    callbacks = [ModelCheckpoint(weights_file, 'loss', save_best_only=True), \
                 EarlyStopping('loss', mode='auto', patience=5), \
                 ReduceLROnPlateau(monitor='loss')]
  
  sgd = SGD(lr=0.01)
  beta = 0.005
  model.compile(optimizer=sgd, metrics=['acc'], \
    loss={'t': 'categorical_crossentropy', 'recon': 'mse'},\
    loss_weights={'t': 1., 'recon': beta})
                  
  return model, callbacks, trained, 'bootstrap_recon_model'

def plot_results(noise_grid, accs_list, model_names, colours):
  fig = plt.figure(figsize=(15, 5))
  ax1 = fig.add_subplot(1,3,1)
  ax2 = ax1.twinx().twiny()
  for i,accs in enumerate(accs_list):
    ax1.plot(noise_grid, accs, '-', color=colours[i])
    ax1.plot(noise_grid, accs, 'o', markeredgecolor=colours[i], \
             markerfacecolor='None', label=model_names[i])
  ax1.set_title('MNIST with random fixed label noise')
  ax1.set_ylabel('Classification accuracy (%)')
  ax1.set_xlabel('Noise fraction')
  ax1.set_ylim(0.4,1.0)
  ax1.set_xlim(0.3,0.5)
  ax1.xaxis.set_ticks(np.arange(0.3, 0.51, 0.02))
  xleft, xright = ax1.get_xlim()
  ybottom, ytop = ax1.get_ylim()
  # the abs method is used to make sure that all numbers are positive
  # because x and y axis of an axes maybe inversed.
  #ax1.set_aspect(abs((xright-xleft)/(ybottom-ytop))*1.0)
  ax1.legend(loc='lower left')
  ax1.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
  ax1.tick_params(direction='in')
  ax2.set_ylim(0.4,1.0)
  ax2.set_xlim(0.3,0.5)
  ax2.xaxis.set_ticks(np.arange(0.3, 0.51, 0.02))
  ax2.set_yticklabels([])
  ax2.set_xticklabels([])
  ax2.tick_params(direction='in')
  plt.savefig('replicated_results.png')

def train_model_mnist_recon_loss(noise_fraction):

  x_train, y_train, y_train_noisy, x_test, y_test, map, _ = \
    noisify_mnist(noise_fraction)

  m = 50000 # validation split
  
  #base_model = train_baseline_model(x_train[:m], y_train_noisy[:m], \
  #  validation_data=(x_train[m:], y_train[m:]))

  weights_file = \
      './baseline_model/best_baseline_model_noise_fraction_%.2lf.h5'%(noise_fraction)
  base_model, model_name = baseline_model_getter(weights_file)

  print(base_model.evaluate(x_test, np_utils.to_categorical(y_test)))
  
  # build the consistency model
  inputs = Input(shape=(784,))
  #q = base_model(inputs)
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
      callbacks=callbacks, batch_size=256, epochs=100)

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
  
  noise_grid, accs = evaluate_noise_grid(baseline_model_getter, \
                       noise_grid=[x for x in np.arange(0.3,0.51,0.02)], \
                       n_trials=1, \
                       train=True)
                       
  plot_results(noise_grid, [accs], ['baseline'], ['r'])
  
  noise_fraction = 0.40
  train_model_mnist_recon_loss(noise_fraction)

if __name__ == '__main__':
  main()
