# Noisy-Labels-with-Bootstrapping
Keras implementation of Training Deep Neural Networks on Noisy Labels with Bootstrapping, Reed et al. 2015

## MNIST with noisy Labels
Below are the results of experiments from the original paper using MNIST.

![alt text](https://github.com/dwright04/Noisy-Labels-with-Bootstrapping/blob/master/Reed_et_al_figure_2.png)

---

### Open questions for reproducing results
* What data is the classification accuracy in Figure 2. measured on?
* How do they determine when to stop training?
* What is the mini-batch size?

## Replicated MNIST with noisy labels results
![alt text](https://github.com/dwright04/Noisy-Labels-with-Bootstrapping/blob/master/replicated_results.png)


#### Running log
*11-06-2018* Training as anove does not reproduce the results from the paper.  The validation accuracy does not evolve as smoothly with noise fraction as the results appear in the paper.  It seems more likely the results are measure on the training set, with no validation split in which case patience should be used to terminate training instead of checkpointing.

*08-06-2018* Training for 500 epochs on 50000 examples with noise fraction of 0.3 achieves 99.93% classification accuracy measured on the training set with noisy labels.  This is higher than the equivalent result in Figure 2. from the paper shown above.
