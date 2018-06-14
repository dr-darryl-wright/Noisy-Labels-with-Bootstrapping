# Noisy-Labels-with-Bootstrapping
Keras implementation of [Training Deep Neural Networks on Noisy Labels with Bootstrapping, Reed et al. 2015](https://arxiv.org/pdf/1412.6596.pdf)

## MNIST with noisy Labels
Below are the results of experiments from the original paper using MNIST.

![alt text](https://github.com/dwright04/Noisy-Labels-with-Bootstrapping/blob/master/Reed_et_al_figure_2.png)

---

### Open questions for reproducing results
* What data is the classification accuracy in Figure 2. measured on?
  * I'm measuring the classification accuracy on the noisy training labels.
* How do they determine when to stop training?
  * using early stopping monitoring training loss with a patience of 5. 
* What is the mini-batch size?
  * using 256

## Replicated MNIST with noisy labels results
![alt text](https://github.com/dwright04/Noisy-Labels-with-Bootstrapping/blob/master/replicated_results.png)


#### Running log
**11-06-2018** Training as below does not reproduce the results from the paper.  The test accuracy does not evolve as smoothly with noise fraction as the results appear in the paper.  It seems more likely the results are measured on the training set, with no validation split in which case patience should be used to terminate training instead of checkpointing.  Updating the code as described in previous post.  It is unclear if the results should be measured on the noisy training labels or the orginal *clean* labels. The model achieves high accuracy on the noisy labels, if I were then to measure the performance on the original labels then performance would be much worse than is shown in the plot from the paper.  Only measuring the performance on the noisy labels does not make sense for the purpose of the paper, where the aim is to show that the proposed loss functions are robust to noise, we want to show that the perform better than the baseline model on the original labels.

**08-06-2018** Training for 500 epochs on 50000 examples with noise fraction of 0.3 achieves 99.93% classification accuracy measured on the training set with noisy labels.  This is higher than the equivalent result in Figure 2. from the paper shown above.
