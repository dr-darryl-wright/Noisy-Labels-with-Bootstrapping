# Noisy-Labels-with-Bootstrapping
Keras implementation of [Training Deep Neural Networks on Noisy Labels with Bootstrapping, Reed et al. 2015](https://arxiv.org/pdf/1412.6596.pdf)

## MNIST with noisy Labels
Below are the results of experiments from the original paper using MNIST.

![alt text](https://github.com/dwright04/Noisy-Labels-with-Bootstrapping/blob/master/Reed_et_al_figure_2.png)

---

### Open questions for reproducing results
1) The models are trained on 60000 (MNIST training set) noisy labels.

2) The classification accuracy in Figure 2. is calculated on 10000 (MNIST test set) true labels. 

3) The architecture is 784-500-300-10 + a softmax layer with 10 units, and not 784-500-300 + a softmax layer with 10 units i.e. the 10 unit layer in the architecture listed in the paper is hidden and not the output layer.

4) The classification accuracy plotted is measured on a single realisation of the model and not the average of retraining multiple times on, for example, different partitions of the data (k-fold cross validation perhaps) or with different noise patterns. 

5) The size of the mini-batch has negligible effect and so choosing 32 or 256 shouldn't have a huge impact on the measured accuracy.

6) What is the stopping criteria for training? Assuming 1) and 2) are correct and there is no validation set.

## Replicated MNIST with noisy labels results
![alt text](https://github.com/dwright04/Noisy-Labels-with-Bootstrapping/blob/master/replicated_results.png)


#### Running log
**14-06-2018** Thinking a bit more about the data used to measure classification accuracy, I think it makes the most sense that it actually is measured on the test sense since the point is to show how reboust each approach is to learning on noisy data.  However with my current architecture a noise fraction 0f 0.3 only achieves ~75%.  But I may have missed adding a layer in the baseline model.  The paper says the architeture is 784-500-300-10 with relu activations.  However in my case I have set the activations to softmax for the layer with 10 units, taking it to be the classification layer.  I will now try adding an addtional layer with 10 units between the output layer and the layer with 300 units, to see if this extra capacity makes the model more robust to label noise.

**11-06-2018** Training as below does not reproduce the results from the paper.  The test accuracy does not evolve as smoothly with noise fraction as the results appear in the paper.  It seems more likely the results are measured on the training set, with no validation split in which case patience should be used to terminate training instead of checkpointing.  Updating the code as described in previous post.  It is unclear if the results should be measured on the noisy training labels or the orginal *clean* labels. The model achieves high accuracy on the noisy labels, if I were then to measure the performance on the original labels then performance would be much worse than is shown in the plot from the paper.  Only measuring the performance on the noisy labels does not make sense for the purpose of the paper, where the aim is to show that the proposed loss functions are robust to noise, we want to show that the perform better than the baseline model on the original labels.

**08-06-2018** Training for 500 epochs on 50000 examples with noise fraction of 0.3 achieves 99.93% classification accuracy measured on the training set with noisy labels.  This is higher than the equivalent result in Figure 2. from the paper shown above.
