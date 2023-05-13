r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""

# ==============
# Part 1 answers

part1_q1 = r"""

1. False. The test set is used to estimate the out-of-sample error (generalization error) of a model.
in-sample error (training error) is the error rate of a model on the training data.
2. False. For example, suppose we have a dataset with 1000 samples and we split it randomly into a train set with 50 samples and a test set with 950 samples.
In this case, the small train set may not be representative of the distribution of the data, and could lead to overfitting.
3. True. The test set is used to evaluate the final model's performance after all the model selection and hyperparameter tuning is done.
During cross-validation, the data is split into k folds, and k-1 folds are used for training and the remaining fold is used for validation.
The test-set is held out for the final evaluation and should not be used for model selection or hyperparameter tuning to prevent data leakage.
4. 
"""

part1_q2 = r"""
The approach is not justified, as the model selection process should be performed without using the test set.
Using the test set to select the regularization hyperparameter $\lambda$ could result in overfitting to the test set, which in turn could lead to poor performance on new, unseen data.

Instead, one should use a separate validation set or perform cross-validation to select the best value of $\lambda$.
This approach ensures that the model is not overfitting to the test set and provieds a more realistic estimate to the model's performance on new unseen data.

"""

# ==============
# Part 2 answers

part2_q1 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part2_q2 = r"""
**Your answer:**
Avoid overfitting in both cases.

1. When a model is trained on the entire train-set and its performance is evaluated on the same set, it may lead to overfitting, where the model **weights** are tailored to the train-set but fails to generalize to unseen data.

2. When selecting the best model based on test-set accuracy, there is a risk that the model is overly tailored to the training and test data.
If the model is repeatedly trained on the same data and evaluated on the same data. The chosen **hyperparameters** performs well on the test set, but not necessarily on new, unseen data.

In contrast, k-fold CV helps in avoiding overfitting by providing a more realistic estimate of the model's overall performance (weights and hyperparameters)  on unseen data.

"""

# ==============

# ==============
# Part 3 answers

part3_q1 = r"""
**Your answer:**

The selection of $\Delta > 0$ in the SVM loss function is arbitrary because the regularization term controls the model's complexity, which affects the margin $\Delta$.
The regularization term encourages a simpler decision boundary, reducing the number of possible boundaries.
Different values of $\Delta$ may be suitable for different datasets or applications, but the regularization term ensures that the model can generalize well to unseen data.

"""

part3_q2 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part3_q3 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

# ==============

# ==============
# Part 4 answers

part4_q1 = r"""
**Your answer:**

The ideal pattern in a residual plot is a plot where 


"""

part4_q2 = r"""
**Your answer:**


"""

part4_q3 = r"""
**Your answer:**


"""

# ==============
