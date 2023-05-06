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


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

# ==============

# ==============
# Part 3 answers

part3_q1 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

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


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part4_q2 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part4_q3 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

# ==============
