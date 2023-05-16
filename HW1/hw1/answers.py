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
4. True. We can use splits to approximate the generalization error. 
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
Increasing the value of k can lead to improved generalization for unseen data, but this depends on the k. For a small k, the model may overfit as it is sensitive to noise and outliers. Overfitting can impact performance on unseen data because the model may be using noise from the training data to make decisions on unseen data. As k increases the model is better able to handle noise and outlier because more neighbors are used to make a prediction. When more neighbors are used, the generalization is better as the model is less likely to use noise to decide on unseen data. However, when k is too large, the model may start to ignore patterns and meaningful relationships in the data which can also impact performance negatively and underfit.
"""

part2_q2 = r"""
Avoid overfitting in both cases.

1. When a model is trained on the entire train-set and its performance is evaluated on the same set, it may lead to overfitting, where the model **weights** are tailored to the train-set but fails to generalize to unseen data.

2. When selecting the best model based on test-set accuracy, there is a risk that the model is overly tailored to the training and test data.If the model is repeatedly trained on the same data and evaluated on the same data. The chosen **hyperparameters** performs well on the test set, but not necessarily on new, unseen data.

In contrast, k-fold CV helps in avoiding overfitting by providing a more realistic estimate of the model's overall performance (weights and hyperparameters)  on unseen data.

"""

# ==============

# ==============
# Part 3 answers

part3_q1 = r"""
The selection of $\Delta > 0$ in the SVM loss function is arbitrary because the regularization term controls the model's complexity, which affects the margin $\Delta$.
The regularization term encourages a simpler decision boundary, reducing the number of possible boundaries.
Different values of $\Delta$ may be suitable for different datasets or applications, but the regularization term ensures that the model can generalize well to unseen data.

"""

part3_q2 = r"""
1. The linear model learns a linear decision boundary and attempts to distinguish between different classes. The model in our case is learning the weights of the data which are scanned handwritten digits represented as pixels with varied intensity (representing how light or dark the pixel is). As the model is learning the weights for each pixel, it assigns weights according to relationships and may put more weight to pixels that make up the border of a digit or the center (for example). This can lead to potential classification errors when images are rotated, or if a digit was written at an angle. This can also cause classification errors between digits that share similarities such as 3 and 8 or 5 and 6. 

2. kNN , in general, also tries to identify classes based on a patters in the input data and its features, but kNN does not learn a decision boundary, but runs over many instances and tries to classify them based on their nearest neighbors. 
"""

part3_q3 = r"""
1. We can see that we converge quickly and that the validation and training loss decrease in the same direction which implies that our learning rate is *good*. When the learning rate is set to a high value, the model converges quickly, but it may only reach a local minimum, which may not have the best performance, and the optimization may miss the mininum. A low learning rate takes longer to converge but it may produce better results.

2. Based on the metrics, we can see that as we approach concergence the validation accuracy is lower than training accuracy so there may be some overfitting (slight overfitting).
"""

# ==============

# ==============
# Part 4 answers

part4_q1 = r"""
An ideal residual plot should show randomness, constant variance (homoscedasticity), a mean of zero, and independence. 
Randomness indicates that the model captures the relationship between variables without violating assumptions. 
Constant variance suggests that the spread of residuals is consistent across predicted values. 
A mean of zero indicates unbiased predictions. 
Independence means that residuals at one point do not provide information about residuals at other points. 

Based on the provided residual plots, we can make several observations regarding the fitness of the trained model.
Firstly, the plots indicate that the model performs relatively well in predicting mid and low-priced houses, as the residuals exhibit a good fit in those ranges.
However, there seems to be less accuracy when predicting high-priced houses. 
This discrepancy may be attributed to a limited representation of high prices in the available data, which affects the randomness of the residuals.
Despite this limitation, a positive aspect is that the residuals are centered around a zero mean, suggesting that the model does not suffer from a significant problem of over- or underestimation.

```
fig, axes = plt.subplots(nrows=1, ncols=2, sharey=True, figsize=(20,10))

y_pred = hw1linreg.fit_predict_dataframe(model,  df_boston, 'MEDV', top_feature_names)


plot_residuals(y, y_pred, ax=axes[0])

# calculate residual mean
res = y - y_pred
res_mean = res.mean()

# plot mean line in green
axes[0].hlines(y=res_mean, xmin=y_pred.min(), xmax=y_pred.max(), color='green', lw=3)

# Use the best hyperparameters
model.set_params(**best_hypers)

# Train best model on full training set
y_pred_train, mse, rsq = linreg_boston(model, x_train, y_train)
print(f'train: mse={mse:.2f}, rsq={rsq:.2f}')
ax = plot_residuals(y_train, y_pred_train, ax=axes[1], res_label='train')

# Evaluate on test set
y_pred_test, mse, rsq = linreg_boston(model, x_test, y_test, fit=False)
print(f'test:  mse={mse:.2f}, rsq={rsq:.2f}')
ax = plot_residuals(y_test, y_pred_test, ax=ax, res_label='test')

# plot mean line in green
axes[1].hlines(y=res_mean, xmin=y_pred.min(), xmax=y_pred.max(), color='green', lw=3)

```


"""

part4_q2 = r"""
Adding non-linear features to the data would result in a model that is no longer a pure linear regression model.
Linear regression assumes a linear relationship between the predictors and the response variable.
By introducing non-linear features, the relationship between the predictors and the response becomes non-linear, making it a form of polynomial regression or non-linear regression.
So, adding non-linear features changes the nature of the regression model.

Yes, by adding non-linear features, we can capture and fit non-linear functions of the original features.
For example, if we have a feature 'x', we can create additional features like 'x^2', 'x^3', or even more complex non-linear transformations like 'sqrt(x)' or 'log(x)'.
These non-linear features allow the model to capture and represent non-linear relationships between the predictors and the response.

```
from sklearn.linear_model import LinearRegression

# Generate a toy dataset with a cubic relationship
x = np.linspace(-10, 10, 100)
y = x**3 + np.random.normal(scale=50, size=100)

# Fit a linear regression model with x^3 as a feature
model = LinearRegression()
X = pd.DataFrame({'x': x, 'x_cubed': x**3})
model.fit(X, y)

# Print the coefficients for x and x^3
print(model.coef_)

# Make predictions on new data
x_new = np.linspace(-15, 15, 100)
X_new = pd.DataFrame({'x': x_new, 'x_cubed': x_new**3})
y_pred = model.predict(X_new)


plt.scatter(x, y)
plt.plot(x_new, y_pred, color='red')
plt.show()
```

In the case of a linear classification model, adding non-linear features would lead to a decision boundary that is no longer a simple hyperplane.
The decision boundary would become a more complex shape, potentially curved or nonlinear, due to the presence of the non-linear features.
This is because the non-linear features introduce additional dimensions and complexity to the model, allowing it to separate classes that may not be linearly separable in the original feature space.
We can think of polynomial features as an additional layer to our model which no longer keeps it a linear model.

"""

part4_q3 = r"""
1. This is beneficial for cross-validation as it allows us to explore a wider range of hyperparameter values with a smaller number of samples.
This is particularly useful for regularization parameters like $\lambda$, where the optimal value may lie on a logarithmic scale. 

2. The outer loop iterates over the combinations of degree and $\lambda$, resulting in $d \times l$ iterations.
Within each iteration of the outer loop, the model is fitted $k_{\text{folds}}$ times, where $k_{\text{folds}}$ is the number of folds used for cross-validation.
Therefore, the total number of times the model is fitted is $d \times l \times k_{\text{folds}}$.
There are $d$ degrees and $l$ $\lambda$ values, and $k_{\text{folds}}$ is set to 3, the model will be fitted $d \times l \times 3$ times in total.
degree_range has 3 values and the lambda_range has 20 values, the model will be fitted $3 \times 20 \times 3 = 180$ times in total.
Following cross-validation, the model undergoes a final fine-tuned fit on the complete training set, utilizing the selected hyperparameters.

"""

# ==============
