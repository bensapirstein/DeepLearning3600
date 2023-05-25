r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""

# ==============
# Part 1 answers

part1_q1 = r"""
1. The Jacobian tensor of the output will have the shape (N, out_features, in_features), which in our case is (128, 2048, 1024).

2. Total Memory (in GB) = $\frac{{128 \times 2048 \times 1024 \times 4}}{{1024^3}} = 1GB$
"""

# ==============
# Part 2 answers


def part2_overfit_hp():
    wstd, lr, reg = 0, 0, 0
    # TODO: Tweak the hyperparameters until you overfit the small dataset.
    # ====== YOUR CODE: ======
    wstd = 1e-1
    lr = 1e-1
    reg = 1e-1
    # ========================
    return dict(wstd=wstd, lr=lr, reg=reg)


def part2_optim_hp():
    wstd, lr_vanilla, lr_momentum, lr_rmsprop, reg, = 0, 0, 0, 0, 0

    # TODO: Tweak the hyperparameters to get the best results you can.
    # You may want to use different learning rates for each optimizer.
    # ====== YOUR CODE: ======
    wstd = 1e-1
    lr_vanilla = 5e-2
    lr_momentum = 1e-3
    lr_rmsprop = 1e-4
    reg = 5e-5
    
    # ========================
    return dict(
        wstd=wstd,
        lr_vanilla=lr_vanilla,
        lr_momentum=lr_momentum,
        lr_rmsprop=lr_rmsprop,
        reg=reg,
    )


def part2_dropout_hp():
    wstd, lr, = 0, 0
    # ====== YOUR CODE: ======
    wstd = 1e-1
    lr = 1e-2
    # ========================
    return dict(wstd=wstd, lr=lr)


part2_q1 = r"""
The no-dropout graph shows that the model is able to achieve 100% accuracy on the training set. This is expected, as the model is not penalized for overfitting and is able to memorize the training data perfectly thanks to its capacity.
However, the model does not generalize well to the test set, as it achieves about 25% accuracy. This is because the model has learned to rely on specific features of the training data that are not present in the test data.
The dropout graph shows that the model achieves lower accuracy on the training set, but a higher accuracy on the test set (for low-dropout). This is because dropout prevents the model from relying on any specific features of the training data.
dropout forces the model to learn to use all the features of the data, which makes it more generalizable to new data.

The low-dropout setting (0.4) has the best performance on the test set because the model is able to learn to use most of the features in the data without being penalized too much for overfitting.
The high-dropout setting (0.8) has the worst performance on the test set. This is because the model is too regularized and is not able to learn to use all the features properly.
However it is surprizing how well the high-dropout performs on the test-set given a very low accuracy on the training set. This is probably because the model is using the few active neurons to learn the most important features of the data, and is able to generalize well to the test set.
"""

part2_q2 = r"""
Yes it is possible. During the initial epochs of training, the model may still be in the process of learning and exploring the solution space. 
It could result in following a promising gradiant for several important features which leads to correct classifications and increase of accuracy, while simultaneously exploring unpromising directions in noisy features which would increase the loss.

"""
# ==============

# ==============
# Part 3 answers

part3_q1 = r"""
**Your answer:**

"""

part3_q2 = r"""
**Your answer:**

"""

part3_q3 = r"""
**Your answer:**

"""

part3_q4 = r"""
**Your answer:**

"""

part3_q5 = r"""
**Your answer:**

"""

part3_q6 = r"""
**Your answer:**

"""
# ==============
