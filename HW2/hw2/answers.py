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
A regular block performing two convolutions with $3\times3$ kernels on an input channel of 256 has $2 \cdot ((3^{2}\cdot 256 +1)\cdot 256)  = 1,180,160$ parameters, while the bottleneck block has $(1^{2}\cdot 256+1)\cdot 64+(3^{2}\cdot 64+1)\cdot 64+(1^{2}\cdot 64+1)\cdot 256  = 70,016$ parameters because the bottleneck block introduces additional methods to reduce the parameter count. It involves a $1\times1$ convolution for channel dimension projection, following which the compact representation (a smaller feature map) is operated on by a $3\times3$ convolution (like the regular block) to extract spatial information, and it concludes with another $1\times1$ convolution to restore the original 256 channels. We can observe that that there is a significant reduction in the parameter count.

The bottleneck block also requires fewer operations for floating point operations compared to the regular block as it processes the input through fewer convolutions, which reduces the computational cost. Assuming an input size of $(256, H, W)$, we consider the number of floating-point operations involved. The regular block requires $(1,180,160 + 256) \times H \times W$ operations, while the bottleneck block requires $(70,016 + 256) \times H \times W$ operations.

The bottleneck block also affects both the spatial dimensions and the number of feature maps of the input, while the regular block only impacts the input spatially. By projecting the channels to a lower dimension and then back to the original dimension, the bottleneck block allows the network to capture spatial and cross-feature map information more effectively, despite having fewer parameters. This can reduce computational complexity and maintain the network's representation ability. Overall, the bottleneck block demonstrates a substantial reduction in parameters and complexity compared to the regular block.
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
