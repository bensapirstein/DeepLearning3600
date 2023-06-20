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
In general, increasing the depth of a network can lead to improved accuracy, up to a certain point. The caveat is that as the depth increases, the accuracy may stop improving or even decrease. In shallow networks, it may be difficult to capture complex patterns and relationships in the data, so adding more layers lets the network learn more representations, with each layer detecting different features. As the depth increases, the network may struggle with diminishing gradients during back propagation, which may limit the ability of the network's performance. As the depth increases, the network may also learn training examples and over-fit. 

In our experiment, we saw that we achieved the best results are with $L = 2$ and $L = 4$, and that at $L=8$ and $L=16$ we see what essentially can be defined as a flatline (which was confirmed by evaluating the JSON file generated by our experiment). At $L =8$ and $L=16$ the reduced performance could be attributed to gradient related issues, where the small gradients fail to backpropagate. For these two values of $L$, the network failed to optimize the loss, with a train accuracy of roughly $13\%$ for the 100 epochs, and the test accuracy stablizing at $7.5\%$. 

In our experiment, at $L=8$ and $L=16$, the network was untrainable as it was unable to update its weights and optimize the loss function. A potential reason for this is that during backpropagation as gradients were calculated and used to update the weights of the network they got smaller exponentially, resulting in the network failing to update the layers, impacting performance. A potential solution to this is the ResNet architecture we implemented. It introduces residual connections which let the gradient bypass layers and directly propagate between shallow and deep layers. We have already used ReLU to allow the gradient to propagate better between layers, but another solution is to attempt to pretrain the layers and improve performance that way or implement batch normalization, which was added in the ResNet as well. 
"""

part3_q3 = r"""
The number of filters per layer can have a significant impact on the performance of a neural network. Increasing the number of filters per layer generally allows the network to capture more complex and fine-grained features in the data. However, we did not see an improvement between the first and second experiment in terms of accuracy, and it was especially evident that at $L=8$ that the model was unable to learn the relationships and had reached a saturation point. We observed in this experiment that as the network depth increased with a fixed $L$, the model was increasingly difficult to train overall. The exception is in $L=8$ where we had similar results as in previous experiments, regardless of the number of filters at this $L$ we had a model that was entirely unable to train and converged at $13\%$ and $7.5\%$ accuracy for the test and train, no matter the size of the filters. This could potentially be mitigated by selecting an even smaller number of filters but we did not test this hypothesis. In addition to the gradients which we discussed in previous answers, another factor that may contribute to results is the impact of the number of filters can make it harder for the model to learn and generalize effectively as $K$ increases. Despite this, we observed that the model achieved the same train accuracy for all kernel sizes, while in the test accuracy we saw that a higher filter number $(256)$ generally trended toward being the lowest. Another observation is that as the number of filters increases, it takes longer for the model to reach its top accuracy. This could be due to the increased complexity delaying the convergence. An addtional observation of interest was that we saw that as $L$ increased in this experiment, the highest accuracy was achieved with a diminishing $K$. This could be because as the network deepens, the network is able to capture more complex features so using fewer filters per layer may help prevent overfitting, in addition, decreasing the number of filters can also reduce complexity. This may also reduce the propagation path of the gradients when properly tuned.

"""

part3_q4 = r"""
For the configurations with $K=[64, 128, 256]$ and varying $L$ we saw that at $L=4$ the model fails to train effectively as indicated by the flatlining of both training and testing accuracy. The loss also indicates that the model struggles to optimize and improve its performance. Similar to $L=4$, at $L=3$, the model fails to train effectively, with both training and testing accuracy plateauing at the same low values as $L=4$. At $L=2$ we can see that the model is actually learning (an improvement), and has a much better result in training and testing accuracy compared to $L=3$ and $L=4$. However, the accuracy values are still relatively low. The training loss decreases gradually, indicating some learning progress. For $L=1$ The model accuracy appears to disappear after a few epochs while the loss calculations continue (this was confirmed with the JSON). This abrupt disappearance could be due to issues with model stability, overfitting, or suboptimal learning rate, as discussed earlier. 
"""

part3_q5 = r"""
Experiment 1.4, which involves the ResNet architecture with skip connections, differs from the previous experiments (1.1, 1.2, and 1.3) in terms of its architectural design. The previous experiments used a standard convolutional neural network (CNN) without skip connections, while experiment 1.4 introduced the ResNet architecture to address the vanishing gradient problem and enable the training of deeper networks. By incorporating skip connections, the ResNet architecture addresses the vanishing gradient problem and enables the training of deeper networks. In particular, we can observe here that while our accuracy is still not great on test, we can see that for larger values of $L$ the model is able to learn, although it takes longer to reach a high accuracy on train, and that the loss also takes longer to diminish. While the previous experiments may have not achieved high accuracy due to limited epochs and batch sizes, we anticipate that by training the models in experiment 4 with larger batch sizes and more epochs, we may see improved generalization compared to the previous CNN architecture. Additionally, the skip connections may help prevent overfitting and contribute to better overall performance.
"""

part3_q6 = r"""
**Your answer:**

"""
# ==============
