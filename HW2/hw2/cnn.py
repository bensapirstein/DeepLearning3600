import torch
import torch.nn as nn
import itertools as it
from typing import Sequence

ACTIVATIONS = {"relu": nn.ReLU, "lrelu": nn.LeakyReLU}
POOLINGS = {"avg": nn.AvgPool2d, "max": nn.MaxPool2d}


class ConvClassifier(nn.Module):
    """
    A convolutional classifier model based on PyTorch nn.Modules.

    The architecture is:
    [(CONV -> ACT)*P -> POOL]*(N/P) -> (FC -> ACT)*M -> FC
    """

    def __init__(
        self,
        in_size,
        out_classes: int,
        channels: Sequence[int],
        pool_every: int,
        hidden_dims: Sequence[int],
        conv_params: dict = {},
        activation_type: str = "relu",
        activation_params: dict = {},
        pooling_type: str = "max",
        pooling_params: dict = {},
    ):
        """
        :param in_size: Size of input images, e.g. (C,H,W).
        :param out_classes: Number of classes to output in the final layer.
        :param channels: A list of of length N containing the number of
            (output) channels in each conv layer.
        :param pool_every: P, the number of conv layers before each max-pool.
        :param hidden_dims: List of of length M containing hidden dimensions of
            each Linear layer (not including the output layer).
        :param conv_params: Parameters for convolution layers.
      
        :param activation_type: Type of activation function; supports either 'relu' or
            'lrelu' for leaky relu.
        :param activation_params: Parameters passed to activation function.
        :param pooling_type: Type of pooling to apply; supports 'max' for max-pooling or
            'avg' for average pooling.
        :param pooling_params: Parameters passed to pooling layer.
        """
        super().__init__()
        assert channels and hidden_dims

        self.in_size = in_size
        self.out_classes = out_classes
        self.channels = channels
        self.pool_every = pool_every
        self.hidden_dims = hidden_dims
        self.conv_params = conv_params
        self.activation_type = activation_type
        self.activation_params = activation_params
        self.pooling_type = pooling_type
        self.pooling_params = pooling_params

        if activation_type not in ACTIVATIONS or pooling_type not in POOLINGS:
            raise ValueError("Unsupported activation or pooling type")

        self.feature_extractor = self._make_feature_extractor()
        self.classifier = self._make_classifier()

    def _make_feature_extractor(self):
        in_channels, in_h, in_w, = tuple(self.in_size)

        layers = []
        # TODO: Create the feature extractor part of the model:
        #  [(CONV -> ACT)*P -> POOL]*(N/P)
        #  Apply activation function after each conv, using the activation type and
        #  parameters.
        #  Apply pooling to reduce dimensions after every P convolutions, using the
        #  pooling type and pooling parameters.
        #  Note: If N is not divisible by P, then N mod P additional
        #  CONV->ACTs should exist at the end, without a POOL after them.
        # ====== YOUR CODE: ======
        # Loop over groups of P output channels and create a block from them.
        N = len(self.channels)
        P = self.pool_every
        for i in range(0, N, P):
            channels = self.channels[i : i + P]
            for out_channels in channels:
                # Create a Conv2d layer with the given parameters.
                layers.append(nn.Conv2d(in_channels, out_channels, **self.conv_params))

                # Create an activation layer with the given type and parameters.
                layers.append(ACTIVATIONS[self.activation_type](**self.activation_params))

                # Update the number of input channels for the next conv.
                in_channels = out_channels

            if i + P <= N:
                # Create a pooling layer with the given type and parameters.
                layers.append(POOLINGS[self.pooling_type](**self.pooling_params))

        # ========================
        return nn.Sequential(*layers)

    def _n_features(self) -> int:
        """
        Calculates the number of extracted features going into the the classifier part.
        :return: Number of features.
        """
        # Make sure to not mess up the random state.
        rng_state = torch.get_rng_state()
        try:
            # ====== YOUR CODE: ======
            # Create a random tensor with the same shape as the input tensor.
            x = torch.randn(1, *self.in_size)

            # Pass the tensor through the feature extractor.
            x = self.feature_extractor(x)
            x = torch.flatten(x, 1)

            # Return the number of features.
            return x.shape[1]

            # ========================
        finally:
            torch.set_rng_state(rng_state)

    def _make_classifier(self):
        layers = []

        # Discover the number of features after the CNN part.
        n_features = self._n_features()

        # TODO: Create the classifier part of the model:
        #  (FC -> ACT)*M -> Linear
        #  The last Linear layer should have an output dim of out_classes.
        # ====== YOUR CODE: ======
        in_channels = n_features

        # Create Fully Connected layers.
        for hidden_dim in self.hidden_dims:
            # Create a Linear layer with the given parameters.
            linear = nn.Linear(in_channels, hidden_dim)

            # Create an activation layer with the given type and parameters.
            activation = ACTIVATIONS[self.activation_type](**self.activation_params)

            layers += [linear, activation]

            # Update the number of input channels for the next conv.
            in_channels = hidden_dim

        # Create the last Linear layer.
        layers.append(nn.Linear(in_channels, self.out_classes))
        # ========================

        return nn.Sequential(*layers)

    def forward(self, x):
        # TODO: Implement the forward pass.
        #  Extract features from the input, run the classifier on them and
        #  return class scores.
        # ====== YOUR CODE: ======
        # Pass the input through the feature extractor.
        # Flatten the output of the feature extractor.
        x = torch.flatten(self.feature_extractor(x), 1)
        # ========================
        # Pass the flattened output through the classifier.
        return self.classifier(x)


class ResidualBlock(nn.Module):
    """
    A general purpose residual block.
    """

    def __init__(
        self,
        in_channels: int,
        channels: Sequence[int],
        kernel_sizes: Sequence[int],
        batchnorm: bool = False,
        dropout: float = 0.0,
        activation_type: str = "relu",
        activation_params: dict = {},
        **kwargs,
    ):
        """
        :param in_channels: Number of input channels to the first convolution.
        :param channels: List of number of output channels for each
            convolution in the block. The length determines the number of
            convolutions.
        :param kernel_sizes: List of kernel sizes (spatial). Length should
            be the same as channels. Values should be odd numbers.
        :param batchnorm: True/False whether to apply BatchNorm between
            convolutions.
        :param dropout: Amount (p) of Dropout to apply between convolutions.
            Zero means don't apply dropout.
        :param activation_type: Type of activation function; supports either 'relu' or
            'lrelu' for leaky relu.
        :param activation_params: Parameters passed to activation function.
        """
        super().__init__()
        assert channels and kernel_sizes
        assert len(channels) == len(kernel_sizes)
        assert all(map(lambda x: x % 2 == 1, kernel_sizes))

        if activation_type not in ACTIVATIONS:
            raise ValueError("Unsupported activation type")

        self.main_path, self.shortcut_path = None, None

        # TODO: Implement a generic residual block.
        #  Use the given arguments to create two nn.Sequentials:
        #  - main_path, which should contain the convolution, dropout,
        #    batchnorm, relu sequences (in this order).
        #    Should end with a final conv as in the diagram.
        #  - shortcut_path which should represent the skip-connection and
        #    may contain a 1x1 conv.
        #  Notes:
        #  - Use convolutions which preserve the spatial extent of the input.
        #  - Use bias in the main_path conv layers, and no bias in the skips.
        #  - For simplicity of implementation, assume kernel sizes are odd.
        #  - Don't create layers which you don't use! This will prevent
        #    correct comparison in the test.
        # ====== YOUR CODE: ======        
        layers = []
        start_channels = in_channels
        for i, (out_channels, kernel_size) in enumerate(zip(channels, kernel_sizes)):
            # Create a conv layer with the given type and parameters.
            conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size//2, bias=True)
            layers.append(conv)

            if i < len(channels) - 1:

                # Create a dropout layer with the given parameters.
                if dropout > 0:
                    layers.append(nn.Dropout2d(p=dropout))

                # Create a batchnorm layer with the given parameters.
                if batchnorm:
                    layers.append(nn.BatchNorm2d(out_channels))

                # Create an activation layer with the given type and parameters.
                layers.append(ACTIVATIONS[activation_type](**activation_params))

            # Update the number of input channels for the next conv.
            in_channels = out_channels

        # Create the main path.
        self.main_path = nn.Sequential(*layers)

        # Create the shortcut path.
        self.shortcut_path = nn.Sequential(nn.Conv2d(start_channels, channels[-1], 1, bias=False)) if start_channels != channels[-1] else nn.Sequential()
        # ========================


    def forward(self, x):
        out = self.main_path(x)
        out += self.shortcut_path(x)
        return torch.relu(out)


class ResidualBottleneckBlock(ResidualBlock):
    """
    A residual bottleneck block.
    """

    def __init__(
        self,
        in_out_channels: int,
        inner_channels: Sequence[int],
        inner_kernel_sizes: Sequence[int],
        **kwargs,
    ):
        """
        :param in_out_channels: Number of input and output channels of the block.
            The first conv in this block will project from this number, and the
            last conv will project back to this number of channel.
        :param inner_channels: List of number of output channels for each internal
            convolution in the block (i.e. not the outer projections)
            The length determines the number of convolutions.
        :param inner_kernel_sizes: List of kernel sizes (spatial) for the internal
            convolutions in the block. Length should be the same as inner_channels.
            Values should be odd numbers.
        :param kwargs: Any additional arguments supported by ResidualBlock.
        """
        super().__init__(
            in_out_channels,
            [inner_channels[0], *inner_channels, in_out_channels],
            [1, *inner_kernel_sizes, 1],
            **kwargs)


class ResNetClassifier(ConvClassifier):
    def __init__(
        self,
        in_size,
        out_classes,
        channels,
        pool_every,
        hidden_dims,
        batchnorm=False,
        dropout=0.0,
        **kwargs,
    ):
        """
        See arguments of ConvClassifier & ResidualBlock.
        """
        self.batchnorm = batchnorm
        self.dropout = dropout
        super().__init__(
            in_size, out_classes, channels, pool_every, hidden_dims, **kwargs
        )

    def _make_feature_extractor(self):
        in_channels, in_h, in_w, = tuple(self.in_size)

        layers = []
        # TODO: Create the feature extractor part of the model:
        #  [-> (CONV -> ACT)*P -> POOL]*(N/P)
        #   \------- SKIP ------/
        #  For the ResidualBlocks, use only dimension-preserving 3x3 convolutions.
        #  Apply Pooling to reduce dimensions after every P convolutions.
        #  Notes:
        #  - If N is not divisible by P, then N mod P additional
        #    CONV->ACT (with a skip over them) should exist at the end,
        #    without a POOL after them.
        #  - Use your own ResidualBlock implementation.
        # ====== YOUR CODE: ======
        # Loop over groups of P output channels and create a block from them.
        N = len(self.channels)
        P = self.pool_every
        for i in range(0, N, P):
            channels = self.channels[i : i + P]
            kernel_sizes = [3] * len(channels)
            layers.append(ResidualBlock(
                in_channels=in_channels, channels=channels, 
                kernel_sizes=kernel_sizes, batchnorm=self.batchnorm,
                dropout=self.dropout, activation_type=self.activation_type,
                activation_params=self.activation_params))

            if i + P < N:
                layers.append(POOLINGS[self.pooling_type](**self.pooling_params))

            in_channels = channels[-1]

        # ========================
        return nn.Sequential(*layers)

class YourCodeNet(ConvClassifier):
    def __init__(self, *args, dropout_rate=0.5, pool_kernel=2, stride=2, batch_norm=True, kernel_sizes=[1, 3, 5], kernel_size=3, **kwargs):
        """
        See ConvClassifier.__init__
        """
        self.kernel_size = kernel_size
        self.kernel_sizes = kernel_sizes
        self.pool_kernel = pool_kernel
        self.stride = stride
        self.dropout_rate = dropout_rate
        self.batch_norm = batch_norm
        super().__init__(*args, **kwargs)

    def _make_feature_extractor(self):
        in_channels, _, _ = tuple(self.in_size)
        layers = []
        cur_in_channels = in_channels
        i = 0
        while i < len(self.channels):
            layers.append(
                self._make_skip_connection(self._make_inception_block(in_channels=cur_in_channels, out_channels=self.channels[i],), in_channels=cur_in_channels, out_channels=self.channels[i],))
            if i % self.pool_every == 0: layers.append(nn.MaxPool2d(kernel_size=self.pool_kernel, stride=self.stride))
            i += 1
            cur_in_channels = self.channels[i - 1]

        return nn.Sequential(*layers)

    def _make_skip_connection(self, main_path, in_channels, out_channels):
        return SkipConnection(main_path, in_channels, out_channels)

    def _make_inception_block(self, in_channels: int, out_channels: int) -> nn.Module:
        return InceptionBlock(
            in_channels,
            out_channels,
            kernel_sizes=self.kernel_sizes,
            batchnorm=True,
            activation_type="lrelu",
            activation_params={'negative_slope': 0.01},
        )

    def forward(self, x):
        features = self.feature_extractor(x)
        flattened = torch.flatten(features, 1)
        if self.dropout_rate > 0:
            flattened = nn.Dropout(self.dropout_rate)(flattened)
        return self.classifier(flattened)


class InceptionBlock(nn.Module):
    """
    Generate a general purpose Inception block
    """

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_sizes: Sequence[int] = [1, 3, 5],
            batchnorm: bool = False,
            dropout: float = 0.0,
            activation_type: str = "relu",
            activation_params: dict = {},
            **kwargs,
    ):
        super().__init__()
        out_channels = int(out_channels / 4)
        self.module = [[nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1)]] + [[nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=True)] for kernel_size in kernel_sizes]
        self.module = [conv + ([nn.Dropout2d(p=dropout)] if dropout > 0 else []) + ([nn.BatchNorm2d(num_features=out_channels)] if batchnorm else []) for conv in self.module]        
        self.module = nn.ModuleList([nn.Sequential(*filt) for filt in self.module])
        self.activation_layer = ACTIVATIONS[activation_type](**activation_params)

    def forward(self, x):
        return torch.cat([self.activation_layer(element) for element in [conv(x) for conv in self.module]], 1)


class SkipConnection(nn.Module):
    """
    A skip connection module.

    Args:
        main_path (nn.Module): The main path of the skip connection.
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.

    Attributes:
        main_path (nn.Module): The main path of the skip connection.
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        shortcut (nn.Module): The shortcut path of the skip connection.
    """
    def __init__(self, main_path, in_channels, out_channels):
        super().__init__()
        self.main_path = main_path
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.shortcut = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(out_channels)) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        return self.main_path(x) + self.shortcut(x)