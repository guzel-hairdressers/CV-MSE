from interface import *

# ================================= 1.4.1 SGD ================================

class SGD(Optimizer):
    def __init__(self, lr=.01, weight_decay=0.0):
        self.lr = lr
        self.weight_decay = weight_decay

    def get_parameter_updater(self, parameter_shape):
        """
        :param parameter_shape: tuple, the shape of the associated parameter
        :return: the updater function for that parameter
        """

        def updater(parameter, parameter_grad):
            """
            :param parameter: np.array, current parameter values
            :param parameter_grad: np.array, current gradient, dLoss/dParam
            :return: np.array, new parameter values
            """
            return parameter - self.lr*(parameter_grad + self.weight_decay*parameter)

        return updater


# ============================= 1.4.2 SGDMomentum ============================

class SGDMomentum(Optimizer):
    def __init__(self, lr=.01, momentum=0.0, weight_decay=0.0):
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay

    def get_parameter_updater(self, parameter_shape):
        """
        :param parameter_shape: tuple, the shape of the associated parameter
        :return: the updater function for that parameter
        """
        velocity = np.zeros(parameter_shape)

        def updater(parameter, parameter_grad):
            """
            :param parameter: np.array, current parameter values
            :param parameter_grad: np.array, current gradient, dLoss/dParam
            :return: np.array, new parameter values
            """
            nonlocal velocity
            velocity = self.momentum*velocity + self.lr*(parameter_grad + self.weight_decay*parameter)
            return parameter - velocity

        return updater


# ================================ 2.1.1 ReLU ================================

class ReLU(Layer):
    def forward_impl(self, inputs):
        """
        :param inputs: np.array((n, ...)), input values
        :return: np.array((n, ...)), output values
        n - batch size
        ... - arbitrary shape (the same for input and output)
        """
        self.gradient = np.where(inputs<0, 0, 1)
        return np.where(inputs<0, 0, inputs)

    def backward_impl(self, grad_outputs):
        """
        :param grad_outputs: np.array((n, ...)), dLoss/dOutputs
        :return: np.array((n, ...)), dLoss/dInputs
        n - batch size
        ... - arbitrary shape (the same for input and output)
        """
        return self.gradient * grad_outputs


# =============================== 2.1.2 Softmax ==============================

class Softmax(Layer):
    def forward_impl(self, inputs):
        """
        :param inputs: np.array((n, d)), input values
        :return: np.array((n, d)), output values
        n - batch size
        d - number of units
        """
        shifted_inputs = inputs - np.max(inputs, -1, keepdims=True)
        exp_inputs = np.exp(shifted_inputs)
        return exp_inputs / (np.sum(exp_inputs, axis=-1, keepdims=True))

    def backward_impl(self, grad_outputs):
        """
        :param grad_outputs: np.array((n, d)), dLoss/dOutputs
        :return: np.array((n, d)), dLoss/dInputs
        n - batch size
        d - number of units
        """
        return grad_outputs


# ================================ 2.1.3 Dense ===============================

class Dense(Layer):
    def __init__(self, units, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.output_units = units
        self.weights, self.weights_grad = None, None
        self.biases, self.biases_grad = None, None

    def build(self, *args, **kwargs):
        super().build(*args, **kwargs)
        input_units, = self.input_shape
        output_units = self.output_units

        # Register weights and biases as trainable parameters
        # Note, that the parameters and gradients *must* be stored in
        # self.<parameter_name> and self.<parameter_name>_grad, where
        # <parameter_name> is the name specified in self.add_parameter

        self.weights, self.weights_grad = self.add_parameter(
            name='weights',
            shape=(input_units, output_units),
            initializer=he_initializer(input_units)
        )

        self.biases, self.biases_grad = self.add_parameter(
            name='biases',
            shape=(output_units,),
            initializer=np.zeros
        )

        self.output_shape = (output_units,)

    def forward_impl(self, inputs):
        """
        :param inputs: np.array((n, d)), input values
        :return: np.array((n, c)), output values
        n - batch size
        d - number of input units
        c - number of output units
        """
        self.inputs = inputs
        return inputs@self.weights + self.biases

    def backward_impl(self, grad_outputs):
        """
        :param grad_outputs: np.array((n, c)), dLoss/dOutputs
        :return: np.array((n, d)), dLoss/dInputs
        n - batch size
        d - number of input units
        c - number of output units
        """
        self.weights_grad = self.inputs.T @ grad_outputs
        self.biases_grad = np.sum(grad_outputs, axis=0)
        return grad_outputs @ self.weights.T


# ============================ 2.2.1 Crossentropy ============================

class CategoricalCrossentropy(Loss):
    def value_impl(self, y_gt, y_pred):
        """
        :param y_gt: np.array((n, d)), ground truth (correct) labels
        :param y_pred: np.array((n, d)), estimated target values
        :return: np.array((1,)), mean Loss scalar for batch
        n - batch size
        d - number of units
        """
        return np.asarray(((-np.sum(y_gt * np.log(np.clip(y_pred,1e-9,1-1e-9)), axis=-1)).mean(),))

    def gradient_impl(self, y_gt, y_pred):
        """
        :param y_gt: np.array((n, d)), ground truth (correct) labels
        :param y_pred: np.array((n, d)), estimated target values
        :return: np.array((n, d)), dLoss/dY_pred
        n - batch size
        d - number of units
        """
        return (y_pred - y_gt) / y_gt.shape[0]


# ======================== 2.3 Train and Test on MNIST =======================

def train_mnist_model(x_train, y_train, x_valid, y_valid):
    """
    Create and train a model to classify MNIST digits.
    
    :param x_train: np.array, training set inputs
    :param y_train: np.array, training set labels 
    :param x_valid: np.array, validation set inputs
    :param y_valid: np.array, validation set labels
    :return: trained Model instance
    """
    x_train_flat = x_train.reshape(x_train.shape[0], -1)
    x_valid_flat = x_valid.reshape(x_valid.shape[0], -1)
    
    # Normalize pixel values to [0, 1]
    x_train_flat = x_train_flat.astype(np.float32) / 255.0
    x_valid_flat = x_valid_flat.astype(np.float32) / 255.0
    
    # Convert labels to one-hot encoding
    num_classes = 10
    y_train_onehot = np.eye(num_classes)[y_train.astype(int)]
    y_valid_onehot = np.eye(num_classes)[y_valid.astype(int)]
    
    # Create model architecture
    model = Model(optimizer=SGDMomentum(), loss=CategoricalCrossentropy())
    
    # Compile model
    model.add(Dense(256, input_shape=(784,)))
    model.add(ReLU())
    model.add(Dense(256))
    model.add(ReLU())
    model.add(Dense(10))
    model.add(Softmax())
    
    # Train model
    model.fit(
        x_train_flat, y_train_onehot,
        batch_size=32,
        epochs=10,
        x_valid=x_valid_flat,
        y_valid=y_valid_onehot,
        verbose=True
    )
    
    return model


# ============================== 3.3.2 convolve ==============================

def convolve(inputs, kernels, padding=0.):
    """
    :param inputs: np.array((n, d, ih, iw)), input values
    :param kernels: np.array((c, d, kh, kw)), convolution kernels
    :param padding: int >= 0, the size of padding, 0 means 'valid'
    :return: np.array((n, c, oh, ow)), output values
    n - batch size
    d - number of input channels
    c - number of output channels
    (ih, iw) - input image shape
    (oh, ow) - output image shape
    """
    # !!! Don't change this function, it's here for your reference only !!!
    assert isinstance(padding, int) and padding >= 0
    assert inputs.ndim == 4 and kernels.ndim == 4
    assert inputs.shape[1] == kernels.shape[1]
    padding = int(padding)

    if os.environ.get('USE_FAST_CONVOLVE', False):
        return convolve_pytorch(inputs, kernels, padding)
    else:
        return convolve_numpy(inputs, kernels, padding)


def convolve_numpy(inputs, kernels, padding=0.):
    """
    :param inputs: np.array((n, d, ih, iw)), input values
    :param kernels: np.array((c, d, kh, kw)), convolution kernels
    :param padding: int >= 0, the size of padding, 0 means 'valid'
    :return: np.array((n, c, oh, ow)), output values
    n - batch size
    d - number of input channels
    c - number of output channels
    (ih, iw) - input image shape
    (oh, ow) - output image shape
    """
    n, _, ih, iw = inputs.shape
    c, _, kh, kw = kernels.shape
    oh = ih - kh + 2 * padding + 1
    ow = iw - kw + 2 * padding + 1
    input_padded = np.pad(inputs, ((0,0), (0,0), (padding, padding), (padding, padding)))
    convoluted = np.zeros((n, c, oh, ow))
    for i in range(oh):
        for j in range(ow):
            convoluted[:, :, i, j] = np.sum(
                input_padded[:, None, :, i:i+kh, j:j+kw] * kernels[None, :, :, :, :],
                axis=(2, 3, 4)
            )
    return convoluted

# =============================== 4.1.1 Conv2D ===============================

class Conv2D(Layer):
    def __init__(self, output_channels, kernel_size=3, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert kernel_size % 2, "Kernel size should be odd"
        self.output_channels = output_channels
        self.kernel_size = kernel_size
        self.kernels, self.kernels_grad = None, None
        self.biases, self.biases_grad = None, None

    def build(self, *args, **kwargs):
        super().build(*args, **kwargs)
        input_channels, input_h, input_w = self.input_shape
        output_channels = self.output_channels
        kernel_size = self.kernel_size

        self.kernels, self.kernels_grad = self.add_parameter(
            name='kernels',
            shape=(output_channels, input_channels, kernel_size, kernel_size),
            initializer=he_initializer(input_h * input_w * input_channels)
        )

        self.biases, self.biases_grad = self.add_parameter(
            name='biases',
            shape=(output_channels,),
            initializer=np.zeros
        )

        self.output_shape = (output_channels,) + self.input_shape[1:]

    def forward_impl(self, inputs):
        """
        :param inputs: np.array((n, d, h, w)), input values
        :return: np.array((n, c, h, w)), output values
        n - batch size
        d - number of input channels
        c - number of output channels
        (h, w) - image shape
        """
        self.inputs = inputs
        
        return convolve(inputs, self.kernels, padding=self.kernel_size//2) + self.biases[None,:,None,None]

    def backward_impl(self, grad_outputs):
        """
        :param grad_outputs: np.array((n, c, h, w)), dLoss/dOutputs
        :return: np.array((n, d, h, w)), dLoss/dInputs
        n - batch size
        d - number of input channels
        c - number of output channels
        (h, w) - image shape
        """
        
        self.kernels_grad = convolve(self.inputs[:,:,::-1,::-1].swapaxes(0,1), grad_outputs.swapaxes(0,1), padding=self.kernel_size//2).swapaxes(0,1)
        self.biases_grad = np.sum(grad_outputs, axis=(0, 2, 3))
        return convolve(grad_outputs, self.kernels[:,:,::-1,::-1].swapaxes(0,1), padding=self.kernel_size//2)


# ============================== 4.1.2 Pooling2D =============================

class Pooling2D(Layer):
    def __init__(self, pool_size=2, pool_mode='max', *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert pool_mode in {'avg', 'max'}
        self.pool_size = pool_size
        self.pool_mode = pool_mode
        # self.forward_idxs = None

    def build(self, *args, **kwargs):
        super().build(*args, **kwargs)
        channels, input_h, input_w = self.input_shape
        output_h, rem_h = divmod(input_h, self.pool_size)
        output_w, rem_w = divmod(input_w, self.pool_size)

        assert not rem_h, "Input height should be divisible by the pool size"
        assert not rem_w, "Input width should be divisible by the pool size"

        self.output_shape = (channels, output_h, output_w)

    def forward_impl(self, inputs):
        """
        :param inputs: np.array((n, d, ih, iw)), input values
        :return: np.array((n, d, oh, ow)), output values
        n - batch size
        d - number of channels
        (ih, iw) - input image shape
        (oh, ow) - output image shape
        """
        n, d, _, _ = inputs.shape
        _, oh, ow = self.output_shape
        input_windows = inputs.reshape(n, d, oh, self.pool_size, ow, self.pool_size).swapaxes(3,4)
        if self.pool_mode == 'avg':
            self.pool_mask = np.full_like(input_windows, 1/(self.pool_size**2))
            return np.mean(input_windows, axis=(-2, -1))
        if self.pool_mode == 'max':
            pooled = np.max(input_windows, axis=(-1, -2))
            self.pool_mask = input_windows == pooled[:, :, :, :, None, None]
            return pooled

    def backward_impl(self, grad_outputs):
        """
        :param grad_outputs: np.array((n, d, oh, ow)), dLoss/dOutputs
        :return: np.array((n, d, ih, iw)), dLoss/dInputs
        n - batch size
        d - number of channels
        (ih, iw) - input image shape
        (oh, ow) - output image shape
        """
        n, d, oh, ow = grad_outputs.shape
        return (self.pool_mask*grad_outputs[:,:,:,:,None,None]).swapaxes(3,4).reshape(n, d, oh*self.pool_size, ow*self.pool_size)
        


# ============================== 4.1.3 BatchNorm =============================

class BatchNorm(Layer):
    def __init__(self, momentum=0.1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.momentum = momentum
        self.running_mean = None
        self.running_var = None
        self.beta, self.beta_grad = None, None
        self.gamma, self.gamma_grad = None, None
        self.forward_inverse_std = None

    def build(self, *args, **kwargs):
        super().build(*args, **kwargs)
        input_channels, _, _ = self.input_shape

        self.running_mean = np.zeros((input_channels,))
        self.running_var = np.ones((input_channels,))

        self.beta, self.beta_grad = self.add_parameter(
            name='beta',
            shape=(input_channels,),
            initializer=np.zeros
        )

        self.gamma, self.gamma_grad = self.add_parameter(
            name='gamma',
            shape=(input_channels,),
            initializer=np.ones
        )

    def forward_impl(self, inputs):
        """
        :param inputs: np.array((n, d, h, w)), input values
        :return: np.array((n, d, h, w)), output values
        n - batch size
        d - number of channels
        (h, w) - image shape
        """
        if not self.is_training:
            self.inputs_normalized = (inputs-self.running_mean[None,:,None,None]) / np.sqrt(self.running_var[None,:,None,None]+1e-5)
            return self.inputs_normalized * self.gamma[None,:,None,None] + self.beta[None,:,None,None]

        mean = np.mean(inputs, axis=(0,2,3))
        var = np.var(inputs, axis=(0,2,3))
        self.running_mean = self.momentum*self.running_mean + (1-self.momentum) * mean
        self.running_var = self.momentum*self.running_var + (1-self.momentum) * np.var(inputs, axis=(0,2,3), ddof=1)
        self.forward_inverse_std = 1 / np.sqrt(var[None,:,None,None]+1e-5)

        self.inputs_normalized = (inputs-mean[None,:,None,None]) * self.forward_inverse_std
        return self.inputs_normalized * self.gamma[None,:,None,None] + self.beta[None,:,None,None]

    def backward_impl(self, grad_outputs):
        """
        :param grad_outputs: np.array((n, d, h, w)), dLoss/dOutputs
        :return: np.array((n, d, h, w)), dLoss/dInputs
        n - batch size
        d - number of channels
        (h, w) - image shape
        """
        n = grad_outputs.shape[0] * grad_outputs.shape[2] * grad_outputs.shape[3]
        self.gamma_grad = np.sum(grad_outputs * self.inputs_normalized, axis=(0, 2, 3))
        self.beta_grad = np.sum(grad_outputs, axis=(0, 2, 3))
        return 1./n * self.gamma[None,:,None,None] * self.forward_inverse_std * \
            (n*grad_outputs - np.sum(grad_outputs, axis=(0,2,3))[None,:,None,None] - self.inputs_normalized*np.sum(grad_outputs * self.inputs_normalized, axis=(0,2,3))[None,:,None,None])


# =============================== 4.1.4 Flatten ==============================

class Flatten(Layer):
    def build(self, *args, **kwargs):
        super().build(*args, **kwargs)
        self.output_shape = (np.prod(self.input_shape),)

    def forward_impl(self, inputs):
        """
        :param inputs: np.array((n, d, h, w)), input values
        :return: np.array((n, (d * h * w))), output values
        n - batch size
        d - number of input channels
        (h, w) - image shape
        """
        
        return inputs.reshape(inputs.shape[0], -1)

    def backward_impl(self, grad_outputs):
        """
        :param grad_outputs: np.array((n, (d * h * w))), dLoss/dOutputs
        :return: np.array((n, d, h, w)), dLoss/dInputs
        n - batch size
        d - number of units
        (h, w) - input image shape
        """
        return grad_outputs.reshape(np.hstack((len(grad_outputs), self.input_shape)))


# =============================== 4.1.5 Dropout ==============================

class Dropout(Layer):
    def __init__(self, p=.1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.p = p
        self.forward_mask = None

    def forward_impl(self, inputs):
        """
        :param inputs: np.array((n, ...)), input values
        :return: np.array((n, ...)), output values
        n - batch size
        ... - arbitrary shape (the same for input and output)
        """
        if not self.is_training:
            return (1 - self.p) * inputs
        
        self.forward_mask = np.random.binomial(1, 1-self.p, inputs.shape)
        return self.forward_mask * inputs
            

    def backward_impl(self, grad_outputs):
        """
        :param grad_outputs: np.array((n, ...)), dLoss/dOutputs
        :return: np.array((n, ...)), dLoss/dInputs
        n - batch size
        ... - arbitrary shape (the same for input and output)
        """
        return self.forward_mask * grad_outputs
    
class SpatialDropout(Layer):
    def __init__(self, p=.1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.p = p
        self.forward_mask = None

    def forward_impl(self, inputs):
        """
        :param inputs: np.array((n, ...)), input values
        :return: np.array((n, ...)), output values
        n - batch size
        ... - arbitrary shape (the same for input and output)
        """
        if not self.is_training:
            return (1 - self.p) * inputs
        
        self.forward_mask = np.random.binomial(1, 1-self.p, inputs.shape[1])[None, :, None, None]
        return self.forward_mask * inputs
            

    def backward_impl(self, grad_outputs):
        """
        :param grad_outputs: np.array((n, ...)), dLoss/dOutputs
        :return: np.array((n, ...)), dLoss/dInputs
        n - batch size
        ... - arbitrary shape (the same for input and output)
        """
        return self.forward_mask * grad_outputs


# ====================== 4.2 Train and Test on CIFAR-10 ======================
def train_cifar10_model(x_train, y_train, x_valid, y_valid):
    """
    Create and train a model to classify CIFAR-10 images.
    
    :param x_train: np.array, training set inputs
    :param y_train: np.array, training set labels 
    :param x_valid: np.array, validation set inputs
    :param y_valid: np.array, validation set labels
    :return: trained Model instance
    """
    os.environ['USE_FAST_CONVOLVE'] = '1'
    
    lr = 1e-2
    weight_decay = 1e-3
    momentum = .9
    batch_size = 128
    epochs = 10
    print('Learning rate: ', lr, ', momentum: ', momentum, ', batch size: ', batch_size, ', epochs: ', epochs, ', weight_decay:', weight_decay)
    model = Model(optimizer=SGDMomentum(lr, momentum, weight_decay), loss=CategoricalCrossentropy())

    model_layers = (
        Conv2D(16, input_shape=(3, 32, 32)), ReLU(), BatchNorm(),
        Pooling2D(), SpatialDropout(.1),

        Conv2D(64), ReLU(), BatchNorm(),
        Pooling2D(), SpatialDropout(.2),

        Flatten(),

        Dense(256), ReLU(), Dropout(.3),
        Dense(10), Softmax()
    )

    for layer in model_layers:
        model.add(layer)
    
    # Train model
    model.fit(
        x_train, y_train,
        batch_size=batch_size,
        epochs=epochs,
        x_valid=x_valid,
        y_valid=y_valid,
        verbose=True,
        gradient_clip_value=None
    )
    
    return model