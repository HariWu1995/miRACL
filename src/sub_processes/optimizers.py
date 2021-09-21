import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.optimizers import SGD


def get_optimizer(optimizer_option: str, use_grad_centralized: bool=True, learning_rate: float=0.001, **kwargs):

    optimizer_option = optimizer_option.lower()
    if optimizer_option == 'adam':
        if use_grad_centralized:
            from gctf.optimizers import adam as Optimizer
        else:
            from tensorflow.keras.optimizers import Adam as Optimizer
    elif optimizer_option == 'nadam':
        if use_grad_centralized:
            from gctf.optimizers import nadam as Optimizer
        else:
            from tensorflow.keras.optimizers import Nadam as Optimizer
    elif optimizer_option == 'adagrad':
        if use_grad_centralized:
            from gctf.optimizers import adagrad as Optimizer
        else:
            from tensorflow.keras.optimizers import Adagrad as Optimizer
    elif optimizer_option == 'adadelta':
        if use_grad_centralized:
            from gctf.optimizers import adadelta as Optimizer
        else:
            from tensorflow.keras.optimizers import Adadelta as Optimizer
    elif optimizer_option == 'rmsprop':
        if use_grad_centralized:
            from gctf.optimizers import rmsprop as Optimizer
        else:
            from tensorflow.keras.optimizers import RMSprop as Optimizer
    else:
        if use_grad_centralized:
            from gctf.optimizers import sgd as Optimizer
        else:
            from tensorflow.keras.optimizers import SGD as Optimizer

    return Optimizer(learning_rate=learning_rate, **kwargs)


class SWATS(Callback):
    """
    Switching from ADAM to SGD

        Keras' callback
            that switches to the SGD optimizer from ADAM

        Reference: https://arxiv.org/pdf/1712.07628.pdf

        Usage:
            model = Sequential()
            ...
            model.compile(...)

            def switching_func(x, y, **kwargs):
                ...

            OptimizerSwitcher = SWATS(on_train_end=switching_func(x=train_x, y=train_y, ...))
            train_history = model.fit(train_x, train_y, callbacks=[OptimizerSwitcher], ...)
    """
    def __init__(self, **kwargs):
        self.phase = 'adam'
        self.w = 0
        super(SWATS, self).__init__(**kwargs)

    def on_epoch_end(self, epoch, logs={}):
        if self.phase == 'adam':
            lr, beta_2 = self.model.optimizer.lr, self.model.optimizer.beta_2
            bias_corrected_exponential_average = lr / (1.-beta_2)
            if (K.abs(bias_corrected_exponential_average-lr) < K.epsilon()):
                print(f"\n\nSwitching optimizer from ADAM to SGD ...\n")
                self.phase = 'sgd'
                # weights = self.model.get_weights()
                self.model.optimizer = SGD(lr=bias_corrected_exponential_average, momentum=K.epsilon())
                # self.model.set_weights(weights)

