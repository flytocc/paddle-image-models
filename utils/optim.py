import paddle.optimizer as optim


class AdamW(optim.AdamW):
    r"""
    The AdamW optimizer
    """


class Momentum(optim.Momentum):

    def __init__(self, *args, parameters=None, weight_decay=None, apply_decay_param_fun=None, **kwargs):
        # model_list is None in static graph
        if apply_decay_param_fun is not None:
            params_with_decay = []
            params_without_decay = []
            for param in parameters:
                if apply_decay_param_fun(param.name):
                    params_with_decay.append(param)
                else:
                    params_without_decay.append(param)
            parameters = [{
                "params": params_with_decay,
                "weight_decay": weight_decay
            }, {
                "params": params_without_decay,
                "weight_decay": 0.0
            }]

        super().__init__(*args, parameters=parameters, weight_decay=weight_decay, **kwargs)


class SGD(optim.SGD):

    def __init__(self, *args, apply_decay_param_fun=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._apply_decay_param_fun = apply_decay_param_fun

    def _create_regularization_of_grad(self, param, grad, regularization=None):
        # Whether we should do weight decay for the parameter.
        if self._apply_decay_param_fun is not None \
                and not self._apply_decay_param_fun(param.name):
            regularization = None

        return super()._create_regularization_of_grad(param, grad, regularization)
