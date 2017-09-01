import chainer
from chainer import cuda
from chainer.training import extensions

_default_initialW = None


def to_device(elem, device=None):
    # type: (chainer.Variable, any) -> any
    if device is None:
        return elem
    elif device < 0:
        elem.to_cpu()
    else:
        elem.to_gpu(device=device)


def to_variable(elem, device=None, volatile='auto'):
    if elem is None:
        return None
    elif isinstance(elem, chainer.Variable):
        pass
    else:
        elem = chainer.Variable(elem, volatile=volatile)

    to_device(elem, device)
    return elem


def to_variable_recursive(obj, device=None, volatile='auto'):
    if isinstance(obj, tuple) or isinstance(obj, list):
        return [to_variable_recursive(elem, device, volatile) for elem in obj]
    elif isinstance(obj, dict):
        return {key: to_variable_recursive(obj[key], device, volatile) for key in obj}
    else:
        return to_variable(obj, device, volatile)


def concat_recursive(batch):
    first = batch[0]

    if isinstance(first, tuple):
        return tuple([concat_recursive([example[i] for example in batch]) for i in range(len(first))])
    elif isinstance(first, dict):
        return {key: concat_recursive([example[key] for example in batch]) for key in first}
    else:
        return _concat_arrays(batch)


def unwrap_variable_recursive(obj):
    if isinstance(obj, tuple) or isinstance(obj, list):
        return [unwrap_variable_recursive(elem) for elem in obj]
    elif isinstance(obj, dict):
        return {key: unwrap_variable_recursive(obj[key]) for key in obj}
    elif isinstance(obj, chainer.Variable):
        return obj.data
    else:
        return obj


def change_volatile_variable_recursive(obj, volatile):
    if isinstance(obj, tuple) or isinstance(obj, list):
        return [change_volatile_variable_recursive(elem, volatile) for elem in obj]
    elif isinstance(obj, dict):
        return {key: change_volatile_variable_recursive(obj[key], volatile) for key in obj}
    elif isinstance(obj, chainer.Variable):
        obj.volatile = volatile


def converter_recursive(batch, device=None):
    batch = concat_recursive(batch)
    batch = to_variable_recursive(batch, device)
    return batch


def _concat_arrays(arrays):
    xp = cuda.get_array_module(arrays[0])
    with cuda.get_device(arrays[0]):
        return xp.concatenate([array[None] for array in arrays])


def set_default_initialW(initialW):
    global _default_initialW

    if isinstance(initialW, str):
        if initialW == 'Orthogonal':
            initialW = chainer.initializers.Orthogonal()
        else:
            raise NotImplementedError(initialW)

    _default_initialW = initialW


class Link(object):
    @staticmethod
    def create_convolution_2d(*args, **kwargs):
        initialW = kwargs.pop('initialW', _default_initialW)
        return chainer.links.Convolution2D(*args, initialW=initialW, **kwargs)

    @staticmethod
    def create_deconvolution_2d(*args, **kwargs):
        initialW = kwargs.pop('initialW', _default_initialW)
        return chainer.links.Deconvolution2D(*args, initialW=initialW, **kwargs)

    @staticmethod
    def create_dilated_convolution_2d(*args, **kwargs):
        initialW = kwargs.pop('initialW', _default_initialW)
        return chainer.links.DilatedConvolution2D(*args, initialW=initialW, **kwargs)

    @staticmethod
    def create_linear(*args, **kwargs):
        initialW = kwargs.pop('initialW', _default_initialW)
        return chainer.links.Linear(*args, initialW=initialW, **kwargs)


class NoVariableEvaluator(extensions.Evaluator):
    """
    this wrapper will become unnecessary in chainer2.
    """

    def evaluate(self):
        from chainer import reporter
        import copy

        iterator = self._iterators['main']
        target = self._targets['main']
        eval_func = self.eval_func or target

        if self.eval_hook:
            self.eval_hook(self)
        it = copy.copy(iterator)
        summary = reporter.DictSummary()

        for batch in it:
            observation = {}
            with reporter.report_scope(observation):
                in_arrays = self.converter(batch, self.device)
                if isinstance(in_arrays, tuple):
                    eval_func(*in_arrays)
                elif isinstance(in_arrays, dict):
                    eval_func(**in_arrays)
                else:
                    eval_func(in_arrays)

            summary.add(observation)

        return summary.compute_mean()
