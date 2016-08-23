import collections


def flatten(d, parent_key='', sep=''):
    """
    >>> flatten({'a': {'b': 10}}, sep='/')
    {'a/b': 10}
    """
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, collections.MutableMapping):
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)
