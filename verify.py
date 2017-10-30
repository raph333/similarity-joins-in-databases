from math import ceil


def eqo(r, s, t=0.5):
    return t/(t+1) * (len(r) + len(s))


def verify(r, m, t=0.5):
    """
    calculates a list of pairs
    :param r: array or list of integers
    :param m: dictionary
    :param t: threshold
    :return: list of tuples
    """
    # todo evaluation and testing
    verified_pairs = []
    for key, value in m.items():
        if value >= ceil(eqo(r, key, t)):
            verified_pairs.append(tuple(r, key))

    return verified_pairs
