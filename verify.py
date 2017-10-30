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
    verified_pairs = []
    for key, value in m.items():
        print(key,  value)
        if value >= ceil(eqo(r, key, t)):
            verified_pairs.append(tuple(r, key))

    return verified_pairs


# Test
R = [(1, 3, 5), (1, 2, 3, 4), (1, 2, 4, 9, 11), (1, 3, 5, 8, 9, 10, 11, 12, 13, 14)]

M = {R[0]: 1, R[1]: 2}

verify(R[2], M)
ceil(eqo(R[2], R[0]))
# todo denkfehler noch ausbessern und intersect nur bis inkl. Prefix probe von aktuellem r rechnen.



