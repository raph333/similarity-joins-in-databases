from math import ceil


def eqo(r, s, t=0.5):
    return t/(t+1) * (len(r) + len(s))


R = [(1, 3, 5), (1, 2, 3, 4), (1, 2, 4, 9, 11), (1, 3, 5, 8, 9, 10, 11, 12, 13, 14)]

M = {R[0]: 1, R[1]: 2}


def verify1(r, s, threshold, overlap, p_r, p_s):
    """
    :param r: tuple1
    :param s: tuple2
    :param threshold: float
    :param overlap: integer
    :param p_r: integer index for tuple r
    :param p_s: integer index for tupls s
    :return: tuple of tuples if the overlap to specified indices exceeds a given threshold
    """
    #todo muss noch mit den richtigen Argumenten in den Alg eingebaut werden.
    t = ceil(eqo(r, s, threshold))
    max_r, max_s = len(r) - p_r + overlap, len(s) - p_s + overlap
    while max_r >= t & max_s >= t & overlap < t:
        if r[p_r] == s[p_r]:
            p_r, p_s, overlap = p_r + 1, p_s + 1, overlap + 1
        elif r[p_r] < s[p_r]:
            p_r, max_r = p_r + 1, max_r - 1
        else:
            p_s, max_s = p_s + 1, max_s - 1

    return r, s


verify1(R[2], R[1], 0.5, overlap=3, p_r=2, p_s=3)



