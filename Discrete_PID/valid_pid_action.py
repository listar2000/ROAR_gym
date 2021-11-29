from collections import OrderedDict


VALID_ACTIONS = {}

MAX_SPEED = 150

TARGET_SPEED = 100

def init_actions_space():
    p_s = 0
    p_e = 1
    i_s = 0
    i_e = 1
    d_s = 0
    d_e = 1
    
    dp = 0.2
    di = 0.2
    dd = 0.2

    p = p_s
    i = i_s
    d = d_s

       
    n = 0
    while p <= p_e:
        while i <= i_e:
            while d <= d_e:
                VALID_ACTIONS[n] = [p, i, d]
                n += 1
                d += dd
            d = d_s
            i += di
        i = i_s
        p += dp

    return n





