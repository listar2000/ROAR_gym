from collections import OrderedDict


VALID_ACTIONS = {}

MAX_SPEED = 180

TARGET_SPEED = 110

def init_actions_space():
    p_s = 0
    p_e = 0.6
    i_s = 0
    i_e = 0.6
    d_s = 0
    d_e = 0.6
    
    dp = 0.1
    di = 0.1
    dd = 0.1

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





