import numpy as np

import tensorly as tl
tl.set_backend('pytorch')


def count_cp4_parameters(tensor_shape,
                         rank = 8):
    cout, cin, kh, kw = tensor_shape
    cp4_count = rank * (cin + kh + kw + cout)
    return cp4_count

def count_cp3_parameters(tensor_shape,
                         rank = 8):
    cout, cin, kh, kw = tensor_shape
    cp3_count = rank * (cin + kh*kw + cout)
    return cp3_count

def count_tucker2_parameters(tensor_shape,
                             ranks = [8,8]):
    cout, cin, kh, kw = tensor_shape
    
    if type(ranks)!=list or type(ranks)!=tuple:
        ranks = [ranks, ranks]
    tucker2_count = ranks[-2]*cin + np.prod(ranks[-2:])*kh*kw + ranks[-1]*cout
    return np.array(tucker2_count)
    

def count_parameters(tensor_shape,
                     rank = None,
                     key = 'cp3'):
    cout, cin, kh, kw = tensor_shape

    if key == 'cp3':
        params_count = count_cp3_parameters(tensor_shape, rank=rank)
    elif key == 'tucker2':
        params_count = count_tucker2_parameters(tensor_shape, ranks=rank)    
    
    return params_count


def estimate_rank_for_compression_rate(tensor_shape,
                                  rate = 2,
                                  key = 'tucker2'):

    initial_count = np.prod(tensor_shape)
    cout, cin, kh, kw = tensor_shape

    if key == 'cp3':
        max_rank = initial_count//(rate*(cin + kh*kw + cout))

    elif key == 'tucker2':
        if cout > cin:
             beta=(1/rate)*(cout/cin-1)+1

        elif cin>cout:
            beta = (1 / rate) * (cin / cout - 1) + 1
            beta=1/beta

        else:
            beta = 1.

        a = 1
        b = (cin + beta * cout) / (beta * kh * kw)
        c = -cin*cout/rate/beta




        discr = b**2 - 4*a*c
        max_rank = int((-b + np.sqrt(discr))/2/a)
        # [R4, R3]
        max_rank = (int(np.ceil(beta*max_rank)), max_rank)
    
    
    return max_rank


