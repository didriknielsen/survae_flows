import torch


def bisection_inverse(fn, z, init_x, init_lower, init_upper, eps=1e-10, max_iters=100):
    '''Bisection method to find the inverse of `fn`. Computed by finding the root of `z-fn(x)=0`.'''

    def body(x_, lb_, ub_, cur_z_):
        gt = (cur_z_ > z).type(z.dtype)
        lt = 1 - gt
        new_x_ = gt * (x_ + lb_) / 2. + lt * (x_ + ub_) / 2.
        new_lb = gt * lb_ + lt * x_
        new_ub = gt * x_ + lt * ub_
        return new_x_, new_lb, new_ub

    x, lb, ub = init_x, init_lower, init_upper
    cur_z = fn(x)
    diff = float('inf')
    i = 0
    while diff > eps and i < max_iters:
        x, lb, ub = body(x, lb, ub, cur_z)
        cur_z = fn(x)
        diff = (z - cur_z).abs().max()
        i += 1

    return x
