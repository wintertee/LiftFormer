from torch.optim.lr_scheduler import _LRScheduler

from torch.optim.lr_scheduler import ReduceLROnPlateau

# class NoamOpt():
#     '''A simple wrapper class for learning rate scheduling'''
#     def __init__(self, optimizer, lr_mul, d_model, n_warmup_steps):
#         self._optimizer = optimizer
#         self.lr_mul = lr_mul
#         self.d_model = d_model
#         self.n_warmup_steps = n_warmup_steps
#         self.n_steps = 0

#     def step_and_update_lr(self):
#         "Step with the inner optimizer"
#         self._update_learning_rate()
#         self._optimizer.step()

#     def zero_grad(self):
#         "Zero out the gradients with the inner optimizer"
#         self._optimizer.zero_grad()

#     def _get_lr_scale(self):
#         d_model = self.d_model
#         n_steps, n_warmup_steps = self.n_steps, self.n_warmup_steps
#         return (d_model**-0.5) * min(n_steps**(-0.5), n_steps * n_warmup_steps**(-1.5))

#     def _update_learning_rate(self):
#         ''' Learning rate scheduling per step '''

#         self.n_steps += 1
#         lr = self.lr_mul * self._get_lr_scale()

#         for param_group in self._optimizer.param_groups:
#             param_group['lr'] = lr

#     def load_state_dict(self, *args, **kwargs):
#         return self._optimizer.load_state_dict(*args, **kwargs)

#     def state_dict(self, *args, **kwargs):
#         return self._optimizer.state_dict(*args, **kwargs)


class NoamLR(_LRScheduler):
    """
    Implements the Noam Learning rate schedule. This corresponds to increasing the learning rate
    linearly for the first ``warmup_steps`` training steps, and decreasing it thereafter proportionally
    to the inverse square root of the step number, scaled by the inverse square root of the
    dimensionality of the model. Time will tell if this is just madness or it's actually important.
    Parameters
    ----------
    warmup_steps: ``int``, required.
        The number of steps to linearly increase the learning rate.

    MIT License

    Copyright (c) 2018 Erdene-Ochir Tuguldur

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.
    """
    def __init__(self, optimizer, warmup_steps):
        self.warmup_steps = warmup_steps
        super().__init__(optimizer)

    def get_lr(self):
        last_epoch = max(1, self.last_epoch)
        scale = self.warmup_steps**0.5 * min(last_epoch**(-0.5), last_epoch * self.warmup_steps**(-1.5))
        return [base_lr * scale for base_lr in self.base_lrs]
