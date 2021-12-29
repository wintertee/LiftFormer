from torch.optim import Adam


class NoamAdam(Adam):
    '''A simple wrapper class for learning rate scheduling'''
    def __init__(self, use_noam, d_model=None, n_warmup_steps=None, writer=None, *args, **kwargs):
        super(NoamAdam, self).__init__(*args, **kwargs)
        self.use_noam = use_noam
        if self.use_noam:
            self.d_model = d_model
            self.n_warmup_steps = n_warmup_steps
            self.n_steps = 0
            self.writer = writer

    def set_n_steps(self, n):
        self.n_steps = n

    def step(self):
        "Step with the inner optimizer"
        self.writer.add_scalar('lr', self.param_groups[0]['lr'], self.n_steps)
        if self.use_noam:
            self.n_steps += 1
            for param_group in self.param_groups:
                param_group['lr'] = self._get_lr_scale()
        super(NoamAdam, self).step()

    def _get_lr_scale(self):
        d_model = self.d_model
        n_steps, n_warmup_steps = self.n_steps, self.n_warmup_steps
        return (d_model**-0.5) * min(n_steps**(-0.5), n_steps * n_warmup_steps**(-1.5))
