from functools import partial
from torch.optim.lr_scheduler import LambdaLR

def _get_lr_linear_decay_lr_lambda(current_step: int, *, num_keep_steps: int, num_training_steps: int):
    if current_step < num_keep_steps:
        return 1.0
    return max(0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_keep_steps)))

def get_lr_linear_decay(optimizer, num_keep_steps, num_training_steps, last_epoch=-1):
    lr_lambda = partial(
        _get_lr_linear_decay_lr_lambda,
        num_keep_steps=num_keep_steps,
        num_training_steps=num_training_steps,
    )
    return LambdaLR(optimizer, lr_lambda, last_epoch)