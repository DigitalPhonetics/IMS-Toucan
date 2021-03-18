def adjust_learning_rate(optimizer, step_num, warmup_steps=8000):
    """
    noam style warmup scheduler, taken from https://github.com/soobinseo/Transformer-TTS
    """

    orig_lr = 1.0

    lr = orig_lr * warmup_steps ** 0.5 * min(step_num * warmup_steps ** -1.5, step_num ** -0.5)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
