def adjust_learning_rate(optimizer, step_num, warmup_step=16000):
    """
    noam style warmup scheduler, taken from https://github.com/soobinseo/Transformer-TTS
    """
    lr = 0.001 * warmup_step ** 0.5 * min(step_num * warmup_step ** -1.5, step_num ** -0.5)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
