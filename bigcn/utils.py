import torch


def to_device(gpu):
    if gpu is not None and torch.cuda.is_available():
        device_str = 'cuda:{}'.format(gpu)
    else:
        device_str = 'cpu'
    return torch.device(device_str)


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)
