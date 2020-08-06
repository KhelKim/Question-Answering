import json

def read_text(path):
    with open(path) as f:
        data_raw = f.readlines()
    return data_raw

def read_json(path):
    with open(path) as j:
        result = json.load(j)
    return result

def save_json(path, obj):
    with open(path, 'w') as j:
        json.dump(obj, j, ensure_ascii=False)

def make_bool(string):
    if string.lower() in ["true", 'y']:
        return True
    return False

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        return "{} {:.3f} ({:.3f})".format(self.name, self.val, self.avg)
