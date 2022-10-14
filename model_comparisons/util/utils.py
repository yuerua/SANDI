import json

class Options:
    def toJSON(self):
        return json.dumps(self, default=lambda o: o.__dict__,
                          sort_keys=False, indent=4,
                          separators=(',', ': '), ensure_ascii=False)

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    def __getattr__(*args):
        val = dict.get(*args)
        return dotdict(val) if type(val) is dict else val
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
