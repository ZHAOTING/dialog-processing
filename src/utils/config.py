class ConfigFromDict(object):
    def __init__(self, attr_dict):
        for k, v in attr_dict.items():
            setattr(self, k, v)
