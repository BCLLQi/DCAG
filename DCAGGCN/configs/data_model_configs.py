
def get_dataset_class(dataset_name):

    if dataset_name not in globals():
        raise NotImplementedError("Dataset not found: {}".format(dataset_name))

    return globals()[dataset_name]


class PHM2012():
    def __init__(self):
        super(PHM2012, self)
        self.sequence_len = 2560
        self.input_channels = 1

        self.shuffle = False
        self.drop_last = False
        self.normalize = False


class XJTU_SY():
    def __init__(self):
        super(XJTU_SY, self)
        self.sequence_len = 30768
        self.input_channels = 1

        self.shuffle = False
        self.drop_last = False
        self.normalize = False