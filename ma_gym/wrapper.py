class Wrapper(object):
    def __init__(self, base_class, *args):
        if base_class not in type(self).__bases__:
            type(self).__bases__ += (base_class,)
        super().__init__(*args)