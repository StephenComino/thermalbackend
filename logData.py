class logData:
    def __init__(self):
        self.x_cords = []
        self.y_cords = []
        self.max_temp = 0
        self.min_temp = 0
        self.avg_temp = 0
        self.distance = 0
    def __getstate__(self):
     return self.__dict__
    def __setstate__(self, d):
     self.__dict__ = d