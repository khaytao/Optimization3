

class ObjectiveFunction:
    def __init__(self, f, df):
        self.f = f
        self.df = df

    def evaluate(self, x):
        return self.f(x)

    def gradient(self, x):
        return self.df(x)