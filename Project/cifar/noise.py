import numpy as np


class GaussNoise:
    def __init__(self, mean=(0, 64), std=(0.02, 0.1)):
        self.mean = np.random.uniform(mean[0], mean[1])
        self.std = np.random.uniform(std[0], std[1])

    def __call__(self, image):
        h, w, c = image.shape
        noise = np.random.rand(h, w, c) * self.std + self.mean
        out = image + noise
        out = np.clip(out, 0, 255).astype(np.uint8)
        return out


class SaltAndPepper:
    def __init__(self, p=0.025):
        super(SaltAndPepper, self).__init__()
        self.p = p

    def __call__(self, image):
        h, w = image.shape[:2]
        output = image.copy()
        threshold = 1 - self.p
        for i in range(h):
            for j in range(w):
                rdn = np.random.random()  # 随机生成0-1之间的数字
                if rdn < self.p:
                    output[i][j] = [0, 0, 0]
                elif rdn > threshold:
                    output[i][j] = [255, 255, 255]
        return output
