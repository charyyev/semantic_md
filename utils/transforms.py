import numpy as np
import torch
import torchvision.transforms.functional as TF

#compose several transformations
class Compose(object):
    def __init__(self, transforms, p=1.0):
        self.transforms = transforms
        self.p = p

    def __call__(self, image, label):
        if np.random.random() <= self.p:
            for t in self.transforms:
                image, label = t(image, label)
        return image, label

# select one of the transforms and apply
class OneOf(object):
    def __init__(self, transforms, p=1.0):
        self.transforms = transforms
        self.p = p

    def __call__(self, image, label):
        if np.random.random() < self.p and len(self.transforms) > 0:
            choice = np.random.randint(low=0, high=len(self.transforms))
            image, label = self.transforms[choice](image, label)

        return image, label

# apply random rotation with angle within limit_angle and with probability p
class Random_Rotation(object):
    def __init__(self, limit_angle=20.):
        self.limit_angle = limit_angle

    def __call__(self, image, label):
        angle = np.random.uniform(-self.limit_angle, self.limit_angle)
        image = TF.rotate(image, angle)
        label = TF.rotate(label, angle)

        return image, label

# apply affine transformation within provided limits
class Affine(object):
    def __init__(self, limit_angle, limit_translation, limit_shear, limit_scale):
        self.limit_angle = limit_angle
        self.limit_translation = limit_translation
        self.limit_shear = limit_shear
        self.limit_scale = limit_scale

    def __call__(self, image, label):
        angle = np.random.uniform(-self.limit_angle, self.limit_angle)
        shear_angle = np.random.uniform(-self.limit_shear, self.limit_shear)
        scale = np.random.uniform(self.limit_scale[0], self.limit_scale[1])
        translate_x = np.random.randint(self.limit_translation)
        translate_y = np.random.randint(self.limit_translation)
        translate = [translate_x, translate_y]
        image = TF.affine(image, angle, translate, scale, shear = shear_angle)
        label = TF.affine(label, angle, translate, scale, shear = shear_angle)
            
        return image, label