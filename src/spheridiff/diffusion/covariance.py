import math


class IdentityCovariance:
    def apply(self, field):
        return field


class SpectralCovariance:
    def __init__(self, transform):
        self.transform = transform
        self.scale = 4.0 * math.pi / transform.coefficients

    def apply(self, field):
        return self.scale * self.transform.synthesise(self.transform.synthesis_adjoint(field))
