"""
This is a library with basic classes and methods for ray tracing.
The structure loosely follows the tutorial "Ray Tracing in a weekend" by P. Shirley.
Implemented by @nvladimus

local import:
sys.path.append('C:/Users/nvladim/Documents/GitHub/ray_tracer/weekend_shirley')
"""
import numpy as np


class Vec3:
    def __init__(self, *coord):
        if len(coord) == 3:
            self.e = np.array([float(coord[0]), float(coord[1]), float(coord[2])])
        elif len(coord) == 1:
            assert len(coord[0]) == 3, "Array-like coordinate must have length 3."
            self.e = np.asarray(coord[0]).astype(float)
        else:
            raise ValueError("Initialize as Vec3(x,y,z) or Vec([x,y,z]).")

    def dot(self, other):
        return np.sum(self.e * other.e)

    def cross(self, other):
        return Vec3(self.e[1] * other.e[2] - self.e[2] * other.e[1],
                    -(self.e[0] * other.e[2] - self.e[2] * other.e[0]),
                    self.e[0] * other.e[1] - self.e[1] * other.e[0])

    def __eq__(self, other):
        return (self.e == other.e).all()

    def __add__(self, other):
        if isinstance(other, Vec3):
            return Vec3(self.e + other.e)
        elif isinstance(other, float):
            return Vec3(self.e + other)
        else:
            raise ValueError("Second argument must be a vector or a float")

    def __neg__(self):
        return Vec3(-self.e)

    def __sub__(self, other):
        if isinstance(other, Vec3):
            return Vec3(self.e - other.e)
        elif isinstance(other, float):
            return Vec3(self.e - other)
        else:
            raise ValueError("Second argument must be a vector or a float")

    def __mul__(self, other):
        if isinstance(other, Vec3):
            return Vec3(self.e * other.e)
        elif isinstance(other, float):
            return Vec3(self.e * other)
        else:
            raise ValueError("Second argument must be a vector or a float")

    def __div__(self, other):
        if isinstance(other, Vec3):
            return Vec3(self.e / other.e)
        elif isinstance(other, float):
            return Vec3(self.e / other)
        else:
            raise ValueError("Second argument must be a vector or a float")

    def len(self):
        return np.sqrt(self.e[0]**2 + self.e[1]**2 + self.e[2]**2)

    def __str__(self):
        return f"Vect3({self.e})"

    def normalize(self):
        return Vec3(self.e / self.len())


class Ray:
    def __init__(self, origin: Vec3, direction: Vec3):
        assert isinstance(origin, Vec3), "Origin must be Vec3 class instance"
        assert isinstance(direction, Vec3), "Direction must be Vec3 class instance"
        self.ori = origin
        self.dir = direction

    def point_at_parameter(self, t: float):
        return self.ori + self.dir * float(t)


