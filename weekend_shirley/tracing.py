"""
This is a library with basic classes and methods for ray tracing.
The structure loosely follows the tutorial "Ray Tracing in a weekend" by P. Shirley.
Implemented by @nvladimus

local import:
sys.path.append('C:/Users/nvladim/Documents/GitHub/ray_tracer/weekend_shirley')
"""
import numpy as np
import copy

class Vec3:
    def __init__(self, *coord):
        """Constructor options:
        Vec3() for empty vector filled with (None) values.
        Vec3(x: float, y: float, z: float)
        Vec3([0, 2, 5.5])
        Vec3(np.array[0, 2, 5.5])
        """
        if len(coord) == 0:
            self.e = np.array([None, None, None])
        elif len(coord) == 1:
            assert len(coord[0]) == 3, "Array-like coordinate must have length 3."
            self.e = np.asarray(coord[0]).astype(float)
        elif len(coord) == 3:
            self.e = np.array([float(coord[0]), float(coord[1]), float(coord[2])])
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
        elif isinstance(other, (float, int)):
            return Vec3(self.e + other)
        else:
            raise ValueError("Second argument must be a vector or a float")

    __radd__ = __add__  # commutative operation

    def __neg__(self):
        return Vec3(-self.e)

    def __sub__(self, other):
        """Vec3 - other"""
        if isinstance(other, Vec3):
            return Vec3(self.e - other.e)
        elif isinstance(other, (float, int)):
            return Vec3(self.e - other)
        else:
            raise ValueError("Second argument must be a vector or a float")

    def __rsub__(self, other):
        """other - Vec3"""
        if isinstance(other, (float, int)):
            return Vec3(other - self.e)

    def __mul__(self, other):
        if isinstance(other, Vec3):
            return Vec3(self.e * other.e)
        elif isinstance(other, (float, int)):
            return Vec3(self.e * other)
        else:
            raise ValueError("Second argument must be a vector or a float")

    __rmul__ = __mul__  # commutative operation

    def __truediv__(self, other):
        if isinstance(other, Vec3):
            return Vec3(self.e / other.e)
        elif isinstance(other, (float, int)):
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
        """Ray class containing origin and direction vectors.
        Constuctor options:
        Ray(origin: Vec3, direction: Vec3)
        Example:
        r = Ray(Vec3(0, 0, 0), Vec(1, 0, 0))"""
        assert isinstance(origin, Vec3), "Origin must be Vec3 class instance"
        assert isinstance(direction, Vec3), "Direction must be Vec3 class instance"
        self.ori = origin
        self.dir = direction

    def point_at_parameter(self, t: float):
        return self.ori + self.dir * float(t)

    def __str__(self):
        return f"Ray origin({self.ori}, direction{self.dir})"


class HitRecord:
    """Structure-like class for saving hit record"""
    def __init__(self, t=None, p=None, normal=None):
        self.t = None
        self.p = None
        self.normal = None
        if t is not None:
            assert isinstance(t, (float, int)), "Parameter (t) must be float or int"
            self.t = float(t)
        if p is not None:
            assert isinstance(p, Vec3), "Parameter (p) must be Vec3()"
            self.p = p
        if normal is not None:
            assert isinstance(normal, Vec3), "Parameter (normal) must be Vec3()"
            self.normal = normal


class Surface:
    """Abstract class for any surface that can be hit by a ray (hitable)."""
    def __init__(self):
        pass

    def hit(self, ray: Ray, t_min: float, t_max: float, record: HitRecord) -> bool:
        """Return True if the surface is hit by the ray"""
        pass


class Sphere(Surface):
    def __init__(self, center=None, radius=None):
        """Constructor options:
            Sphere() for unit sphere at the origin.
            Sphere(center: Vec3, radius: float) for any other sphere.
        """
        if center is None:
            self.center = Vec3([0, 0, 0])
        else:
            assert isinstance(center, Vec3), "Sphere's center must be Vec3 instance."
            self.center = center
        if radius is None:
            self.radius = 1.0
        else:
            assert isinstance(radius, (float, int)), "Sphere's radius must be float or int."
            self.radius = radius

    def hit(self, ray: Ray, t_min: float, t_max: float, record: HitRecord) -> bool:
        """Returns True of the ray hits the sphere.
        If True, the argument record: HitRecord saves the hit record. """
        oc = ray.ori - self.center
        a = ray.dir.dot(ray.dir)
        b = oc.dot(ray.dir)
        c = oc.dot(oc) - self.radius**2
        discriminant = b**2 - a * c
        if discriminant > 0:
            temp = (-b - np.sqrt(discriminant)) / a
            if t_min < temp < t_max:
                record.t = temp
                record.p = ray.point_at_parameter(record.t)
                record.normal = (record.p - self.center) / self.radius
                return True
            temp = (-b + np.sqrt(discriminant)) / a
            if t_min < temp < t_max:
                record.t = temp
                record.p = ray.point_at_parameter(record.t)
                record.normal = (record.p - self.center) / self.radius
                return True
        else:
            return False


class SurfaceList(Surface):
    def __init__(self, surf_list):
        assert len(surf_list) > 0, "Provide a non-empty list of surfaces."
        self.list = surf_list

    def hit(self, ray: Ray, t_min: float, t_max: float, record: HitRecord) -> bool:
        assert len(self.list) > 0, "Provide a non-empty list of surfaces."
        temp_rec = HitRecord()
        hit_anything = False
        closest_so_far = t_max
        for surf in self.list:
            if surf.hit(ray, t_min, t_max, temp_rec):
                hit_anything = True
                if temp_rec.t < closest_so_far:
                    closest_so_far = temp_rec.t
                    (record.t, record.p, record.normal) = (temp_rec.t, temp_rec.p, temp_rec.normal)
        return hit_anything
