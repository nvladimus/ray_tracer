"""
A basic tracer for optics (lenses etc).
local import:
sys.path.append('C:/Users/username/Documents/GitHub/ray_tracer/optics_tracer/')
by @nvladimus
"""
import numpy as np
import sys


class Vec3:
    def __init__(self, x=None, y=None, z=None):
        """This class works for both individual vectors and numpy arrays of vectors.
        Initialization examples:

        Scalar coordinates:
        v = Vec3(): empty vector filled with zero values.
        v = Vec3(0, 1.1, 2)

        Array coordinates:
        x = np.tile(np.linspace(0,2,200), 100)
        y = np.repeat(np.linspace(0,1,100), 200)
        z = np.zeros(200*100)
        v = Vec3(x, y, z)
        """
        if (x is None) and (y is None) and (z is None):
            self.x, self.y, self.z = 0, 0, 0
        elif isinstance(x, (float, int)) and isinstance(y, (float, int)) and isinstance(z, (float, int)):
                self.x, self.y, self.z = float(x), float(y), float(z)
        elif len(x) == len(y) == len(z):
                self.x, self.y, self.z = np.array(x).astype(float), np.array(y).astype(float), np.array(z).astype(float)
        else:
            raise ValueError("Initialize as Vec3(x,y,z) where x, y, z are scalars or arrays.")

    def dot(self, other):
        return self.x * other.x + self.y * other.y + self.z * other.z

    def cross(self, other):
        return Vec3(self.y * other.z - self.z * other.y,
                    -(self.x * other.z - self.z * other.x),
                    self.x * other.y - self.y * other.x)

    def __eq__(self, other):
        return np.equal(self.x, other.x).all() and np.equal(self.y, other.y).all() and np.equal(self.z, other.z).all()

    def __add__(self, other):
        if isinstance(other, Vec3):
            return Vec3(self.x + other.x, self.y + other.y, self.z + other.z)
        elif isinstance(other, (float, int, np.ndarray)):
            return Vec3(self.x + other, self.y + other, self.z + other)
        else:
            raise ValueError("Second argument must be a Vec3(), np.ndarray, int, or float.")

    __radd__ = __add__ # commutative operation

    def __neg__(self):
        return Vec3(-self.x, -self.y, -self.z)

    def __sub__(self, other):
        """Vec3 - other"""
        if isinstance(other, Vec3):
            return Vec3(self.x - other.x, self.y - other.y, self.z - other.z)
        elif isinstance(other, (float, int, np.ndarray)):
            return Vec3(self.x - other, self.y - other, self.z - other)
        else:
            raise ValueError("Second argument must be a Vec3(), int, or float.")

    def __rsub__(self, other):
        """other - Vec3"""
        if isinstance(other, (float, int, np.ndarray)):
            return Vec3(other - self.x, other - self.y, other - self.z)

    def __mul__(self, other):
        if isinstance(other, Vec3):
            return Vec3(self.x * other.x, self.y * other.y, self.z * other.z)
        elif isinstance(other, (float, int, np.ndarray)):
            return Vec3(self.x * other, self.y * other, self.z * other)
        else:
            raise ValueError("Second argument must be a Vec3(), np.ndarray, int, or float.")

    __rmul__ = __mul__  # commutative operation. Does not work for np.ndarray, because of their __mul__ operator

    def __truediv__(self, other):
        if isinstance(other, Vec3):
            return Vec3(self.x / other.x, self.y / other.y, self.z / other.z)
        elif isinstance(other, (float, int, np.ndarray)):
            return Vec3(self.x / other, self.y / other, self.z / other)
        else:
            raise ValueError("Second argument must be a Vec3(), int, or float.")

    def len(self):
        return np.sqrt(self.x*self.x + self.y*self.y + self.z*self.z)

    def __str__(self):
        return f"Vec3({self.x},{self.y},{self.z})"

    def normalize(self):
        if self.len() > 1e-6:
            return Vec3(self.x / self.len(), self.y / self.len(), self.z / self.len())
        else:
            return Vec3(0, 0, 0)

    def where(self, condition, new_vector):
        """Reset vector values that satisfy the condition, eg (discriminant == 0), to the new vector"""
        if isinstance(self.x, float) and condition:
            return new_vector
        elif isinstance(self.x, np.ndarray) and condition:
            self.x[condition] = Vec3.x
            self.y[condition] = Vec3.y
            self.z[condition] = Vec3.z
            return self
        else:
            return self


class Ray:
    def __init__(self, origin: Vec3, direction: Vec3):
        """Ray class containing origin and direction vectors.
        The direction vector gets normalized.
        Initialization options:
        Ray(origin: Vec3, direction: Vec3)
        Example:
        r = Ray(Vec3(0, 0, 0), Vec(1, 0, 0))"""
        assert isinstance(origin, Vec3), "Origin must be Vec3 class instance"
        assert isinstance(direction, Vec3), "Direction must be Vec3 class instance"
        self.ori = origin
        self.dir = direction.normalize()

    def point_at_parameter(self, t: float) -> Vec3:
        return self.ori + self.dir * t

    def __str__(self):
        return f"Ray origin {self.ori}, direction {self.dir}"


def reflect(incident: Vec3, normal: Vec3) -> Vec3:
    """Compute reflected ray"""
    reflected = incident - normal * 2 * incident.dot(normal)
    return reflected


def refract(incident: Vec3, normal: Vec3, ni_over_nt: float) -> Vec3:
    """Ray refraction"""
    incident_norm = incident.normalize()
    dt = incident_norm.dot(normal)
    discriminant = np.maximum(1.0 - ni_over_nt*ni_over_nt * (1.0 - dt*dt), 0)
    refracted = ni_over_nt * (incident_norm - normal * dt) - normal * np.sqrt(discriminant)
    refracted = refracted.where(discriminant == 0, Vec3(0, 0, 0)).normalize()
    return refracted


class Surface:
    """Abstract class for any surface that can be hit by a ray."""
    def __init__(self):
        pass

    def hit(self, ray: Ray, t_min: float, t_max: float) -> (np.ndarray, Vec3):
        """Returns ray parameter (ndarray) where the ray hits the surface,
        and vector (Vec3) containing normals at the hit points."""
        pass


class Sphere(Surface):
    def __init__(self, center: Vec3 = None, radius: Vec3 = None):
        """Constructor options:
            Sphere() for unit sphere at the origin.
            Sphere(center: Vec3, radius: float) for any other sphere.
        """
        if center is None:
            self.center = Vec3(0, 0, 0)
        else:
            assert isinstance(center, Vec3), "Sphere's center must be Vec3 instance."
            self.center = center
        if radius is None:
            self.radius = 1.0
        else:
            assert isinstance(radius, (float, int)), "Sphere's radius must be float or int."
            self.radius = radius

    def hit(self, ray: Ray, t_min: float, t_max: float) -> np.ndarray:
        """Returns ray parameter (t) it the ray hits the sphere.
        Ray direction vector can be individual or numpy.ndarray.
        """
        oc = ray.ori - self.center
        a = ray.dir.dot(ray.dir)
        b = oc.dot(ray.dir)
        c = oc.dot(oc) - self.radius**2
        discriminant = b*b - a * c
        sq = np.sqrt(np.maximum(0, discriminant))
        t1 = (-b - sq) / a
        t2 = (-b + sq) / a
        cond_t1 = (t1 < t2) & (t_min < t1) & (t1 < t_max)
        t_hits = np.where(cond_t1, t1, t2)
        cond_t2 = (discriminant > 0) & (t_min < t_hits) & (t_hits < t_max)
        t_hits = np.where(cond_t2, t_hits, t_max)
        return t_hits


class Plane(Surface):
    def __init__(self, point: Vec3, normal: Vec3):
        """Plane defined by a point lying on it (Vec3) and a normal vector.
        The normal vector gets normalized."""
        assert isinstance(point, Vec3), "Point must be Vec3() class instance"
        assert isinstance(normal, Vec3), "Normal must be Vec3() class instance"
        self.point = point
        self.normal = normal.normalize()

    def hit(self, ray: Ray, t_min: float, t_max: float) -> np.ndarray:
        """Find the ray parameter where it intersects the plane"""
        denom = np.maximum(ray.dir.dot(self.normal), 1e-6)
        t = (self.point - ray.ori).dot(self.normal) / denom
        t_hits = np.where((t_min < t) & (t < t_max), t, t_max)
        return t_hits


class Disk(Surface):
    def __init__(self, center: Vec3, normal: Vec3, radius: float):
        """Plane defined by a point lying on it (Vec3) and a normal vector.
                The normal vector gets normalized."""
        assert isinstance(center, Vec3), "Point must be Vec3() class instance"
        assert isinstance(normal, Vec3), "Normal must be Vec3() class instance"
        assert isinstance(radius, (float, int)), "Radius must be float or int"
        self.center = center
        self.normal = normal.normalize()
        self.radius = float(radius)

    def hit(self, ray: Ray, t_min: float, t_max: float) -> np.ndarray:
        """Find the ray parameter where it intersects the disk"""
        # find intersection with disk plane
        denom = np.maximum(ray.dir.dot(self.normal), 1e-6)
        t = (self.center - ray.ori).dot(self.normal) / denom
        # compute distance from the center
        disk_vectors = ray.point_at_parameter(t) - self.center
        t_hits = np.where((t_min < t) & (t < t_max) & (disk_vectors.len() < self.radius), t, t_max)
        return t_hits

