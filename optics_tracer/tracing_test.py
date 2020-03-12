"""
Testing with current directory automatically added to sys path:
> python -m pytest
If the import modules are already in sys path, simply
> pytest
"""

from tracing import *
import numpy as np


def test_vec3():
    """Basic assumptions about vector operations"""
    e0 = Vec3(1, 0, 0)
    e1 = Vec3(0, 1, 0)
    e2 = Vec3(0, 0, 1)
    a0 = Vec3(0, 0, 0)
    a1 = Vec3(1, 1, 1)
    assert a0.len() == 0.0
    assert e0.len() == e1.len() == e2.len() == 1.0
    assert a1.len() == np.sqrt(3.0)
    assert a1.normalize().len() == 1
    assert e0.dot(e1) == e1.dot(e2) == 0
    assert e0 + e1 + e2 == a1
    assert e0 - e1 == - (e1 - e0) == -1 * (e1 - e0) == (e1 - e0) * (-1)
    assert e0 - 1 == e0 - 1.0 == -1 + e0 == -1.0 + e0
    assert e0 + 1 == e0 + 1.0 == 1 + e0 == 1.0 + e0
    assert e0 * e1 * e2 == Vec3(0, 0, 0)
    assert e0 / 2 == e0 / 2.0 == 0.5 * e0 == e0 * 0.5
    assert e0.cross(e1) == e2
    a0 = a1
    assert a0 == Vec3(1, 1, 1)


def test_vec3_array():
    """Vec3() is initialized and operated as array of vectors"""
    v1 = Vec3(-2, -1, -1)
    v2 = Vec3(4, 0, 0)
    x = np.arange(0, 10)
    assert v1 + v2*x == v2*x + v1
    # Don't multiply in order np.ndarray * Vec3(), it creates an array of Vec3() objects.


def test_ray():
    ori = Vec3(0, 0, 0)
    dr = Vec3(1, 1, 1)
    r0 = Ray(ori, dr)
    assert r0.point_at_parameter(0) == ori
    assert r0.point_at_parameter(1.0) == r0.point_at_parameter(1) == dr.normalize()
    assert r0.point_at_parameter(2.0) == r0.point_at_parameter(2) == dr.normalize() * 2


def test_sphere():
    t_min = 0
    t_max = 1000
    sphere1 = Sphere()
    sphere100 = Sphere(Vec3(1, 0, 0), 100)
    ray1 = Ray(Vec3(0, 0, 0), Vec3(1, 0, 0))
    # test the unit sphere
    assert sphere1.hit(ray1, t_min, t_max) == 1
    # test the big sphere with center offset
    assert sphere100.hit(ray1, t_min, t_max) == 101


def test_plane():
    t_min = 0
    t_max = 1000
    plane1 = Plane(Vec3(1, 0, 0), Vec3(1, 0, 0))
    ray1 = Ray(Vec3(0, 0, 0), Vec3(1, -1, 0))
    assert plane1.hit(ray1, t_min, t_max) == np.sqrt(2)


def test_refraction():
    ni, nt = 1.0, 1.5
    incident = Vec3(1, 0, 0)
    normal = Vec3(-1, 1, 0).normalize()
    refracted = refract(incident, normal, ni/nt)
    theta_refracted = np.arcsin(ni/nt * np.sin(np.pi/4))
    x_refracted = np.cos(np.pi/4 - theta_refracted)
    y_refracted = - np.sin(np.pi / 4 - theta_refracted)
    assert abs(refracted.len() - 1) < 1e-6
    assert refracted.x == x_refracted
    assert refracted.y == y_refracted
