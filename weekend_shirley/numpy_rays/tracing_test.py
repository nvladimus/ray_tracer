"""
Testing with current directory automatically added to sys path:
> python -m pytest
If the import modules are already in sys path, simply
> pytest
"""

from tracing import Vec3, Ray, Sphere, SurfaceList
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
    assert r0.point_at_parameter(1.0) == r0.point_at_parameter(1) == dr
    assert r0.point_at_parameter(2.0) == r0.point_at_parameter(2) == dr * 2.0 == 2 * dr


def test_sphere():
    t_min = 0
    t_max = 1000
    sphere1 = Sphere()
    sphere100 = Sphere(Vec3(1, 0, 0), 100)
    ray1 = Ray(Vec3(0, 0, 0), Vec3(1, 0, 0))
    # test the unit sphere
    assert sphere1.hit(ray1, t_min, t_max)[0] == 1
    # test the big sphere with center offset
    assert sphere100.hit(ray1, t_min, t_max)[0] == 101


def test_world():
    t_min = 0
    t_max = 1000
    sphere1 = Sphere()
    world = SurfaceList([sphere1])
    ray1 = Ray(Vec3(0, 0, 0), Vec3(1, 0, 0))
    # test the unit sphere
    assert world.hit(ray1, t_min, t_max)[0] == 1
