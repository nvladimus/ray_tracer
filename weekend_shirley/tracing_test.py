"""
Testing with current directory automatically added to sys path:
> python -m pytest
If the import modules are already in sys path, simply
> pytest
"""

from tracing import Vec3, Ray
import numpy as np


def test_vec3():
    e0 = Vec3(1, 0, 0)
    e1 = Vec3(0, 1, 0)
    e2 = Vec3(0, 0, 1)
    a0 = Vec3(0, 0, 0)
    a1 = Vec3(1, 1, 1)
    assert a0.len() == 0.0
    assert e0.len() == e1.len() == e2.len() == 1.0
    assert a1.len() == np.sqrt(3.0)
    assert e0.dot(e1) == e1.dot(e2) == 0
    assert e0 + e1 + e2 == a1
    assert e0.cross(e1) == e2
    # try different initialization styles:
    assert Vec3(1, 0, 0) == Vec3([1, 0, 0]) == Vec3(np.array([1, 0, 0]))


def test_ray():
    ori = Vec3(0, 0, 0)
    dr = Vec3(1, 1, 1)
    r0 = Ray(ori, dr)
    assert r0.point_at_parameter(0) == ori
    assert r0.point_at_parameter(1.0) == dr
    assert r0.point_at_parameter(2.0) == dr * 2.0

