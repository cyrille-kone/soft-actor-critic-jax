# coding=utf-8
"""
PyCharm Editor
@git Team
"""
import chex
import acme
import enum
import typing
# on peut definir tous nos customs type ici

@chex.dataclass
class Trajectory(object):
  observations: acme.types.NestedArray  # [T, B, ...]
  actions: acme.types.NestedArray  # [T, B, ...]
  rewards: chex.ArrayNumpy  # [T, B]
  dones: chex.ArrayNumpy  # [T, B]
  discounts: chex.ArrayNumpy # [T, B]
