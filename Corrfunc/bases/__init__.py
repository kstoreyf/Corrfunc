#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Helper functions for basis function used in suave continuous-function estimation.
"""
from __future__ import (division, print_function, absolute_import,
                        unicode_literals)

__author__ = ('Manodeep Sinha')
__all__ = ('bao', 'dcosmo', 'spline',)

import sys

from .bao import bao
from .dcosmo import dcosmo
from .spline import spline

if sys.version_info[0] < 3:
    __all__ = [n.encode('ascii') for n in __all__]
