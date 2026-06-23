#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
graphrep — wspólny framework: reprezentacje grafowe → klastrowanie → bezpieczeństwo.

Spina trzy nurty pracy (enzymy/białka, CIFAR kolegów, grafy losowe) w jedną oś:
graf jako reprezentacja OBFUSKUJĄCA dane, badana pod kątem użyteczności
(klastrowanie/separowalność) ORAZ prywatności (wyciek klasy + odwracalność).
"""
from .config import Config
from . import data, features, embeddings, evaluate, plots

__all__ = ["Config", "data", "features", "embeddings", "evaluate", "plots"]
