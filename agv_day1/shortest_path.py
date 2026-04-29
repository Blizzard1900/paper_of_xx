from __future__ import annotations

from typing import List, Tuple

import networkx as nx

from .models import Instance


def shortest_path(instance: Instance, source: str, target: str) -> List[str]:
    return nx.shortest_path(instance.graph, source, target, weight="distance")


def shortest_distance(instance: Instance, source: str, target: str) -> float:
    return nx.shortest_path_length(instance.graph, source, target, weight="distance")


def shortest_time(instance: Instance, source: str, target: str) -> float:
    return nx.shortest_path_length(instance.graph, source, target, weight="base_time")


def shortest_energy(instance: Instance, source: str, target: str) -> float:
    return nx.shortest_path_length(instance.graph, source, target, weight="energy_cost")


def shortest_path_metrics(instance: Instance, source: str, target: str) -> Tuple[List[str], float, float, float]:
    path = shortest_path(instance, source, target)
    distance = 0.0
    base_time = 0.0
    energy = 0.0
    for left, right in zip(path, path[1:]):
        edge = instance.edges[(left, right)]
        distance += edge.distance
        base_time += edge.base_time
        energy += edge.energy_cost
    return path, distance, base_time, energy
