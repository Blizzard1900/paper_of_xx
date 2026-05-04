from __future__ import annotations

import math
import os
import random
from dataclasses import dataclass, field
from pathlib import Path
from pprint import pprint
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import networkx as nx

try:
    import tkinter as tk
    from tkinter import messagebox, ttk
    from tkinter.scrolledtext import ScrolledText
except Exception:
    tk = None
    ttk = None
    messagebox = None
    ScrolledText = None


SECONDS_PER_MINUTE = 60.0


@dataclass(frozen=True)
class GlobalParams:
    speed_mps: float = 0.5
    energy_ratio_per_m: float = 5e-4
    battery_capacity_kwh: float = 10.0
    battery_min_kwh: float = 2.0
    battery_threshold_kwh: float = 4.0
    charge_power_kw: float = 10.0
    lane_count: int = 3
    lane_capacity: int = 1
    corridor_capacity: int = 3
    lane_change_penalty_s: float = 2.0
    congestion_alpha: float = 0.6
    worker_delay_rate: float = 0.02
    default_pick_service_time_s: float = 30.0
    default_drop_service_time_s: float = 30.0
    replenishment_trigger: float = 20.0
    replenishment_qty: float = 40.0
    safety_stock: float = 5.0
    line_stop_penalty_weight: float = 3000.0
    time_window_penalty_weight: float = 500.0
    congestion_penalty_weight: float = 10.0
    inventory_holding_weight: float = 0.1
    simulation_horizon_s: float = 7200.0
    simulation_step_s: float = 30.0
    cycle_sigma_min: float = 0.01
    qualified_rate: float = 0.95
    target_qualified_units: float = 50.0
    target_completion_penalty_weight: float = 2000.0
    output_buffer_capacity: float = 120.0
    random_seed: int = 42

    @property
    def energy_kwh_per_m(self) -> float:
        return self.energy_ratio_per_m * self.battery_capacity_kwh

    @property
    def expected_worker_delay_s(self) -> float:
        return 1.0 / self.worker_delay_rate

    @property
    def cycle_sigma_s(self) -> float:
        return self.cycle_sigma_min * SECONDS_PER_MINUTE


@dataclass(frozen=True)
class Node:
    node_id: str
    node_type: str
    region: str
    x: float
    y: float
    service_capable: bool = False


@dataclass(frozen=True)
class Edge:
    edge_id: str
    from_node: str
    to_node: str
    distance: float
    base_time: float
    energy_cost: float
    capacity: int
    edge_type: str
    is_production_edge: bool
    resource_id: str


@dataclass(frozen=True)
class Task:
    task_id: str
    task_type: str
    pickup_node: str
    drop_node: str
    qty: float
    earliest_start: float
    latest_finish: float
    pickup_service_time: float
    drop_service_time: float
    source_inventory_id: Optional[str] = None
    target_inventory_id: Optional[str] = None
    generated_time: float = 0.0


@dataclass(frozen=True)
class AGV:
    agv_id: str
    start_node: str
    battery_init: float
    battery_max: float
    battery_threshold: float
    battery_min: float
    charge_power: float
    load_capacity: float
    status: str


@dataclass(frozen=True)
class InventoryBin:
    inventory_id: str
    node_id: str
    item_type: str
    role: str
    capacity: float
    initial_level: float
    reorder_point: float
    replenish_qty: float
    safety_stock: float
    line_id: Optional[str] = None
    workstation_id: Optional[str] = None


@dataclass(frozen=True)
class WorkstationConfig:
    workstation_id: str
    line_id: str
    node_id: str
    input_demands: Dict[str, float]
    output_flows: Dict[str, float]
    cycle_time_s: float

    @property
    def rate_per_second(self) -> float:
        return 1.0 / self.cycle_time_s


@dataclass
class Instance:
    global_params: GlobalParams
    nodes: Dict[str, Node]
    edges: Dict[Tuple[str, str], Edge]
    graph: nx.DiGraph
    agvs: Dict[str, AGV]
    chargers: List[str]
    inventory_bins: Dict[str, InventoryBin]
    workstations: Dict[str, WorkstationConfig]
    supply_sources: Dict[str, List[str]]
    qualified_storage_bins: List[str]
    defect_storage_bins: List[str]
    shortest_path_matrix: Dict[Tuple[str, str], List[str]] = field(default_factory=dict)
    shortest_metric_matrix: Dict[Tuple[str, str], Tuple[float, float, float]] = field(default_factory=dict)
    critical_buffers_by_line: Dict[str, List[str]] = field(default_factory=dict)


@dataclass
class EdgeTraversal:
    agv_id: str
    from_node: str
    to_node: str
    resource_id: str
    start_time: float
    end_time: float
    duration: float
    distance: float
    energy_cost: float
    wait_before: float
    edge_type: str


@dataclass
class ChargingEvent:
    charger_id: str
    start_time: float
    end_time: float
    battery_before: float
    battery_after: float


@dataclass
class TaskExecution:
    task_id: str
    agv_id: str
    pickup_arrival: float
    pickup_start: float
    pickup_end: float
    drop_arrival: float
    drop_start: float
    drop_end: float
    late_amount: float


@dataclass
class BatteryRecord:
    time: float
    battery: float
    note: str


@dataclass
class AGVDecodedSchedule:
    agv_id: str
    assigned_tasks: List[str] = field(default_factory=list)
    visited_nodes: List[str] = field(default_factory=list)
    traversals: List[EdgeTraversal] = field(default_factory=list)
    charging_events: List[ChargingEvent] = field(default_factory=list)
    task_executions: List[TaskExecution] = field(default_factory=list)
    battery_trace: List[BatteryRecord] = field(default_factory=list)
    total_distance: float = 0.0
    total_energy: float = 0.0
    total_wait: float = 0.0
    final_node: Optional[str] = None
    final_time: float = 0.0
    final_battery: float = 0.0


@dataclass
class InventorySnapshot:
    time: float
    levels: Dict[str, float]
    line_stop_ratio: Dict[str, float]


@dataclass
class InventorySimulationResult:
    horizon: float
    snapshots: List[InventorySnapshot]
    final_levels: Dict[str, float]
    line_stop_time: float
    average_inventory: float
    stop_by_line: Dict[str, float]
    qualified_output_total: float
    defect_output_total: float
    capacity_penalty: float


@dataclass
class DecodedSolution:
    agv_schedules: Dict[str, AGVDecodedSchedule]
    task_executions: Dict[str, TaskExecution]
    assignment_map: Dict[str, List[str]]


@dataclass
class Evaluation:
    fitness: float
    makespan: float
    total_distance: float
    total_energy: float
    late_penalty: float
    late_task_count: int
    charge_count: int
    congestion_wait: float
    battery_penalty: float
    capacity_penalty: float
    load_penalty: float
    line_stop_time: float
    average_inventory: float
    stop_penalty: float
    target_shortfall_penalty: float
    inventory_result: InventorySimulationResult
    agv_schedules: Dict[str, AGVDecodedSchedule]


@dataclass
class ExperimentRunResult:
    instance: Instance
    task_pool: Dict[str, Task]
    planning_levels: Dict[str, float]
    manual_decoded: DecodedSolution
    manual_evaluation: Evaluation
    random_solution: Dict[str, object]
    random_evaluation: Evaluation
    best_solution: Dict[str, object]
    best_decoded: DecodedSolution
    best_evaluation: Evaluation
    ga_result: GAResult


@dataclass(frozen=True)
class FitnessWeights:
    makespan: float = 1.0
    total_distance: float = 1.0
    total_energy: float = 60.0
    congestion_wait: float = 10.0
    line_stop_time: float = 200.0
    late_penalty: float = 500.0
    battery_penalty: float = 3000.0
    capacity_penalty: float = 3000.0
    load_penalty: float = 2000.0
    inventory_holding: float = 0.1


@dataclass(frozen=True)
class GAConfig:
    population_size: int = 50
    generations: int = 80
    crossover_rate: float = 0.8
    mutation_rate: float = 0.15
    elite_ratio: float = 0.05
    tournament_size: int = 3
    heuristic_seed_count: int = 10


@dataclass
class GAIndividual:
    solution: Dict[str, object]
    evaluation: Optional[Evaluation] = None

    @property
    def fitness(self) -> float:
        if self.evaluation is None:
            raise ValueError("Individual has not been evaluated.")
        return self.evaluation.fitness


@dataclass
class GAGenerationStats:
    generation: int
    best_fitness: float
    average_fitness: float
    best_makespan: float
    best_line_stop: float


@dataclass
class GAResult:
    best_individual: GAIndividual
    history: List[GAGenerationStats]


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(value, upper))


def add_node(
    nodes: Dict[str, Node],
    node_id: str,
    node_type: str,
    region: str,
    x: float,
    y: float,
    *,
    service_capable: bool = False,
) -> None:
    nodes[node_id] = Node(
        node_id=node_id,
        node_type=node_type,
        region=region,
        x=x,
        y=y,
        service_capable=service_capable,
    )


def manhattan(a: Node, b: Node) -> float:
    return abs(a.x - b.x) + abs(a.y - b.y)


def resource_id_for_edge(source: str, target: str, edge_type: str) -> str:
    if edge_type.startswith("lane"):
        return f"lane:{tuple(sorted((source, target)))}"
    if edge_type.startswith("charger"):
        return f"charger-link:{tuple(sorted((source, target)))}"
    return f"edge:{source}->{target}"


def make_edge(
    nodes: Dict[str, Node],
    params: GlobalParams,
    source: str,
    target: str,
    *,
    edge_type: str,
    capacity: int = 1,
    is_production_edge: bool = False,
) -> Edge:
    distance = manhattan(nodes[source], nodes[target])
    return Edge(
        edge_id=f"E_{source}__{target}",
        from_node=source,
        to_node=target,
        distance=distance,
        base_time=distance / params.speed_mps,
        energy_cost=distance * params.energy_kwh_per_m,
        capacity=capacity,
        edge_type=edge_type,
        is_production_edge=is_production_edge,
        resource_id=resource_id_for_edge(source, target, edge_type),
    )


def add_bidirectional_edge(
    edge_map: Dict[Tuple[str, str], Edge],
    nodes: Dict[str, Node],
    params: GlobalParams,
    source: str,
    target: str,
    *,
    edge_type: str,
    capacity: int = 1,
    is_production_edge: bool = False,
) -> None:
    edge_map[(source, target)] = make_edge(
        nodes,
        params,
        source,
        target,
        edge_type=edge_type,
        capacity=capacity,
        is_production_edge=is_production_edge,
    )
    edge_map[(target, source)] = make_edge(
        nodes,
        params,
        target,
        source,
        edge_type=edge_type,
        capacity=capacity,
        is_production_edge=is_production_edge,
    )


def connect_chain(
    edge_map: Dict[Tuple[str, str], Edge],
    nodes: Dict[str, Node],
    params: GlobalParams,
    ordered_nodes: Iterable[str],
    *,
    edge_type: str,
    capacity: int = 1,
    is_production_edge: bool = False,
) -> None:
    node_list = list(ordered_nodes)
    for left, right in zip(node_list, node_list[1:]):
        add_bidirectional_edge(
            edge_map,
            nodes,
            params,
            left,
            right,
            edge_type=edge_type,
            capacity=capacity,
            is_production_edge=is_production_edge,
        )


def sample_cycle_time(mean_min: float, params: GlobalParams, rng: random.Random) -> float:
    sampled_min = max(0.1, rng.gauss(mean_min, params.cycle_sigma_min))
    return sampled_min * SECONDS_PER_MINUTE


def build_nodes(params: GlobalParams) -> Dict[str, Node]:
    nodes: Dict[str, Node] = {}

    for node_id, x, y in [
        ("J1", -5.0, 32.0),
        ("J2", -60.0, 32.0),
        ("J3", -5.0, 12.0),
        ("J4", 1.0, 12.0),
        ("J5", -5.0, 0.0),
        ("J6", -39.0, 0.0),
        ("J7", 2.5, 34.5),
        ("J8", 2.5, 2.5),
        ("J9", 2.5, -0.3),
        ("J10", 2.5, -31.8),
    ]:
        add_node(nodes, node_id, "junction", "road", x, y)

    for idx in range(1, 11):
        add_node(
            nodes,
            f"C{idx}",
            "charger",
            "charging",
            10.5 + 1.5 * (idx - 1),
            43.5,
            service_capable=False,
        )

    for idx in range(1, 4):
        lane_x = 1.0 + 0.5 * (idx - 1)
        add_node(nodes, f"LANE_{idx}_TOP", "lane", "lane", lane_x, 34.0)
        add_node(nodes, f"LANE_{idx}_BOTTOM", "lane", "lane", lane_x, -31.8)

    for row in range(1, 11):
        for col in range(1, 13):
            add_node(
                nodes,
                f"S_{row}_{col}",
                "storage",
                "raw_storage",
                4.2 + 3.2 * (col - 1),
                5.2 + 3.2 * (row - 1),
                service_capable=True,
            )

    for row in range(1, 10):
        for col in range(1, 13):
            add_node(
                nodes,
                f"F_{row}_{col}",
                "finished_storage",
                "finished_storage",
                4.2 + 3.2 * (col - 1),
                -3.0 - 3.2 * (row - 1),
                service_capable=True,
            )

    for col in range(1, 13):
        add_node(
            nodes,
            f"D_{col}",
            "defect_storage",
            "defect_storage",
            4.2 + 3.2 * (col - 1),
            -31.8,
            service_capable=True,
        )

    for idx in range(1, 14):
        x = -34.0 - 2.0 * (idx - 1)
        add_node(nodes, f"L1_WS_{idx}", "workstation", "line1", x, 32.5)
        add_node(nodes, f"L1_BUF_{idx}", "buffer", "line1_buffer", x, 34.5, service_capable=True)

    for idx in range(1, 6):
        x = -17.0 - 3.0 * (idx - 1)
        add_node(nodes, f"L2_WS_{idx}", "workstation", "line2", x, 31.0)
        add_node(nodes, f"L2_B_BUF_{idx}", "buffer", "line2_buffer", x, 28.6, service_capable=True)
        add_node(nodes, f"L2_S1_BUF_{idx}", "buffer", "line2_buffer", x, 26.6, service_capable=True)

    for idx in range(1, 4):
        x = -33.0 - 8.0 * (idx - 1)
        add_node(nodes, f"L3_WS_{idx}", "workstation", "line3", x, 31.0)
        add_node(nodes, f"L3_C_BUF_{idx}", "buffer", "line3_buffer", x, 28.6, service_capable=True)
        add_node(nodes, f"L3_S2_BUF_{idx}", "buffer", "line3_buffer", x, 26.6, service_capable=True)

    for idx in range(1, 10):
        x = -20.0 - 2.0 * (idx - 1)
        add_node(nodes, f"L4_WS_{idx}", "workstation", "line4", x, 0.0)
        add_node(nodes, f"L4_D_BUF_{idx}", "buffer", "line4_buffer", x, 2.4, service_capable=True)
        add_node(nodes, f"L4_P_BUF_{idx}", "buffer", "line4_buffer", x, 4.4, service_capable=True)

    for node_id, x, y in [
        ("A_END", 32.5, -60.0),
        ("B_END", 30.5, -27.0),
        ("C_END", 31.0, -33.0),
        ("D_END", -36.0, 0.0),
        ("E_END", -39.0, 0.0),
    ]:
        add_node(nodes, node_id, "line_end_buffer", "buffer", x, y, service_capable=True)

    return nodes


def build_edges(nodes: Dict[str, Node], params: GlobalParams) -> Dict[Tuple[str, str], Edge]:
    edge_map: Dict[Tuple[str, str], Edge] = {}

    connect_chain(
        edge_map,
        nodes,
        params,
        ["J2"] + [f"L1_WS_{idx}" for idx in range(13, 0, -1)] + ["J1"],
        edge_type="main_road",
        is_production_edge=True,
    )
    add_bidirectional_edge(edge_map, nodes, params, "J1", "J3", edge_type="branch_road", is_production_edge=True)
    add_bidirectional_edge(edge_map, nodes, params, "J3", "J4", edge_type="access_road", is_production_edge=True)
    add_bidirectional_edge(edge_map, nodes, params, "J3", "J5", edge_type="branch_road", is_production_edge=True)
    connect_chain(edge_map, nodes, params, ["J6", "D_END", "D_END", "J5"], edge_type="line4_road", is_production_edge=True)
    add_bidirectional_edge(edge_map, nodes, params, "E_END", "J6", edge_type="buffer_link", is_production_edge=True)

    for idx in range(1, 14):
        add_bidirectional_edge(edge_map, nodes, params, f"L1_WS_{idx}", f"L1_BUF_{idx}", edge_type="buffer_link", is_production_edge=True)

    for idx in range(1, 6):
        add_bidirectional_edge(edge_map, nodes, params, f"L2_WS_{idx}", "J1", edge_type="service_link", is_production_edge=True)
        add_bidirectional_edge(edge_map, nodes, params, f"L2_WS_{idx}", f"L2_B_BUF_{idx}", edge_type="buffer_link", is_production_edge=True)
        add_bidirectional_edge(edge_map, nodes, params, f"L2_WS_{idx}", f"L2_S1_BUF_{idx}", edge_type="buffer_link", is_production_edge=True)

    for idx in range(1, 4):
        add_bidirectional_edge(edge_map, nodes, params, f"L3_WS_{idx}", "J1", edge_type="service_link", is_production_edge=True)
        add_bidirectional_edge(edge_map, nodes, params, f"L3_WS_{idx}", f"L3_C_BUF_{idx}", edge_type="buffer_link", is_production_edge=True)
        add_bidirectional_edge(edge_map, nodes, params, f"L3_WS_{idx}", f"L3_S2_BUF_{idx}", edge_type="buffer_link", is_production_edge=True)

    connect_chain(
        edge_map,
        nodes,
        params,
        [f"L4_WS_{idx}" for idx in range(1, 10)],
        edge_type="service_line",
        is_production_edge=True,
    )
    add_bidirectional_edge(edge_map, nodes, params, "L4_WS_1", "J5", edge_type="service_link", is_production_edge=True)
    for idx in range(1, 10):
        add_bidirectional_edge(edge_map, nodes, params, f"L4_WS_{idx}", f"L4_D_BUF_{idx}", edge_type="buffer_link", is_production_edge=True)
        add_bidirectional_edge(edge_map, nodes, params, f"L4_WS_{idx}", f"L4_P_BUF_{idx}", edge_type="buffer_link", is_production_edge=True)

    for node_id in ["A_END", "B_END", "C_END"]:
        add_bidirectional_edge(edge_map, nodes, params, node_id, "J10", edge_type="buffer_link")
    add_bidirectional_edge(edge_map, nodes, params, "D_END", "J6", edge_type="buffer_link", is_production_edge=True)

    add_bidirectional_edge(edge_map, nodes, params, "J4", "LANE_1_TOP", edge_type="lane_access")
    add_bidirectional_edge(edge_map, nodes, params, "J10", "LANE_1_BOTTOM", edge_type="lane_access")
    for idx in range(1, 4):
        add_bidirectional_edge(
            edge_map,
            nodes,
            params,
            f"LANE_{idx}_TOP",
            f"LANE_{idx}_BOTTOM",
            edge_type="lane",
            capacity=params.lane_capacity,
        )
    for idx in range(1, 3):
        add_bidirectional_edge(edge_map, nodes, params, f"LANE_{idx}_TOP", f"LANE_{idx + 1}_TOP", edge_type="lane_top_link")
        add_bidirectional_edge(edge_map, nodes, params, f"LANE_{idx}_BOTTOM", f"LANE_{idx + 1}_BOTTOM", edge_type="lane_bottom_link")

    connect_chain(edge_map, nodes, params, [f"C{idx}" for idx in range(1, 11)], edge_type="charger_link")
    add_bidirectional_edge(edge_map, nodes, params, "C1", "J7", edge_type="charger_access")
    add_bidirectional_edge(edge_map, nodes, params, "J7", "J4", edge_type="storage_to_road")

    for row in range(1, 11):
        connect_chain(edge_map, nodes, params, [f"S_{row}_{col}" for col in range(1, 13)], edge_type="storage_grid")
    for col in range(1, 13):
        connect_chain(edge_map, nodes, params, [f"S_{row}_{col}" for row in range(1, 11)], edge_type="storage_grid")
    add_bidirectional_edge(edge_map, nodes, params, "J8", "S_1_1", edge_type="storage_access")
    add_bidirectional_edge(edge_map, nodes, params, "J7", "S_10_1", edge_type="storage_access")

    for row in range(1, 10):
        connect_chain(edge_map, nodes, params, [f"F_{row}_{col}" for col in range(1, 13)], edge_type="finish_grid")
    for col in range(1, 13):
        connect_chain(edge_map, nodes, params, [f"F_{row}_{col}" for row in range(1, 10)], edge_type="finish_grid")
    connect_chain(edge_map, nodes, params, [f"D_{col}" for col in range(1, 13)], edge_type="defect_grid")
    for col in range(1, 13):
        add_bidirectional_edge(edge_map, nodes, params, f"F_9_{col}", f"D_{col}", edge_type="defect_link")
    add_bidirectional_edge(edge_map, nodes, params, "J9", "F_1_1", edge_type="finish_access")
    add_bidirectional_edge(edge_map, nodes, params, "J10", "D_1", edge_type="defect_access")
    add_bidirectional_edge(edge_map, nodes, params, "J10", "J9", edge_type="finish_to_road")

    edge_map.pop(("D_END", "D_END"), None)
    return edge_map


def build_graph(nodes: Dict[str, Node], edges: Dict[Tuple[str, str], Edge]) -> nx.DiGraph:
    graph = nx.DiGraph()
    for node in nodes.values():
        graph.add_node(
            node.node_id,
            x=node.x,
            y=node.y,
            node_type=node.node_type,
            region=node.region,
            service_capable=node.service_capable,
        )
    for edge in edges.values():
        graph.add_edge(
            edge.from_node,
            edge.to_node,
            distance=edge.distance,
            base_time=edge.base_time,
            energy_cost=edge.energy_cost,
            capacity=edge.capacity,
            edge_type=edge.edge_type,
            is_production_edge=edge.is_production_edge,
            resource_id=edge.resource_id,
        )
    return graph


def build_inventory_and_workstations(
    nodes: Dict[str, Node],
    params: GlobalParams,
    rng: random.Random,
) -> Tuple[
    Dict[str, InventoryBin],
    Dict[str, WorkstationConfig],
    Dict[str, List[str]],
    List[str],
    List[str],
    Dict[str, List[str]],
]:
    inventory_bins: Dict[str, InventoryBin] = {}
    workstations: Dict[str, WorkstationConfig] = {}
    supply_sources: Dict[str, List[str]] = {"A": [], "B": [], "C": [], "D": []}
    critical_buffers_by_line: Dict[str, List[str]] = {"L1": [], "L2": [], "L3": [], "L4": []}

    for row in range(1, 11):
        for col in range(1, 13):
            item_type = "A" if col <= 3 else "B" if col <= 6 else "C" if col <= 9 else "D"
            inventory_id = f"RAW_{row}_{col}"
            inventory_bins[inventory_id] = InventoryBin(
                inventory_id=inventory_id,
                node_id=f"S_{row}_{col}",
                item_type=item_type,
                role="raw_storage",
                capacity=120.0,
                initial_level=120.0,
                reorder_point=0.0,
                replenish_qty=0.0,
                safety_stock=0.0,
            )
            supply_sources[item_type].append(inventory_id)

    qualified_storage_bins: List[str] = []
    for row in range(1, 10):
        for col in range(1, 13):
            inventory_id = f"FG_{row}_{col}"
            inventory_bins[inventory_id] = InventoryBin(
                inventory_id=inventory_id,
                node_id=f"F_{row}_{col}",
                item_type="qualified",
                role="finished_storage",
                capacity=120.0,
                initial_level=0.0,
                reorder_point=0.0,
                replenish_qty=0.0,
                safety_stock=0.0,
            )
            qualified_storage_bins.append(inventory_id)

    defect_storage_bins: List[str] = []
    for col in range(1, 13):
        inventory_id = f"DF_{col}"
        inventory_bins[inventory_id] = InventoryBin(
            inventory_id=inventory_id,
            node_id=f"D_{col}",
            item_type="defect",
            role="defect_storage",
            capacity=120.0,
            initial_level=0.0,
            reorder_point=0.0,
            replenish_qty=0.0,
            safety_stock=0.0,
        )
        defect_storage_bins.append(inventory_id)

    def add_buffer(
        inventory_id: str,
        node_id: str,
        item_type: str,
        role: str,
        capacity: float,
        initial_level: float,
        line_id: str,
        workstation_id: Optional[str],
    ) -> None:
        inventory_bins[inventory_id] = InventoryBin(
            inventory_id=inventory_id,
            node_id=node_id,
            item_type=item_type,
            role=role,
            capacity=capacity,
            initial_level=initial_level,
            reorder_point=params.replenishment_trigger,
            replenish_qty=params.replenishment_qty,
            safety_stock=params.safety_stock,
            line_id=line_id,
            workstation_id=workstation_id,
        )

    for idx in range(1, 14):
        ws_id = f"L1_WS_{idx}"
        buf_id = f"INV_L1_A_{idx}"
        add_buffer(buf_id, f"L1_BUF_{idx}", "A", "station_input", 60.0, 60.0, "L1", ws_id)
        critical_buffers_by_line["L1"].append(buf_id)
        cycle_time = sample_cycle_time(2.0, params, rng)
        workstations[ws_id] = WorkstationConfig(
            workstation_id=ws_id,
            line_id="L1",
            node_id=f"L1_WS_{idx}",
            input_demands={buf_id: 1.0},
            output_flows={"INV_A_END": 1.0},
            cycle_time_s=cycle_time,
        )

    for idx in range(1, 6):
        ws_id = f"L2_WS_{idx}"
        b_buf = f"INV_L2_B_{idx}"
        s1_buf = f"INV_L2_S1_{idx}"
        add_buffer(b_buf, f"L2_B_BUF_{idx}", "B", "station_input", 80.0, 80.0, "L2", ws_id)
        add_buffer(s1_buf, f"L2_S1_BUF_{idx}", "semi1", "station_input", 120.0, 120.0, "L2", ws_id)
        critical_buffers_by_line["L2"].extend([b_buf, s1_buf])
        cycle_time = sample_cycle_time(1.0, params, rng)
        workstations[ws_id] = WorkstationConfig(
            workstation_id=ws_id,
            line_id="L2",
            node_id=f"L2_WS_{idx}",
            input_demands={b_buf: 1.0, s1_buf: 1.0},
            output_flows={"INV_B_END": 1.0},
            cycle_time_s=cycle_time,
        )

    for idx in range(1, 4):
        ws_id = f"L3_WS_{idx}"
        c_buf = f"INV_L3_C_{idx}"
        s2_buf = f"INV_L3_S2_{idx}"
        add_buffer(c_buf, f"L3_C_BUF_{idx}", "C", "station_input", 120.0, 120.0, "L3", ws_id)
        add_buffer(s2_buf, f"L3_S2_BUF_{idx}", "semi2", "station_input", 120.0, 120.0, "L3", ws_id)
        critical_buffers_by_line["L3"].extend([c_buf, s2_buf])
        cycle_time = sample_cycle_time(0.55, params, rng)
        workstations[ws_id] = WorkstationConfig(
            workstation_id=ws_id,
            line_id="L3",
            node_id=f"L3_WS_{idx}",
            input_demands={c_buf: 1.0, s2_buf: 1.0},
            output_flows={"INV_C_END": 1.0},
            cycle_time_s=cycle_time,
        )

    for idx in range(1, 10):
        ws_id = f"L4_WS_{idx}"
        d_buf = f"INV_L4_D_{idx}"
        p_buf = f"INV_L4_P_{idx}"
        add_buffer(d_buf, f"L4_D_BUF_{idx}", "D", "station_input", 120.0, 120.0, "L4", ws_id)
        add_buffer(p_buf, f"L4_P_BUF_{idx}", "product", "station_input", 120.0, 120.0, "L4", ws_id)
        critical_buffers_by_line["L4"].extend([d_buf, p_buf])
        cycle_time = sample_cycle_time(1.5, params, rng)
        workstations[ws_id] = WorkstationConfig(
            workstation_id=ws_id,
            line_id="L4",
            node_id=f"L4_WS_{idx}",
            input_demands={d_buf: 1.0, p_buf: 1.0},
            output_flows={
                "INV_D_END": params.qualified_rate,
                "INV_E_END": 1.0 - params.qualified_rate,
            },
            cycle_time_s=cycle_time,
        )

    for inventory_id, node_id, item_type, line_id in [
        ("INV_A_END", "A_END", "semi1", "L1"),
        ("INV_B_END", "B_END", "semi2", "L2"),
        ("INV_C_END", "C_END", "product", "L3"),
        ("INV_D_END", "D_END", "qualified", "L4"),
        ("INV_E_END", "E_END", "defect", "L4"),
    ]:
        inventory_bins[inventory_id] = InventoryBin(
            inventory_id=inventory_id,
            node_id=node_id,
            item_type=item_type,
            role="line_end",
            capacity=params.output_buffer_capacity,
            initial_level=0.0,
            reorder_point=params.output_buffer_capacity - params.replenishment_qty,
            replenish_qty=params.replenishment_qty,
            safety_stock=0.0,
            line_id=line_id,
            workstation_id=None,
        )

    return (
        inventory_bins,
        workstations,
        supply_sources,
        qualified_storage_bins,
        defect_storage_bins,
        critical_buffers_by_line,
    )


def build_agvs(params: GlobalParams, agv_count: int = 3) -> Dict[str, AGV]:
    start_nodes = ["C1", "J5", "J10", "C2", "J1", "J4", "C3", "J3", "J7", "J8"]
    agvs: Dict[str, AGV] = {}
    for index in range(max(1, agv_count)):
        agv_id = f"AGV_{index + 1}"
        start_node = start_nodes[index % len(start_nodes)]
        agvs[agv_id] = AGV(
            agv_id,
            start_node,
            params.battery_capacity_kwh,
            params.battery_capacity_kwh,
            params.battery_threshold_kwh,
            params.battery_min_kwh,
            params.charge_power_kw,
            100.0,
            "idle",
        )
    return agvs


def precompute_shortest_paths(instance: Instance) -> None:
    for source in instance.graph.nodes:
        lengths, paths = nx.single_source_dijkstra(instance.graph, source, weight="distance")
        for target, path in paths.items():
            instance.shortest_path_matrix[(source, target)] = path
            distance = lengths[target]
            base_time = 0.0
            energy = 0.0
            for left, right in zip(path, path[1:]):
                edge = instance.edges[(left, right)]
                base_time += edge.base_time
                energy += edge.energy_cost
            instance.shortest_metric_matrix[(source, target)] = (distance, base_time, energy)


def build_instance(
    seed: int = 42,
    *,
    agv_count: int = 3,
    target_qualified_units: float = 50.0,
) -> Instance:
    params = GlobalParams(
        random_seed=seed,
        target_qualified_units=target_qualified_units,
    )
    rng = random.Random(seed)
    nodes = build_nodes(params)
    edges = build_edges(nodes, params)
    graph = build_graph(nodes, edges)
    (
        inventory_bins,
        workstations,
        supply_sources,
        qualified_storage_bins,
        defect_storage_bins,
        critical_buffers_by_line,
    ) = build_inventory_and_workstations(nodes, params, rng)
    instance = Instance(
        global_params=params,
        nodes=nodes,
        edges=edges,
        graph=graph,
        agvs=build_agvs(params, agv_count=agv_count),
        chargers=[f"C{idx}" for idx in range(1, 11)],
        inventory_bins=inventory_bins,
        workstations=workstations,
        supply_sources=supply_sources,
        qualified_storage_bins=qualified_storage_bins,
        defect_storage_bins=defect_storage_bins,
        critical_buffers_by_line=critical_buffers_by_line,
    )
    precompute_shortest_paths(instance)
    return instance


def shortest_path_metrics(instance: Instance, source: str, target: str) -> Tuple[List[str], float, float, float]:
    path = instance.shortest_path_matrix[(source, target)]
    distance, base_time, energy = instance.shortest_metric_matrix[(source, target)]
    return path, distance, base_time, energy


def initial_inventory_levels(instance: Instance) -> Dict[str, float]:
    return {
        inventory_id: inventory.initial_level
        for inventory_id, inventory in instance.inventory_bins.items()
    }


def build_demo_planning_levels(instance: Instance) -> Dict[str, float]:
    levels = initial_inventory_levels(instance)
    levels["INV_A_END"] = 90.0
    levels["INV_B_END"] = 75.0
    levels["INV_C_END"] = 68.0
    levels["INV_D_END"] = 92.0
    levels["INV_E_END"] = 88.0

    for inventory_id, low_level in {
        "INV_L1_A_1": 8.0,
        "INV_L1_A_7": 14.0,
        "INV_L2_B_1": 10.0,
        "INV_L2_S1_1": 8.0,
        "INV_L3_C_1": 12.0,
        "INV_L3_S2_1": 9.0,
        "INV_L4_D_1": 18.0,
        "INV_L4_P_1": 16.0,
    }.items():
        levels[inventory_id] = low_level

    levels["RAW_1_1"] = 120.0
    levels["RAW_1_4"] = 120.0
    levels["RAW_1_7"] = 120.0
    levels["RAW_1_10"] = 120.0
    return levels


class TaskGenerator:
    def __init__(self, instance: Instance) -> None:
        self.instance = instance

    def _time_to_safety(self, inventory_id: str, current_level: float) -> float:
        inventory = self.instance.inventory_bins[inventory_id]
        workstation = self.instance.workstations.get(inventory.workstation_id or "")
        if workstation is None:
            return self.instance.global_params.simulation_horizon_s
        unit_rate = workstation.rate_per_second * workstation.input_demands.get(inventory_id, 0.0)
        if unit_rate <= 0:
            return self.instance.global_params.simulation_horizon_s
        return max(0.0, (current_level - inventory.safety_stock) / unit_rate)

    def _time_to_full(self, inventory_id: str, current_level: float) -> float:
        inbound_rate = 0.0
        for workstation in self.instance.workstations.values():
            inbound_rate += workstation.rate_per_second * workstation.output_flows.get(inventory_id, 0.0)
        inventory = self.instance.inventory_bins[inventory_id]
        if inbound_rate <= 0:
            return self.instance.global_params.simulation_horizon_s
        return max(0.0, (inventory.capacity - current_level) / inbound_rate)

    def _select_source_bin(self, item_type: str, available_levels: Dict[str, float]) -> Optional[str]:
        if item_type in self.instance.supply_sources:
            candidates = self.instance.supply_sources[item_type]
        elif item_type == "semi1":
            candidates = ["INV_A_END"]
        elif item_type == "semi2":
            candidates = ["INV_B_END"]
        elif item_type == "product":
            candidates = ["INV_C_END"]
        else:
            candidates = []
        candidates = [inventory_id for inventory_id in candidates if available_levels.get(inventory_id, 0.0) > 0.0]
        if not candidates:
            return None
        return max(candidates, key=lambda inventory_id: available_levels.get(inventory_id, 0.0))

    def _select_destination_bin(self, item_type: str, available_levels: Dict[str, float], qty: float) -> Optional[str]:
        if item_type == "qualified":
            candidates = self.instance.qualified_storage_bins
        elif item_type == "defect":
            candidates = self.instance.defect_storage_bins
        else:
            return None
        feasible = [
            inventory_id
            for inventory_id in candidates
            if self.instance.inventory_bins[inventory_id].capacity - available_levels.get(inventory_id, 0.0) >= qty
        ]
        if feasible:
            return feasible[0]
        return max(
            candidates,
            key=lambda inventory_id: self.instance.inventory_bins[inventory_id].capacity - available_levels.get(inventory_id, 0.0),
            default=None,
        )

    def generate_tasks(
        self,
        current_time: float,
        levels: Mapping[str, float],
        active_targets: Optional[set[str]] = None,
    ) -> Dict[str, Task]:
        active_targets = active_targets or set()
        reserved_levels = dict(levels)
        tasks: Dict[str, Task] = {}
        task_index = 1

        for inventory_id, inventory in self.instance.inventory_bins.items():
            current_level = reserved_levels.get(inventory_id, 0.0)

            if inventory.role == "station_input" and current_level < inventory.reorder_point and inventory_id not in active_targets:
                source_inventory_id = self._select_source_bin(inventory.item_type, reserved_levels)
                if source_inventory_id is None:
                    continue
                source_available = reserved_levels.get(source_inventory_id, 0.0)
                qty = min(inventory.replenish_qty, inventory.capacity - current_level, source_available)
                if qty <= 0:
                    continue

                deadline = current_time + self._time_to_safety(inventory_id, current_level)
                task_id = f"TASK_{task_index:03d}"
                task_index += 1
                tasks[task_id] = Task(
                    task_id=task_id,
                    task_type=self._task_type_for_item(inventory.item_type),
                    pickup_node=self.instance.inventory_bins[source_inventory_id].node_id,
                    drop_node=inventory.node_id,
                    qty=qty,
                    earliest_start=current_time,
                    latest_finish=deadline,
                    pickup_service_time=self.instance.global_params.default_pick_service_time_s,
                    drop_service_time=self.instance.global_params.default_drop_service_time_s,
                    source_inventory_id=source_inventory_id,
                    target_inventory_id=inventory_id,
                    generated_time=current_time,
                )
                reserved_levels[source_inventory_id] = source_available - qty
                reserved_levels[inventory_id] = current_level + qty

            if inventory.role == "line_end" and inventory.item_type in {"qualified", "defect"} and current_level >= inventory.reorder_point and inventory_id not in active_targets:
                qty = min(inventory.replenish_qty, current_level)
                destination_inventory_id = self._select_destination_bin(inventory.item_type, reserved_levels, qty)
                if destination_inventory_id is None or qty <= 0:
                    continue
                deadline = current_time + self._time_to_full(inventory_id, current_level)
                task_id = f"TASK_{task_index:03d}"
                task_index += 1
                task_type = "finished_putaway" if inventory.item_type == "qualified" else "defect_putaway"
                tasks[task_id] = Task(
                    task_id=task_id,
                    task_type=task_type,
                    pickup_node=inventory.node_id,
                    drop_node=self.instance.inventory_bins[destination_inventory_id].node_id,
                    qty=qty,
                    earliest_start=current_time,
                    latest_finish=deadline,
                    pickup_service_time=self.instance.global_params.default_pick_service_time_s,
                    drop_service_time=self.instance.global_params.default_drop_service_time_s,
                    source_inventory_id=inventory_id,
                    target_inventory_id=destination_inventory_id,
                    generated_time=current_time,
                )
                reserved_levels[inventory_id] = current_level - qty
                reserved_levels[destination_inventory_id] = reserved_levels.get(destination_inventory_id, 0.0) + qty

        return tasks

    @staticmethod
    def _task_type_for_item(item_type: str) -> str:
        if item_type in {"A", "B", "C", "D"}:
            return "material_replenishment"
        if item_type in {"semi1", "semi2", "product"}:
            return "semi_finished_transfer"
        return "generic_transfer"


def project_agv_state_after_task(
    agv: AGV,
    current_node: str,
    current_time: float,
    current_battery: float,
    task: Task,
    instance: Instance,
) -> Tuple[float, str, float]:
    travel_time = shortest_path_metrics(instance, current_node, task.pickup_node)[2]
    energy_to_pickup = shortest_path_metrics(instance, current_node, task.pickup_node)[3]
    task_travel_time = shortest_path_metrics(instance, task.pickup_node, task.drop_node)[2]
    task_energy = shortest_path_metrics(instance, task.pickup_node, task.drop_node)[3]
    reserve_energy = min(shortest_path_metrics(instance, task.drop_node, charger)[3] for charger in instance.chargers)

    projected_time = current_time
    projected_battery = current_battery
    if projected_battery < max(agv.battery_threshold, energy_to_pickup + task_energy + reserve_energy):
        charger = min(instance.chargers, key=lambda node_id: shortest_path_metrics(instance, current_node, node_id)[2])
        to_charger_time = shortest_path_metrics(instance, current_node, charger)[2]
        to_charger_energy = shortest_path_metrics(instance, current_node, charger)[3]
        projected_time += to_charger_time
        projected_battery -= to_charger_energy
        charge_need = max(0.0, agv.battery_max - projected_battery)
        projected_time += 3600.0 * charge_need / agv.charge_power if agv.charge_power > 0 else 0.0
        projected_battery = agv.battery_max
        current_node = charger
        travel_time = shortest_path_metrics(instance, current_node, task.pickup_node)[2]
        energy_to_pickup = shortest_path_metrics(instance, current_node, task.pickup_node)[3]

    projected_time += travel_time
    projected_time = max(projected_time, task.earliest_start)
    projected_time += task.pickup_service_time + task_travel_time + task.drop_service_time
    projected_battery -= energy_to_pickup + task_energy
    return projected_time, task.drop_node, projected_battery


def build_priority_solution(task_ids: Sequence[str], priorities: Sequence[float]) -> Dict[str, object]:
    return {"task_ids": list(task_ids), "priority": list(priorities)}


def greedy_assign_by_priority(
    solution: Mapping[str, object],
    tasks: Mapping[str, Task],
    instance: Instance,
) -> Dict[str, List[str]]:
    task_ids = list(solution["task_ids"])
    priorities = list(solution["priority"])
    ordered = sorted(zip(priorities, task_ids), key=lambda item: item[0])

    agv_state = {
        agv_id: {
            "time": 0.0,
            "node": agv.start_node,
            "battery": agv.battery_init,
            "tasks": [],
        }
        for agv_id, agv in instance.agvs.items()
    }

    for _, task_id in ordered:
        task = tasks[task_id]
        best_agv_id: Optional[str] = None
        best_finish = math.inf
        best_state: Optional[Tuple[float, str, float]] = None
        for agv_id, agv in instance.agvs.items():
            if task.qty > agv.load_capacity:
                continue
            projected = project_agv_state_after_task(
                agv=agv,
                current_node=agv_state[agv_id]["node"],
                current_time=float(agv_state[agv_id]["time"]),
                current_battery=float(agv_state[agv_id]["battery"]),
                task=task,
                instance=instance,
            )
            if projected[0] < best_finish:
                best_finish = projected[0]
                best_agv_id = agv_id
                best_state = projected
        if best_agv_id is None or best_state is None:
            continue
        agv_state[best_agv_id]["time"] = best_state[0]
        agv_state[best_agv_id]["node"] = best_state[1]
        agv_state[best_agv_id]["battery"] = best_state[2]
        agv_state[best_agv_id]["tasks"].append(task_id)

    return {
        agv_id: list(state["tasks"])
        for agv_id, state in agv_state.items()
    }


def build_round_robin_assignment(task_ids: Sequence[str], agv_ids: Sequence[str]) -> Dict[str, List[str]]:
    assignment = {agv_id: [] for agv_id in agv_ids}
    for index, task_id in enumerate(task_ids):
        assignment[agv_ids[index % len(agv_ids)]].append(task_id)
    return assignment


def decode_assignment_map(
    assignment_map: Mapping[str, Sequence[str]],
    tasks: Mapping[str, Task],
    instance: Instance,
) -> DecodedSolution:
    resource_availability: Dict[str, float] = {}
    charger_availability: Dict[str, float] = {f"charger:{charger_id}": 0.0 for charger_id in instance.chargers}
    agv_schedules: Dict[str, AGVDecodedSchedule] = {}

    for agv_id in sorted(instance.agvs):
        agv_schedules[agv_id] = decode_agv_schedule(
            agv=instance.agvs[agv_id],
            task_ids=list(assignment_map.get(agv_id, [])),
            tasks=tasks,
            instance=instance,
            resource_availability=resource_availability,
            charger_availability=charger_availability,
        )

    task_executions = {
        execution.task_id: execution
        for schedule in agv_schedules.values()
        for execution in schedule.task_executions
    }
    return DecodedSolution(
        agv_schedules=agv_schedules,
        task_executions=task_executions,
        assignment_map={agv_id: list(task_ids) for agv_id, task_ids in assignment_map.items()},
    )


def decode_solution(
    solution: Mapping[str, object],
    tasks: Mapping[str, Task],
    instance: Instance,
) -> DecodedSolution:
    assignment_map = greedy_assign_by_priority(solution, tasks, instance)
    return decode_assignment_map(assignment_map, tasks, instance)


def decode_agv_schedule(
    *,
    agv: AGV,
    task_ids: Sequence[str],
    tasks: Mapping[str, Task],
    instance: Instance,
    resource_availability: Dict[str, float],
    charger_availability: Dict[str, float],
) -> AGVDecodedSchedule:
    current_node = agv.start_node
    current_time = 0.0
    current_battery = agv.battery_init
    visited_nodes: List[str] = [current_node]
    traversals: List[EdgeTraversal] = []
    charging_events: List[ChargingEvent] = []
    task_executions: List[TaskExecution] = []
    battery_trace: List[BatteryRecord] = [BatteryRecord(0.0, current_battery, "initial")]

    for task_id in task_ids:
        task = tasks[task_id]
        current_node, current_time, current_battery, pre_traversals, charge_event = ensure_charge_for_task(
            agv=agv,
            task=task,
            current_node=current_node,
            current_time=current_time,
            current_battery=current_battery,
            instance=instance,
            resource_availability=resource_availability,
            charger_availability=charger_availability,
        )
        traversals.extend(pre_traversals)
        extend_visited_nodes(visited_nodes, pre_traversals)
        if charge_event is not None:
            charging_events.append(charge_event)
            battery_trace.append(BatteryRecord(charge_event.end_time, current_battery, f"charged_at_{charge_event.charger_id}"))

        pickup_arrival, current_battery, pickup_traversals = travel_and_consume(
            agv_id=agv.agv_id,
            source=current_node,
            target=task.pickup_node,
            start_time=current_time,
            battery=current_battery,
            instance=instance,
            resource_availability=resource_availability,
        )
        traversals.extend(pickup_traversals)
        extend_visited_nodes(visited_nodes, pickup_traversals)
        pickup_start = max(pickup_arrival, task.earliest_start)
        pickup_end = pickup_start + task.pickup_service_time

        drop_arrival, current_battery, drop_traversals = travel_and_consume(
            agv_id=agv.agv_id,
            source=task.pickup_node,
            target=task.drop_node,
            start_time=pickup_end,
            battery=current_battery,
            instance=instance,
            resource_availability=resource_availability,
        )
        traversals.extend(drop_traversals)
        extend_visited_nodes(visited_nodes, drop_traversals)
        drop_start = drop_arrival
        drop_end = drop_start + task.drop_service_time

        current_node = task.drop_node
        current_time = drop_end
        task_executions.append(
            TaskExecution(
                task_id=task.task_id,
                agv_id=agv.agv_id,
                pickup_arrival=pickup_arrival,
                pickup_start=pickup_start,
                pickup_end=pickup_end,
                drop_arrival=drop_arrival,
                drop_start=drop_start,
                drop_end=drop_end,
                late_amount=max(0.0, drop_end - task.latest_finish),
            )
        )
        battery_trace.append(BatteryRecord(current_time, current_battery, f"after_{task.task_id}"))

    return AGVDecodedSchedule(
        agv_id=agv.agv_id,
        assigned_tasks=list(task_ids),
        visited_nodes=visited_nodes,
        traversals=traversals,
        charging_events=charging_events,
        task_executions=task_executions,
        battery_trace=battery_trace,
        total_distance=sum(item.distance for item in traversals),
        total_energy=sum(item.energy_cost for item in traversals),
        total_wait=sum(item.wait_before for item in traversals),
        final_node=current_node,
        final_time=current_time,
        final_battery=current_battery,
    )


def ensure_charge_for_task(
    *,
    agv: AGV,
    task: Task,
    current_node: str,
    current_time: float,
    current_battery: float,
    instance: Instance,
    resource_availability: Dict[str, float],
    charger_availability: Dict[str, float],
) -> Tuple[str, float, float, List[EdgeTraversal], Optional[ChargingEvent]]:
    required_energy = required_energy_with_reserve(current_node, task, instance)
    if current_battery >= max(agv.battery_threshold, required_energy):
        return current_node, current_time, current_battery, [], None

    charger_id = min(instance.chargers, key=lambda candidate: shortest_path_metrics(instance, current_node, candidate)[2])
    charge_arrival, battery_after_travel, traversals = travel_and_consume(
        agv_id=agv.agv_id,
        source=current_node,
        target=charger_id,
        start_time=current_time,
        battery=current_battery,
        instance=instance,
        resource_availability=resource_availability,
    )
    charge_resource = f"charger:{charger_id}"
    available_time = charger_availability.get(charge_resource, 0.0)
    charge_start = max(charge_arrival, available_time)
    charge_need = max(0.0, agv.battery_max - battery_after_travel)
    charge_duration = 3600.0 * charge_need / agv.charge_power if agv.charge_power > 0 else 0.0
    charge_end = charge_start + charge_duration
    charger_availability[charge_resource] = charge_end
    return (
        charger_id,
        charge_end,
        agv.battery_max,
        traversals,
        ChargingEvent(charger_id, charge_start, charge_end, battery_after_travel, agv.battery_max),
    )


def required_energy_with_reserve(current_node: str, task: Task, instance: Instance) -> float:
    energy_to_pickup = shortest_path_metrics(instance, current_node, task.pickup_node)[3]
    energy_to_drop = shortest_path_metrics(instance, task.pickup_node, task.drop_node)[3]
    reserve_energy = min(shortest_path_metrics(instance, task.drop_node, charger_id)[3] for charger_id in instance.chargers)
    return energy_to_pickup + energy_to_drop + reserve_energy


def travel_and_consume(
    *,
    agv_id: str,
    source: str,
    target: str,
    start_time: float,
    battery: float,
    instance: Instance,
    resource_availability: Dict[str, float],
) -> Tuple[float, float, List[EdgeTraversal]]:
    if source == target:
        return start_time, battery, []

    path, _, _, _ = shortest_path_metrics(instance, source, target)
    traversals: List[EdgeTraversal] = []
    current_time = start_time
    current_battery = battery
    for from_node, to_node in zip(path, path[1:]):
        edge = instance.edges[(from_node, to_node)]
        wait_before = max(0.0, resource_availability.get(edge.resource_id, 0.0) - current_time)
        traversal_start = current_time + wait_before
        duration = edge.base_time + (instance.global_params.expected_worker_delay_s if edge.is_production_edge else 0.0)
        traversal_end = traversal_start + duration
        resource_availability[edge.resource_id] = traversal_end
        traversals.append(
            EdgeTraversal(
                agv_id=agv_id,
                from_node=from_node,
                to_node=to_node,
                resource_id=edge.resource_id,
                start_time=traversal_start,
                end_time=traversal_end,
                duration=duration,
                distance=edge.distance,
                energy_cost=edge.energy_cost,
                wait_before=wait_before,
                edge_type=edge.edge_type,
            )
        )
        current_time = traversal_end
        current_battery -= edge.energy_cost
    return current_time, current_battery, traversals


def extend_visited_nodes(visited_nodes: List[str], traversals: Sequence[EdgeTraversal]) -> None:
    for traversal in traversals:
        if not visited_nodes or visited_nodes[-1] != traversal.to_node:
            visited_nodes.append(traversal.to_node)


def build_inventory_events(decoded: DecodedSolution, tasks: Mapping[str, Task]) -> List[Tuple[float, str, str, float]]:
    events: List[Tuple[float, str, str, float]] = []
    for task_id, execution in decoded.task_executions.items():
        task = tasks[task_id]
        if task.source_inventory_id is not None:
            events.append((execution.pickup_end, "decrement", task.source_inventory_id, task.qty))
        if task.target_inventory_id is not None:
            events.append((execution.drop_end, "increment", task.target_inventory_id, task.qty))
    events.sort(key=lambda item: item[0])
    return events


def simulate_inventory(
    instance: Instance,
    initial_levels_map: Mapping[str, float],
    decoded: DecodedSolution,
    tasks: Mapping[str, Task],
    horizon: float,
) -> InventorySimulationResult:
    step = instance.global_params.simulation_step_s
    levels = dict(initial_levels_map)
    events = build_inventory_events(decoded, tasks)
    event_index = 0
    snapshots: List[InventorySnapshot] = []
    stop_by_line = {line_id: 0.0 for line_id in instance.critical_buffers_by_line}
    capacity_penalty = 0.0
    tracked_inventory = [inventory_id for inventory_id, bin_info in instance.inventory_bins.items() if bin_info.role in {"station_input", "line_end"}]
    inventory_accumulator = 0.0

    current_time = 0.0
    while current_time <= horizon + 1e-9:
        while event_index < len(events) and events[event_index][0] <= current_time + 1e-9:
            _, action, inventory_id, qty = events[event_index]
            capacity = instance.inventory_bins[inventory_id].capacity
            if action == "decrement":
                levels[inventory_id] = max(0.0, levels.get(inventory_id, 0.0) - qty)
            else:
                levels[inventory_id] = levels.get(inventory_id, 0.0) + qty
                if levels[inventory_id] > capacity:
                    capacity_penalty += levels[inventory_id] - capacity
                    levels[inventory_id] = capacity
            event_index += 1

        if current_time >= horizon:
            snapshots.append(
                InventorySnapshot(
                    time=current_time,
                    levels={inventory_id: levels[inventory_id] for inventory_id in tracked_inventory},
                    line_stop_ratio={line_id: 0.0 for line_id in stop_by_line},
                )
            )
            break

        dt = min(step, horizon - current_time)
        line_stop_ratio = {line_id: 0.0 for line_id in stop_by_line}

        for workstation in instance.workstations.values():
            planned_units = workstation.rate_per_second * dt
            feasible_units = planned_units
            for inventory_id, demand_per_unit in workstation.input_demands.items():
                if demand_per_unit <= 0:
                    continue
                feasible_units = min(feasible_units, levels.get(inventory_id, 0.0) / demand_per_unit)
            for inventory_id, output_ratio in workstation.output_flows.items():
                if output_ratio <= 0:
                    continue
                output_bin = instance.inventory_bins[inventory_id]
                free_capacity = output_bin.capacity - levels.get(inventory_id, 0.0)
                feasible_units = min(feasible_units, free_capacity / output_ratio)
            feasible_units = max(0.0, feasible_units)

            if planned_units > 1e-9 and feasible_units < planned_units:
                shortage_ratio = 1.0 - feasible_units / planned_units
                line_stop_ratio[workstation.line_id] = max(line_stop_ratio[workstation.line_id], shortage_ratio)

            for inventory_id, demand_per_unit in workstation.input_demands.items():
                levels[inventory_id] = max(0.0, levels.get(inventory_id, 0.0) - feasible_units * demand_per_unit)
            for inventory_id, output_ratio in workstation.output_flows.items():
                output_bin = instance.inventory_bins[inventory_id]
                levels[inventory_id] = clamp(
                    levels.get(inventory_id, 0.0) + feasible_units * output_ratio,
                    0.0,
                    output_bin.capacity,
                )

        for line_id, ratio in line_stop_ratio.items():
            stop_by_line[line_id] += ratio * dt

        inventory_accumulator += sum(levels[inventory_id] for inventory_id in tracked_inventory)
        snapshots.append(
            InventorySnapshot(
                time=current_time,
                levels={inventory_id: levels[inventory_id] for inventory_id in tracked_inventory},
                line_stop_ratio=dict(line_stop_ratio),
            )
        )
        current_time += dt

    average_inventory = inventory_accumulator / max(1, len(snapshots) * len(tracked_inventory))
    qualified_total = sum(
        levels[inventory_id]
        for inventory_id, bin_info in instance.inventory_bins.items()
        if bin_info.item_type == "qualified"
    )
    defect_total = sum(
        levels[inventory_id]
        for inventory_id, bin_info in instance.inventory_bins.items()
        if bin_info.item_type == "defect"
    )
    return InventorySimulationResult(
        horizon=horizon,
        snapshots=snapshots,
        final_levels=levels,
        line_stop_time=sum(stop_by_line.values()),
        average_inventory=average_inventory,
        stop_by_line=stop_by_line,
        qualified_output_total=qualified_total,
        defect_output_total=defect_total,
        capacity_penalty=capacity_penalty,
    )


def compute_load_penalty(
    instance: Instance,
    assignment_map: Mapping[str, Sequence[str]],
    tasks: Mapping[str, Task],
) -> float:
    penalty = 0.0
    for agv_id, task_ids in assignment_map.items():
        agv = instance.agvs[agv_id]
        for task_id in task_ids:
            penalty += max(0.0, tasks[task_id].qty - agv.load_capacity)
    return penalty


def compute_battery_penalty(instance: Instance, decoded: DecodedSolution) -> float:
    penalty = 0.0
    for agv_id, schedule in decoded.agv_schedules.items():
        agv = instance.agvs[agv_id]
        min_battery = min((record.battery for record in schedule.battery_trace), default=agv.battery_init)
        penalty += max(0.0, agv.battery_min - min_battery)
    return penalty


def evaluate_decoded_solution(
    decoded: DecodedSolution,
    tasks: Mapping[str, Task],
    instance: Instance,
    initial_levels_map: Mapping[str, float],
    *,
    horizon: Optional[float] = None,
    weights: Optional[FitnessWeights] = None,
) -> Evaluation:
    weights = weights or FitnessWeights()
    horizon = horizon or instance.global_params.simulation_horizon_s
    makespan = max((schedule.final_time for schedule in decoded.agv_schedules.values()), default=0.0)
    total_distance = sum(schedule.total_distance for schedule in decoded.agv_schedules.values())
    total_energy = sum(schedule.total_energy for schedule in decoded.agv_schedules.values())
    congestion_wait = sum(schedule.total_wait for schedule in decoded.agv_schedules.values())
    charge_count = sum(len(schedule.charging_events) for schedule in decoded.agv_schedules.values())
    late_penalty = sum(execution.late_amount for execution in decoded.task_executions.values())
    late_task_count = sum(1 for execution in decoded.task_executions.values() if execution.late_amount > 0)
    battery_penalty = compute_battery_penalty(instance, decoded)
    load_penalty = compute_load_penalty(instance, decoded.assignment_map, tasks)
    inventory_result = simulate_inventory(instance, initial_levels_map, decoded, tasks, horizon)
    stop_penalty = inventory_result.line_stop_time * instance.global_params.line_stop_penalty_weight
    target_shortfall_penalty = max(
        0.0,
        instance.global_params.target_qualified_units - inventory_result.qualified_output_total,
    ) * instance.global_params.target_completion_penalty_weight

    fitness = (
        weights.makespan * makespan
        + weights.total_distance * total_distance
        + weights.total_energy * total_energy
        + weights.congestion_wait * congestion_wait
        + weights.line_stop_time * inventory_result.line_stop_time
        + weights.late_penalty * late_penalty
        + weights.battery_penalty * battery_penalty
        + weights.capacity_penalty * inventory_result.capacity_penalty
        + weights.load_penalty * load_penalty
        + weights.inventory_holding * inventory_result.average_inventory
        + target_shortfall_penalty
    )
    return Evaluation(
        fitness=fitness,
        makespan=makespan,
        total_distance=total_distance,
        total_energy=total_energy,
        late_penalty=late_penalty,
        late_task_count=late_task_count,
        charge_count=charge_count,
        congestion_wait=congestion_wait,
        battery_penalty=battery_penalty,
        capacity_penalty=inventory_result.capacity_penalty,
        load_penalty=load_penalty,
        line_stop_time=inventory_result.line_stop_time,
        average_inventory=inventory_result.average_inventory,
        stop_penalty=stop_penalty,
        target_shortfall_penalty=target_shortfall_penalty,
        inventory_result=inventory_result,
        agv_schedules=decoded.agv_schedules,
    )


def evaluate_priority_solution(
    solution: Mapping[str, object],
    tasks: Mapping[str, Task],
    instance: Instance,
    initial_levels_map: Mapping[str, float],
    *,
    horizon: Optional[float] = None,
    weights: Optional[FitnessWeights] = None,
) -> Evaluation:
    decoded = decode_solution(solution, tasks, instance)
    return evaluate_decoded_solution(decoded, tasks, instance, initial_levels_map, horizon=horizon, weights=weights)


def build_random_solution(task_ids: Sequence[str], rng: random.Random) -> Dict[str, object]:
    return build_priority_solution(task_ids, [rng.random() for _ in task_ids])


def build_heuristic_solution(tasks: Mapping[str, Task], rng: random.Random) -> Dict[str, object]:
    task_ids = list(tasks.keys())
    slack_pairs = []
    for task_id in task_ids:
        task = tasks[task_id]
        slack = max(0.0, task.latest_finish - task.earliest_start)
        priority = slack + rng.uniform(0.0, 1.0)
        slack_pairs.append((task_id, priority))
    return build_priority_solution(task_ids, [priority for _, priority in slack_pairs])


def evaluate_individual(
    individual: GAIndividual,
    tasks: Mapping[str, Task],
    instance: Instance,
    initial_levels_map: Mapping[str, float],
) -> GAIndividual:
    individual.evaluation = evaluate_priority_solution(individual.solution, tasks, instance, initial_levels_map)
    return individual


def initialize_population(
    tasks: Mapping[str, Task],
    instance: Instance,
    initial_levels_map: Mapping[str, float],
    config: GAConfig,
    rng: random.Random,
) -> List[GAIndividual]:
    population: List[GAIndividual] = []
    task_ids = list(tasks.keys())
    for _ in range(min(config.heuristic_seed_count, config.population_size)):
        population.append(evaluate_individual(GAIndividual(build_heuristic_solution(tasks, rng)), tasks, instance, initial_levels_map))
    while len(population) < config.population_size:
        population.append(evaluate_individual(GAIndividual(build_random_solution(task_ids, rng)), tasks, instance, initial_levels_map))
    return population


def tournament_select(population: Sequence[GAIndividual], tournament_size: int, rng: random.Random) -> GAIndividual:
    candidates = rng.sample(list(population), k=min(len(population), tournament_size))
    return min(candidates, key=lambda individual: individual.fitness)


def crossover(parent_a: GAIndividual, parent_b: GAIndividual, rng: random.Random) -> Tuple[GAIndividual, GAIndividual]:
    task_ids = list(parent_a.solution["task_ids"])
    priority_a = list(parent_a.solution["priority"])
    priority_b = list(parent_b.solution["priority"])
    child_a: List[float] = []
    child_b: List[float] = []
    for index in range(len(task_ids)):
        if rng.random() < 0.5:
            child_a.append(priority_a[index])
            child_b.append(priority_b[index])
        else:
            child_a.append(priority_b[index])
            child_b.append(priority_a[index])
    return GAIndividual(build_priority_solution(task_ids, child_a)), GAIndividual(build_priority_solution(task_ids, child_b))


def mutate(individual: GAIndividual, rng: random.Random, mutation_rate: float) -> None:
    priorities = list(individual.solution["priority"])
    for index in range(len(priorities)):
        if rng.random() < mutation_rate:
            priorities[index] = clamp(priorities[index] + rng.uniform(-0.3, 0.3), 0.0, 1.5)
    individual.solution = build_priority_solution(individual.solution["task_ids"], priorities)
    individual.evaluation = None


def clone_individual(individual: GAIndividual) -> GAIndividual:
    return GAIndividual(
        solution=build_priority_solution(individual.solution["task_ids"], individual.solution["priority"]),
        evaluation=individual.evaluation,
    )


def solve_ga(
    tasks: Mapping[str, Task],
    instance: Instance,
    initial_levels_map: Mapping[str, float],
    *,
    config: Optional[GAConfig] = None,
    seed: int = 42,
) -> GAResult:
    config = config or GAConfig()
    rng = random.Random(seed)
    population = initialize_population(tasks, instance, initial_levels_map, config, rng)
    elite_count = max(1, int(round(config.population_size * config.elite_ratio)))
    history: List[GAGenerationStats] = []
    best = min(population, key=lambda individual: individual.fitness)

    for generation in range(config.generations):
        population.sort(key=lambda individual: individual.fitness)
        generation_best = population[0]
        history.append(
            GAGenerationStats(
                generation=generation,
                best_fitness=generation_best.fitness,
                average_fitness=sum(ind.fitness for ind in population) / len(population),
                best_makespan=generation_best.evaluation.makespan,
                best_line_stop=generation_best.evaluation.line_stop_time,
            )
        )
        if generation_best.fitness < best.fitness:
            best = clone_individual(generation_best)

        next_population = [clone_individual(ind) for ind in population[:elite_count]]
        while len(next_population) < config.population_size:
            parent_a = tournament_select(population, config.tournament_size, rng)
            parent_b = tournament_select(population, config.tournament_size, rng)
            if rng.random() < config.crossover_rate:
                child_a, child_b = crossover(parent_a, parent_b, rng)
            else:
                child_a, child_b = clone_individual(parent_a), clone_individual(parent_b)
            mutate(child_a, rng, config.mutation_rate)
            evaluate_individual(child_a, tasks, instance, initial_levels_map)
            next_population.append(child_a)
            if len(next_population) < config.population_size:
                mutate(child_b, rng, config.mutation_rate)
                evaluate_individual(child_b, tasks, instance, initial_levels_map)
                next_population.append(child_b)
        population = next_population

    population.sort(key=lambda individual: individual.fitness)
    if population[0].fitness < best.fitness:
        best = clone_individual(population[0])
    return GAResult(best_individual=best, history=history)


def save_convergence_plot(history: Sequence[GAGenerationStats], output_path: str = "ga_convergence.png") -> Optional[str]:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return None

    x = [item.generation for item in history]
    y = [item.best_fitness for item in history]
    plt.figure(figsize=(8, 4))
    plt.plot(x, y, linewidth=2)
    plt.xlabel("Generation")
    plt.ylabel("Best Fitness")
    plt.title("GA Convergence")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    return output_path


def build_history_figure(history: Sequence[GAGenerationStats]):
    try:
        from matplotlib.figure import Figure
    except Exception:
        return None

    figure = Figure(figsize=(8, 6), dpi=100)
    fitness_ax = figure.add_subplot(211)
    makespan_ax = figure.add_subplot(212)

    generations = [item.generation for item in history]
    best_fitness = [item.best_fitness for item in history]
    avg_fitness = [item.average_fitness for item in history]
    best_makespan = [item.best_makespan for item in history]

    fitness_ax.plot(generations, best_fitness, label="best_fitness", linewidth=2)
    fitness_ax.plot(generations, avg_fitness, label="avg_fitness", linewidth=1.5, linestyle="--")
    fitness_ax.set_title("适应度随代数变化")
    fitness_ax.set_xlabel("代数")
    fitness_ax.set_ylabel("适应度")
    fitness_ax.grid(True, alpha=0.3)
    fitness_ax.legend()

    makespan_ax.plot(generations, best_makespan, color="tab:orange", linewidth=2)
    makespan_ax.set_title("每代完成任务时间")
    makespan_ax.set_xlabel("代数")
    makespan_ax.set_ylabel("时间 / s")
    makespan_ax.grid(True, alpha=0.3)

    figure.tight_layout()
    return figure


def print_task_pool(tasks: Mapping[str, Task]) -> None:
    print("\n=== 当前任务池 ===")
    for task in tasks.values():
        print(
            task.task_id,
            {
                "type": task.task_type,
                "pickup": task.pickup_node,
                "drop": task.drop_node,
                "qty": task.qty,
                "latest_finish": round(task.latest_finish, 2),
                "source_inventory": task.source_inventory_id,
                "target_inventory": task.target_inventory_id,
            },
        )


def print_schedule_summary(decoded: DecodedSolution) -> None:
    print("\n=== AGV 调度方案 ===")
    for agv_id, schedule in decoded.agv_schedules.items():
        print(f"\n--- {agv_id} ---")
        print("任务序列:", schedule.assigned_tasks)
        print("路径节点:", schedule.visited_nodes)
        print("最终位置:", schedule.final_node)
        print("最终时间:", round(schedule.final_time, 2))
        print("最终电量:", round(schedule.final_battery, 4))
        print("充电次数:", len(schedule.charging_events))
        for execution in schedule.task_executions:
            print(
                "任务执行:",
                execution.task_id,
                {
                    "pickup_start": round(execution.pickup_start, 2),
                    "drop_end": round(execution.drop_end, 2),
                    "late": round(execution.late_amount, 2),
                },
            )


def print_inventory_summary(evaluation: Evaluation, target_qualified_units: float) -> None:
    result = evaluation.inventory_result
    print("\n=== 库存/停线结果 ===")
    pprint(
        {
            "line_stop_time": round(result.line_stop_time, 2),
            "stop_by_line": {key: round(value, 2) for key, value in result.stop_by_line.items()},
            "average_inventory": round(result.average_inventory, 2),
            "qualified_total": round(result.qualified_output_total, 2),
            "defect_total": round(result.defect_output_total, 2),
            "capacity_penalty": round(result.capacity_penalty, 4),
            "target_qualified_reached": result.qualified_output_total >= target_qualified_units,
        }
    )


def print_evaluation(title: str, evaluation: Evaluation) -> None:
    print(f"\n=== {title} ===")
    pprint(
        {
            "fitness": round(evaluation.fitness, 2),
            "makespan": round(evaluation.makespan, 2),
            "total_distance": round(evaluation.total_distance, 2),
            "total_energy": round(evaluation.total_energy, 4),
            "late_penalty": round(evaluation.late_penalty, 2),
            "late_task_count": evaluation.late_task_count,
            "charge_count": evaluation.charge_count,
            "congestion_wait": round(evaluation.congestion_wait, 2),
            "battery_penalty": round(evaluation.battery_penalty, 4),
            "capacity_penalty": round(evaluation.capacity_penalty, 4),
            "load_penalty": round(evaluation.load_penalty, 2),
            "line_stop_time": round(evaluation.line_stop_time, 2),
            "average_inventory": round(evaluation.average_inventory, 2),
            "target_shortfall_penalty": round(evaluation.target_shortfall_penalty, 2),
        }
    )


def run_experiment(
    *,
    generations: int = 40,
    target_qualified_units: float = 50.0,
    agv_count: int = 3,
    population_size: int = 40,
    seed: int = 42,
) -> ExperimentRunResult:
    instance = build_instance(
        seed=seed,
        agv_count=agv_count,
        target_qualified_units=target_qualified_units,
    )
    planning_levels = build_demo_planning_levels(instance)
    task_generator = TaskGenerator(instance)
    task_pool = task_generator.generate_tasks(current_time=0.0, levels=planning_levels)
    if not task_pool:
        raise ValueError("当前库存状态下未触发任务。")

    manual_assignment = build_round_robin_assignment(list(task_pool.keys()), sorted(instance.agvs.keys()))
    decoded_manual = decode_assignment_map(manual_assignment, task_pool, instance)
    manual_evaluation = evaluate_decoded_solution(decoded_manual, task_pool, instance, planning_levels)

    random_solution = build_random_solution(list(task_pool.keys()), random.Random(7))
    random_evaluation = evaluate_priority_solution(random_solution, task_pool, instance, planning_levels)

    ga_result = solve_ga(
        task_pool,
        instance,
        planning_levels,
        config=GAConfig(
            population_size=population_size,
            generations=generations,
            heuristic_seed_count=min(8, population_size),
        ),
        seed=seed,
    )
    best_solution = ga_result.best_individual.solution
    best_evaluation = ga_result.best_individual.evaluation
    best_decoded = decode_solution(best_solution, task_pool, instance)

    return ExperimentRunResult(
        instance=instance,
        task_pool=task_pool,
        planning_levels=planning_levels,
        manual_decoded=decoded_manual,
        manual_evaluation=manual_evaluation,
        random_solution=random_solution,
        random_evaluation=random_evaluation,
        best_solution=best_solution,
        best_decoded=best_decoded,
        best_evaluation=best_evaluation,
        ga_result=ga_result,
    )


def print_generation_history(history: Sequence[GAGenerationStats]) -> None:
    print("\n=== 每代结果 ===")
    print(f"{'gen':>4} {'best_fitness':>14} {'avg_fitness':>14} {'best_makespan':>14}")
    for item in history:
        print(
            f"{item.generation:>4} "
            f"{item.best_fitness:>14.2f} "
            f"{item.average_fitness:>14.2f} "
            f"{item.best_makespan:>14.2f}"
        )


def run_console_session() -> None:
    def _read_int(prompt: str, default: int) -> int:
        raw = input(f"{prompt} [{default}]: ").strip()
        return int(raw) if raw else default

    def _read_float(prompt: str, default: float) -> float:
        raw = input(f"{prompt} [{default}]: ").strip()
        return float(raw) if raw else default

    generations = _read_int("请输入GA迭代次数", 40)
    target_qualified_units = _read_float("请输入目标完成合格品数量", 50.0)
    agv_count = _read_int("请输入AGV数量", 3)

    result = run_experiment(
        generations=generations,
        target_qualified_units=target_qualified_units,
        agv_count=agv_count,
        population_size=max(20, min(100, agv_count * 15)),
        seed=42,
    )

    print_task_pool(result.task_pool)
    print_evaluation("人工轮转调度评价", result.manual_evaluation)
    print_schedule_summary(result.manual_decoded)
    print_inventory_summary(result.manual_evaluation, result.instance.global_params.target_qualified_units)
    print_evaluation("随机优先级基线", result.random_evaluation)
    print_evaluation("GA 最优结果", result.best_evaluation)
    print_schedule_summary(result.best_decoded)
    print_inventory_summary(result.best_evaluation, result.instance.global_params.target_qualified_units)
    print_generation_history(result.ga_result.history)

    plot_path = save_convergence_plot(result.ga_result.history)
    if plot_path is not None and Path(plot_path).exists():
        print(f"\n收敛曲线已保存: {plot_path}")


def launch_gui() -> None:
    if tk is None or ttk is None or ScrolledText is None:
        raise RuntimeError("当前环境不支持 tkinter 图形界面。")

    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

    root = tk.Tk()
    root.title("AGV 调度实验交互界面")
    root.geometry("1400x900")

    control_frame = ttk.LabelFrame(root, text="实验参数")
    control_frame.pack(fill="x", padx=10, pady=10)

    ttk.Label(control_frame, text="GA迭代次数").grid(row=0, column=0, padx=6, pady=6, sticky="w")
    generations_var = tk.StringVar(value="40")
    ttk.Entry(control_frame, textvariable=generations_var, width=12).grid(row=0, column=1, padx=6, pady=6)

    ttk.Label(control_frame, text="目标合格品数量").grid(row=0, column=2, padx=6, pady=6, sticky="w")
    target_var = tk.StringVar(value="50")
    ttk.Entry(control_frame, textvariable=target_var, width=12).grid(row=0, column=3, padx=6, pady=6)

    ttk.Label(control_frame, text="AGV数量").grid(row=0, column=4, padx=6, pady=6, sticky="w")
    agv_var = tk.StringVar(value="3")
    ttk.Entry(control_frame, textvariable=agv_var, width=12).grid(row=0, column=5, padx=6, pady=6)

    ttk.Label(control_frame, text="种群规模").grid(row=0, column=6, padx=6, pady=6, sticky="w")
    pop_var = tk.StringVar(value="40")
    ttk.Entry(control_frame, textvariable=pop_var, width=12).grid(row=0, column=7, padx=6, pady=6)

    notebook = ttk.Notebook(root)
    notebook.pack(fill="both", expand=True, padx=10, pady=10)

    summary_tab = ttk.Frame(notebook)
    history_tab = ttk.Frame(notebook)
    notebook.add(summary_tab, text="结果摘要")
    notebook.add(history_tab, text="迭代过程")

    summary_text = ScrolledText(summary_tab, wrap="word", font=("Courier New", 10))
    summary_text.pack(fill="both", expand=True)

    history_frame = ttk.Frame(history_tab)
    history_frame.pack(fill="both", expand=True)

    columns = ("generation", "best_fitness", "avg_fitness", "best_makespan", "best_line_stop")
    history_tree = ttk.Treeview(history_frame, columns=columns, show="headings", height=18)
    for column, title, width in [
        ("generation", "代数", 80),
        ("best_fitness", "最优适应度", 140),
        ("avg_fitness", "平均适应度", 140),
        ("best_makespan", "每代完成时间", 140),
        ("best_line_stop", "停线时间", 120),
    ]:
        history_tree.heading(column, text=title)
        history_tree.column(column, width=width, anchor="center")
    history_tree.pack(side="left", fill="y", padx=(0, 10), pady=5)

    chart_container = ttk.Frame(history_frame)
    chart_container.pack(side="left", fill="both", expand=True)
    canvas_holder: Dict[str, object] = {}

    def append_summary(text: str) -> None:
        summary_text.insert("end", text + "\n")
        summary_text.see("end")

    def run_from_ui() -> None:
        try:
            generations = int(generations_var.get())
            target_qualified_units = float(target_var.get())
            agv_count = int(agv_var.get())
            population_size = int(pop_var.get())
            if generations <= 0 or target_qualified_units <= 0 or agv_count <= 0 or population_size <= 1:
                raise ValueError
        except ValueError:
            if messagebox is not None:
                messagebox.showerror("参数错误", "请输入合法的正数参数。")
            return

        summary_text.delete("1.0", "end")
        for item in history_tree.get_children():
            history_tree.delete(item)
        if "canvas" in canvas_holder:
            canvas_holder["canvas"].get_tk_widget().destroy()
            canvas_holder.clear()

        try:
            result = run_experiment(
                generations=generations,
                target_qualified_units=target_qualified_units,
                agv_count=agv_count,
                population_size=population_size,
                seed=42,
            )
        except Exception as exc:
            if messagebox is not None:
                messagebox.showerror("运行失败", str(exc))
            return

        append_summary("=== 参数 ===")
        append_summary(f"GA迭代次数: {generations}")
        append_summary(f"目标合格品数量: {target_qualified_units}")
        append_summary(f"AGV数量: {agv_count}")
        append_summary(f"种群规模: {population_size}")
        append_summary("")

        append_summary("=== 当前任务池 ===")
        for task in result.task_pool.values():
            append_summary(
                f"{task.task_id} | {task.task_type} | {task.pickup_node} -> {task.drop_node} | qty={task.qty} | latest={task.latest_finish:.2f}"
            )
        append_summary("")

        append_summary("=== 结果对比 ===")
        append_summary(
            f"人工: fitness={result.manual_evaluation.fitness:.2f}, makespan={result.manual_evaluation.makespan:.2f}, line_stop={result.manual_evaluation.line_stop_time:.2f}"
        )
        append_summary(
            f"随机: fitness={result.random_evaluation.fitness:.2f}, makespan={result.random_evaluation.makespan:.2f}, line_stop={result.random_evaluation.line_stop_time:.2f}"
        )
        append_summary(
            f"GA最优: fitness={result.best_evaluation.fitness:.2f}, makespan={result.best_evaluation.makespan:.2f}, line_stop={result.best_evaluation.line_stop_time:.2f}"
        )
        append_summary(
            f"目标合格品: {result.best_evaluation.inventory_result.qualified_output_total:.2f}/{result.instance.global_params.target_qualified_units:.2f}"
        )
        append_summary("")

        append_summary("=== 最优AGV调度 ===")
        for agv_id, schedule in result.best_decoded.agv_schedules.items():
            append_summary(
                f"{agv_id}: tasks={schedule.assigned_tasks}, final_time={schedule.final_time:.2f}, battery={schedule.final_battery:.4f}"
            )

        for item in result.ga_result.history:
            history_tree.insert(
                "",
                "end",
                values=(
                    item.generation,
                    f"{item.best_fitness:.2f}",
                    f"{item.average_fitness:.2f}",
                    f"{item.best_makespan:.2f}",
                    f"{item.best_line_stop:.2f}",
                ),
            )

        figure = build_history_figure(result.ga_result.history)
        if figure is not None:
            canvas = FigureCanvasTkAgg(figure, master=chart_container)
            canvas.draw()
            canvas.get_tk_widget().pack(fill="both", expand=True)
            canvas_holder["canvas"] = canvas

    ttk.Button(control_frame, text="运行实验", command=run_from_ui).grid(row=0, column=8, padx=12, pady=6)
    ttk.Button(control_frame, text="退出", command=root.destroy).grid(row=0, column=9, padx=6, pady=6)

    root.mainloop()


def main() -> None:
    if tk is not None and (os.environ.get("DISPLAY") or os.name == "nt"):
        try:
            launch_gui()
            return
        except Exception:
            pass
    run_console_session()


if __name__ == "__main__":
    main()
