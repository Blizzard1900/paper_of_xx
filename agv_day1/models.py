from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import networkx as nx


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

    @property
    def energy_kwh_per_m(self) -> float:
        return self.energy_ratio_per_m * self.battery_capacity_kwh

    @property
    def expected_worker_delay_s(self) -> float:
        return 1.0 / self.worker_delay_rate


@dataclass(frozen=True)
class Node:
    node_id: str
    node_type: str
    region: str
    x: float
    y: float
    service_capable: bool


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


@dataclass
class Instance:
    global_params: GlobalParams
    nodes: Dict[str, Node]
    edges: List[Edge]
    tasks: Dict[str, Task]
    agvs: Dict[str, AGV]
    graph: nx.DiGraph
    chargers: List[str]
    service_nodes: List[str]
    shortest_path_cache: Dict[Tuple[str, str, str], object] = field(default_factory=dict)


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
class DecodedSolution:
    agv_schedules: Dict[str, AGVDecodedSchedule]
    task_executions: Dict[str, TaskExecution]


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
    agv_schedules: Dict[str, AGVDecodedSchedule]

    def as_dict(self) -> Dict[str, object]:
        return {
            "fitness": self.fitness,
            "makespan": self.makespan,
            "total_distance": self.total_distance,
            "total_energy": self.total_energy,
            "late_penalty": self.late_penalty,
            "late_task_count": self.late_task_count,
            "charge_count": self.charge_count,
            "congestion_wait": self.congestion_wait,
            "battery_penalty": self.battery_penalty,
            "capacity_penalty": self.capacity_penalty,
            "load_penalty": self.load_penalty,
            "agv_schedules": self.agv_schedules,
        }
