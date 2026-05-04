#!/usr/bin/env python3
"""
GA-based AGV scheduler for a CKD workshop production-logistics model.

The program simulates:
  - 10 x 12 raw material storage area, columns mapped to A/B/C/D;
  - 10 x 12 finished/defective storage area, rows 1-9 for qualified goods
    and row 10 for defective goods;
  - four production lines with station-level input buffers;
  - dynamic AGV transport requests triggered by buffer thresholds;
  - line-end semi-finished/product buffers and quality inspection;
  - a genetic algorithm that tunes dispatch priorities.

Run example:
  python ga_agv_scheduler.py --target-good 200 --agv-count 4 --population 30 --generations 40
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import threading
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Callable, Dict, Iterable, List, Optional, Tuple


Coordinate = Tuple[float, float]


SPEED_M_PER_MIN = 0.5 * 60.0
ENERGY_KWH_PER_M = 0.005
BATTERY_MAX_KWH = 10.0
BATTERY_MIN_KWH = 1.0
CHARGE_POWER_KWH_PER_MIN = 1.0
LOAD_UNLOAD_MIN = 0.5


GENE_NAMES = [
    "raw_A_priority",
    "raw_B_priority",
    "raw_C_priority",
    "raw_D_priority",
    "semi1_priority",
    "semi2_priority",
    "good_out_priority",
    "bad_out_priority",
    "urgency_weight",
    "distance_weight",
    "low_battery_weight",
]


MATERIAL_COLUMNS = {
    "A": range(1, 4),
    "B": range(4, 7),
    "C": range(7, 10),
    "D": range(10, 13),
}


TASK_PRIORITY_GENE = {
    "raw_A": "raw_A_priority",
    "raw_B": "raw_B_priority",
    "raw_C": "raw_C_priority",
    "raw_D": "raw_D_priority",
    "semi1": "semi1_priority",
    "semi2": "semi2_priority",
    "good_out": "good_out_priority",
    "bad_out": "bad_out_priority",
}


def manhattan(a: Coordinate, b: Coordinate) -> float:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def raw_storage_coord(row: int, col: int) -> Coordinate:
    return 4.2 + 3.2 * (col - 1), 5.2 + 3.2 * (row - 1)


def finished_storage_coord(row: int, col: int) -> Coordinate:
    if row <= 9:
        return 4.2 + 3.2 * (col - 1), -3.0 - 3.2 * (row - 1)
    return 4.2 + 3.2 * (col - 1), -31.8


def line1_coord(index: int) -> Coordinate:
    return -34.0 - 2.0 * index, 32.5


def line2_coord(index: int) -> Coordinate:
    return -17.0 - 3.0 * index, 31.0


def line3_coord(index: int) -> Coordinate:
    return -33.0 - 8.0 * index, 31.0


def line4_coord(index: int) -> Coordinate:
    return -20.0 - 2.0 * index, 0.0


def charger_coord(index: int) -> Coordinate:
    return 10.5 + 1.5 * index, 43.5


LINE_END_A: Coordinate = (32.5, -60.0)   # semi-finished product 1 source
LINE_END_B: Coordinate = (30.5, -27.0)   # semi-finished product 2 source
LINE_END_C: Coordinate = (31.0, -33.0)   # product before quality inspection
GOOD_BUFFER_COORD: Coordinate = (-36.0, 0.0)
BAD_BUFFER_COORD: Coordinate = (-39.0, 0.0)


@dataclass
class Buffer:
    capacity: int
    quantity: int
    threshold: int = 20
    refill_qty: int = 40
    incoming: int = 0

    @property
    def projected(self) -> int:
        return self.quantity + self.incoming

    @property
    def remaining(self) -> int:
        return self.capacity - self.quantity

    @property
    def projected_remaining(self) -> int:
        return self.capacity - self.projected


@dataclass
class Station:
    name: str
    coord: Coordinate
    cycle_mean_min: float
    input_buffers: Dict[str, Buffer]
    output_buffer_name: str
    next_finish_time: float = 0.0
    produced: int = 0
    blocked_minutes: float = 0.0


@dataclass
class Task:
    key: str
    task_type: str
    qty: int
    pickup: Coordinate
    drop: Coordinate
    created_at: float
    deadline: float
    station_index: Optional[int] = None
    material: Optional[str] = None
    storage_cell: Optional[Tuple[int, int]] = None


@dataclass
class AGV:
    agv_id: int
    pos: Coordinate
    available_time: float = 0.0
    battery_kwh: float = BATTERY_MAX_KWH
    total_distance: float = 0.0
    charging_count: int = 0
    task_count: int = 0


@dataclass
class ScheduledTask:
    task: Task
    agv_id: int
    finish_time: float
    distance: float
    energy_kwh: float


@dataclass
class SimulationResult:
    reached_target: bool
    completion_time: float
    good_stock: int
    bad_stock: int
    total_distance: float
    total_energy: float
    charge_count: int
    completed_tasks: int
    line_blocked_minutes: float
    pending_tasks: int
    fitness: float


@dataclass
class GenerationRecord:
    generation: int
    fitness: float
    completion_time: float
    good_stock: int


class WorkshopSimulator:
    def __init__(
        self,
        target_good: int,
        agv_count: int,
        genes: Dict[str, float],
        seed: int,
        max_time_min: float,
        dt_min: float,
        stochastic: bool,
        initial_material_fill_ratio: float,
        initial_wip_fill_ratio: float,
    ) -> None:
        self.target_good = target_good
        self.genes = genes
        self.rng = random.Random(seed)
        self.max_time_min = max_time_min
        self.dt_min = dt_min
        self.stochastic = stochastic
        self.time = 0.0
        self.completed_tasks = 0
        self.total_energy_kwh = 0.0
        self.inspected_count = 0

        self.raw_storage = self._init_raw_storage()
        self.finished_storage = self._init_finished_storage()
        self.good_stock = 0
        self.bad_stock = 0

        self.output_buffers: Dict[str, Buffer] = {
            "semi1_A": Buffer(capacity=240, quantity=0),
            "semi2_B": Buffer(capacity=240, quantity=0),
            "pre_qc_C": Buffer(capacity=240, quantity=0),
            "good_D": Buffer(capacity=120, quantity=0),
            "bad_E": Buffer(capacity=120, quantity=0),
        }

        self.stations = self._init_stations(
            initial_material_fill_ratio=initial_material_fill_ratio,
            initial_wip_fill_ratio=initial_wip_fill_ratio,
        )
        for station in self.stations:
            station.next_finish_time = self._next_cycle_time(station.cycle_mean_min)

        self.agvs = [
            AGV(agv_id=i, pos=charger_coord(i % 10))
            for i in range(agv_count)
        ]
        self.pending: Dict[str, Task] = {}
        self.in_transit: List[ScheduledTask] = []

    @staticmethod
    def _init_raw_storage() -> Dict[Tuple[int, int], int]:
        return {
            (row, col): 120
            for row in range(1, 11)
            for col in range(1, 13)
        }

    @staticmethod
    def _init_finished_storage() -> Dict[Tuple[int, int], int]:
        return {
            (row, col): 0
            for row in range(1, 11)
            for col in range(1, 13)
        }

    def _init_stations(
        self,
        initial_material_fill_ratio: float,
        initial_wip_fill_ratio: float,
    ) -> List[Station]:
        def initial(capacity: int, ratio: float) -> int:
            return max(0, min(capacity, int(round(capacity * ratio))))

        stations: List[Station] = []

        for i in range(13):
            stations.append(
                Station(
                    name=f"L1-{i + 1}",
                    coord=line1_coord(i),
                    cycle_mean_min=2.0,
                    input_buffers={
                        "A": Buffer(
                            capacity=60,
                            quantity=initial(60, initial_material_fill_ratio),
                            refill_qty=40,
                        ),
                    },
                    output_buffer_name="semi1_A",
                )
            )

        for i in range(5):
            stations.append(
                Station(
                    name=f"L2-{i + 1}",
                    coord=line2_coord(i),
                    cycle_mean_min=1.0,
                    input_buffers={
                        "B": Buffer(
                            capacity=80,
                            quantity=initial(80, initial_material_fill_ratio),
                            refill_qty=40,
                        ),
                        "semi1": Buffer(
                            capacity=120,
                            quantity=initial(120, initial_wip_fill_ratio),
                            refill_qty=40,
                        ),
                    },
                    output_buffer_name="semi2_B",
                )
            )

        for i in range(3):
            stations.append(
                Station(
                    name=f"L3-{i + 1}",
                    coord=line3_coord(i),
                    cycle_mean_min=0.55,
                    input_buffers={
                        "C": Buffer(
                            capacity=120,
                            quantity=initial(120, initial_material_fill_ratio),
                            refill_qty=40,
                        ),
                        "semi2": Buffer(
                            capacity=120,
                            quantity=initial(120, initial_wip_fill_ratio),
                            refill_qty=40,
                        ),
                    },
                    output_buffer_name="pre_qc_C",
                )
            )

        for i in range(9):
            stations.append(
                Station(
                    name=f"L4-{i + 1}",
                    coord=line4_coord(i),
                    cycle_mean_min=1.5,
                    input_buffers={
                        "D": Buffer(
                            capacity=120,
                            quantity=initial(120, initial_material_fill_ratio),
                            refill_qty=40,
                        ),
                        "pre_qc": Buffer(
                            capacity=120,
                            quantity=initial(120, initial_wip_fill_ratio),
                            refill_qty=40,
                        ),
                    },
                    output_buffer_name="quality",
                )
            )

        return stations

    def _next_cycle_time(self, mean_min: float) -> float:
        if not self.stochastic:
            return self.time + mean_min
        return self.time + max(0.01, self.rng.gauss(mean_min, 0.01))

    def run(self) -> SimulationResult:
        while self.time <= self.max_time_min and self.good_stock < self.target_good:
            self._complete_finished_agv_tasks()
            self._produce_due_items()
            self._generate_transport_requests()
            self._dispatch_idle_agvs()
            self.time += self.dt_min

        return self._make_result()

    def _complete_finished_agv_tasks(self) -> None:
        if not self.in_transit:
            return

        remaining: List[ScheduledTask] = []
        for scheduled in self.in_transit:
            if scheduled.finish_time <= self.time + 1e-9:
                self._complete_task(scheduled.task)
                self.completed_tasks += 1
                self.total_energy_kwh += scheduled.energy_kwh
            else:
                remaining.append(scheduled)
        self.in_transit = remaining

    def _produce_due_items(self) -> None:
        for station in self.stations:
            while station.next_finish_time <= self.time + 1e-9:
                if self._station_can_finish_cycle(station):
                    self._finish_station_cycle(station)
                    station.produced += 1
                    station.next_finish_time = self._next_cycle_time(station.cycle_mean_min)
                else:
                    station.blocked_minutes += self.dt_min
                    station.next_finish_time = self.time + self.dt_min
                    break

    def _station_can_finish_cycle(self, station: Station) -> bool:
        for buffer in station.input_buffers.values():
            if buffer.quantity < 1:
                return False

        if station.output_buffer_name == "quality":
            return True

        output = self.output_buffers[station.output_buffer_name]
        return output.quantity < output.capacity

    def _finish_station_cycle(self, station: Station) -> None:
        for material, buffer in station.input_buffers.items():
            buffer.quantity -= 1
            if material == "pre_qc":
                # L4 consumes one product from the line-3 end buffer through an
                # internal conveyor abstraction.
                pass

        if station.output_buffer_name == "quality":
            self.inspected_count += 1
            if self._inspection_passes():
                target = self.output_buffers["good_D"]
            else:
                target = self.output_buffers["bad_E"]

            if target.quantity < target.capacity:
                target.quantity += 1
            else:
                station.blocked_minutes += self.dt_min
            return

        self.output_buffers[station.output_buffer_name].quantity += 1

    def _inspection_passes(self) -> bool:
        if self.stochastic:
            return self.rng.random() < 0.95
        # Deterministic 95% pass rate: 19 qualified items and 1 defective item
        # in every block of 20 inspected products.
        return self.inspected_count % 20 != 0

    def _generate_transport_requests(self) -> None:
        for idx, station in enumerate(self.stations):
            for material, buffer in station.input_buffers.items():
                if buffer.projected >= buffer.threshold:
                    continue

                if material == "pre_qc":
                    self._try_create_pre_qc_task(idx, station, buffer)
                elif material == "semi1":
                    self._try_create_semi_task(
                        idx, station, buffer, "semi1", "semi1_A", LINE_END_A
                    )
                elif material == "semi2":
                    self._try_create_semi_task(
                        idx, station, buffer, "semi2", "semi2_B", LINE_END_B
                    )
                else:
                    self._try_create_raw_task(idx, station, material, buffer)

        self._try_create_output_task("good_out", "good_D", GOOD_BUFFER_COORD)
        self._try_create_output_task("bad_out", "bad_E", BAD_BUFFER_COORD)

    def _try_create_raw_task(
        self,
        station_index: int,
        station: Station,
        material: str,
        buffer: Buffer,
    ) -> None:
        key = f"raw:{station.name}:{material}"
        if key in self.pending:
            return

        qty = min(buffer.refill_qty, buffer.projected_remaining)
        if qty <= 0:
            return

        cell = self._nearest_raw_cell(material, station.coord, qty)
        if cell is None:
            return

        deadline = self.time + max(1.0, buffer.projected * station.cycle_mean_min)
        self.pending[key] = Task(
            key=key,
            task_type=f"raw_{material}",
            qty=qty,
            pickup=raw_storage_coord(*cell),
            drop=station.coord,
            created_at=self.time,
            deadline=deadline,
            station_index=station_index,
            material=material,
            storage_cell=cell,
        )

    def _try_create_semi_task(
        self,
        station_index: int,
        station: Station,
        buffer: Buffer,
        material: str,
        source_buffer_name: str,
        source_coord: Coordinate,
    ) -> None:
        key = f"{material}:{station.name}"
        if key in self.pending:
            return

        qty = min(buffer.refill_qty, buffer.projected_remaining)
        source = self.output_buffers[source_buffer_name]
        if qty <= 0 or source.quantity < qty:
            return

        deadline = self.time + max(1.0, buffer.projected * station.cycle_mean_min)
        self.pending[key] = Task(
            key=key,
            task_type=material,
            qty=qty,
            pickup=source_coord,
            drop=station.coord,
            created_at=self.time,
            deadline=deadline,
            station_index=station_index,
            material=material,
        )

    def _try_create_pre_qc_task(
        self,
        station_index: int,
        station: Station,
        buffer: Buffer,
    ) -> None:
        key = f"pre_qc:{station.name}"
        if key in self.pending:
            return

        qty = min(buffer.refill_qty, buffer.projected_remaining)
        source = self.output_buffers["pre_qc_C"]
        if qty <= 0 or source.quantity < qty:
            return

        deadline = self.time + max(1.0, buffer.projected * station.cycle_mean_min)
        self.pending[key] = Task(
            key=key,
            task_type="pre_qc",
            qty=qty,
            pickup=LINE_END_C,
            drop=station.coord,
            created_at=self.time,
            deadline=deadline,
            station_index=station_index,
            material="pre_qc",
        )

    def _try_create_output_task(
        self,
        task_type: str,
        source_buffer_name: str,
        source_coord: Coordinate,
    ) -> None:
        key = task_type
        if key in self.pending:
            return

        source = self.output_buffers[source_buffer_name]
        if source.remaining >= 40:
            return

        qty = min(40, source.quantity)
        if qty <= 0:
            return

        if task_type == "good_out":
            drop_cell = self._nearest_finished_cell(source_coord, defective=False)
        else:
            drop_cell = self._nearest_finished_cell(source_coord, defective=True)

        if drop_cell is None:
            return

        self.pending[key] = Task(
            key=key,
            task_type=task_type,
            qty=qty,
            pickup=source_coord,
            drop=finished_storage_coord(*drop_cell),
            created_at=self.time,
            deadline=self.time + 5.0,
            storage_cell=drop_cell,
        )

    def _nearest_raw_cell(
        self,
        material: str,
        target: Coordinate,
        min_qty: int,
    ) -> Optional[Tuple[int, int]]:
        candidates: List[Tuple[float, Tuple[int, int]]] = []
        for row in range(1, 11):
            for col in MATERIAL_COLUMNS[material]:
                if self.raw_storage[(row, col)] >= min_qty:
                    candidates.append((manhattan(raw_storage_coord(row, col), target), (row, col)))
        if not candidates:
            return None
        candidates.sort(key=lambda item: item[0])
        return candidates[0][1]

    def _nearest_finished_cell(
        self,
        source: Coordinate,
        defective: bool,
    ) -> Optional[Tuple[int, int]]:
        rows: Iterable[int] = [10] if defective else range(1, 10)
        candidates: List[Tuple[float, Tuple[int, int]]] = []
        for row in rows:
            for col in range(1, 13):
                if self.finished_storage[(row, col)] < 120:
                    coord = finished_storage_coord(row, col)
                    candidates.append((manhattan(source, coord), (row, col)))
        if not candidates:
            return None
        candidates.sort(key=lambda item: item[0])
        return candidates[0][1]

    def _dispatch_idle_agvs(self) -> None:
        if not self.pending:
            return

        for agv in self.agvs:
            if agv.available_time > self.time + 1e-9 or not self.pending:
                continue

            task = self._select_task_for_agv(agv)
            if task is None:
                continue

            if not self._reserve_task_resources(task):
                self.pending.pop(task.key, None)
                continue

            self.pending.pop(task.key, None)
            finish_time, distance, energy = self._schedule_task(agv, task)
            self.in_transit.append(
                ScheduledTask(
                    task=task,
                    agv_id=agv.agv_id,
                    finish_time=finish_time,
                    distance=distance,
                    energy_kwh=energy,
                )
            )

    def _select_task_for_agv(self, agv: AGV) -> Optional[Task]:
        best_score = -float("inf")
        best_task: Optional[Task] = None

        for task in self.pending.values():
            travel_distance = manhattan(agv.pos, task.pickup) + manhattan(task.pickup, task.drop)
            expected_duration = travel_distance / SPEED_M_PER_MIN + 2 * LOAD_UNLOAD_MIN
            slack = task.deadline - (self.time + expected_duration)
            urgency = 1.0 / max(0.2, slack) if slack >= 0 else 5.0 + abs(slack)
            low_battery = max(0.0, BATTERY_MAX_KWH * 0.35 - agv.battery_kwh)

            priority_gene = TASK_PRIORITY_GENE.get(task.task_type, "urgency_weight")
            score = (
                10.0 * self.genes.get(priority_gene, 1.0)
                + self.genes["urgency_weight"] * urgency
                - self.genes["distance_weight"] * travel_distance / 100.0
                - self.genes["low_battery_weight"] * low_battery
            )

            if score > best_score:
                best_score = score
                best_task = task

        return best_task

    def _reserve_task_resources(self, task: Task) -> bool:
        if task.task_type.startswith("raw_"):
            if task.storage_cell is None or task.material is None:
                return False
            cell = task.storage_cell
            if self.raw_storage[cell] < task.qty:
                new_cell = self._nearest_raw_cell(task.material, task.drop, task.qty)
                if new_cell is None:
                    return False
                task.storage_cell = new_cell
                task.pickup = raw_storage_coord(*new_cell)
                cell = new_cell
            self.raw_storage[cell] -= task.qty
            target = self.stations[task.station_index].input_buffers[task.material]  # type: ignore[index]
            target.incoming += task.qty
            return True

        if task.task_type == "semi1":
            return self._reserve_line_to_station(task, "semi1_A", "semi1")

        if task.task_type == "semi2":
            return self._reserve_line_to_station(task, "semi2_B", "semi2")

        if task.task_type == "pre_qc":
            return self._reserve_line_to_station(task, "pre_qc_C", "pre_qc")

        if task.task_type == "good_out":
            source = self.output_buffers["good_D"]
            if source.quantity < task.qty:
                return False
            source.quantity -= task.qty
            return True

        if task.task_type == "bad_out":
            source = self.output_buffers["bad_E"]
            if source.quantity < task.qty:
                return False
            source.quantity -= task.qty
            return True

        return False

    def _reserve_line_to_station(
        self,
        task: Task,
        source_buffer_name: str,
        target_material: str,
    ) -> bool:
        source = self.output_buffers[source_buffer_name]
        if source.quantity < task.qty:
            return False
        source.quantity -= task.qty
        target = self.stations[task.station_index].input_buffers[target_material]  # type: ignore[index]
        target.incoming += task.qty
        return True

    def _schedule_task(self, agv: AGV, task: Task) -> Tuple[float, float, float]:
        task_distance = manhattan(agv.pos, task.pickup) + manhattan(task.pickup, task.drop)
        task_energy = task_distance * ENERGY_KWH_PER_M
        elapsed = 0.0
        total_distance = task_distance

        if agv.battery_kwh - task_energy < BATTERY_MIN_KWH:
            charger = min((charger_coord(i) for i in range(10)), key=lambda c: manhattan(agv.pos, c))
            to_charger = manhattan(agv.pos, charger)
            charge_travel_energy = to_charger * ENERGY_KWH_PER_M
            elapsed += to_charger / SPEED_M_PER_MIN
            total_distance += to_charger
            agv.battery_kwh = max(0.0, agv.battery_kwh - charge_travel_energy)
            charge_needed = BATTERY_MAX_KWH - agv.battery_kwh
            elapsed += charge_needed / CHARGE_POWER_KWH_PER_MIN
            agv.battery_kwh = BATTERY_MAX_KWH
            agv.pos = charger
            agv.charging_count += 1

            task_distance = manhattan(agv.pos, task.pickup) + manhattan(task.pickup, task.drop)
            task_energy = task_distance * ENERGY_KWH_PER_M
            total_distance = to_charger + task_distance

        elapsed += task_distance / SPEED_M_PER_MIN + 2 * LOAD_UNLOAD_MIN
        agv.battery_kwh = max(0.0, agv.battery_kwh - task_energy)
        agv.pos = task.drop
        agv.available_time = self.time + elapsed
        agv.total_distance += total_distance
        agv.task_count += 1

        total_energy = total_distance * ENERGY_KWH_PER_M
        return agv.available_time, total_distance, total_energy

    def _complete_task(self, task: Task) -> None:
        if task.task_type.startswith("raw_") and task.material is not None:
            target = self.stations[task.station_index].input_buffers[task.material]  # type: ignore[index]
            target.quantity = min(target.capacity, target.quantity + task.qty)
            target.incoming = max(0, target.incoming - task.qty)
            return

        if task.task_type == "semi1":
            self._complete_station_delivery(task, "semi1")
            return

        if task.task_type == "semi2":
            self._complete_station_delivery(task, "semi2")
            return

        if task.task_type == "pre_qc":
            self._complete_station_delivery(task, "pre_qc")
            return

        if task.task_type == "good_out":
            stored = self._store_finished_goods(task.qty, defective=False)
            self.good_stock += stored
            return

        if task.task_type == "bad_out":
            stored = self._store_finished_goods(task.qty, defective=True)
            self.bad_stock += stored
            return

    def _complete_station_delivery(self, task: Task, target_material: str) -> None:
        target = self.stations[task.station_index].input_buffers[target_material]  # type: ignore[index]
        target.quantity = min(target.capacity, target.quantity + task.qty)
        target.incoming = max(0, target.incoming - task.qty)

    def _store_finished_goods(self, qty: int, defective: bool) -> int:
        remaining = qty
        rows: Iterable[int] = [10] if defective else range(1, 10)
        for row in rows:
            for col in range(1, 13):
                if remaining <= 0:
                    return qty
                cell = (row, col)
                room = 120 - self.finished_storage[cell]
                if room <= 0:
                    continue
                put = min(room, remaining)
                self.finished_storage[cell] += put
                remaining -= put
        return qty - remaining

    def _make_result(self) -> SimulationResult:
        total_distance = sum(agv.total_distance for agv in self.agvs)
        charge_count = sum(agv.charging_count for agv in self.agvs)
        blocked = sum(station.blocked_minutes for station in self.stations)
        reached = self.good_stock >= self.target_good
        completion_time = min(self.time, self.max_time_min)

        if reached:
            fitness = (
                completion_time
                + 0.002 * total_distance
                + 0.20 * blocked
                + 0.50 * charge_count
            )
        else:
            shortage = self.target_good - self.good_stock
            fitness = (
                self.max_time_min
                + 10000.0
                + 50.0 * shortage
                + 0.20 * blocked
                + 0.002 * total_distance
            )

        return SimulationResult(
            reached_target=reached,
            completion_time=completion_time,
            good_stock=self.good_stock,
            bad_stock=self.bad_stock,
            total_distance=total_distance,
            total_energy=self.total_energy_kwh,
            charge_count=charge_count,
            completed_tasks=self.completed_tasks,
            line_blocked_minutes=blocked,
            pending_tasks=len(self.pending),
            fitness=fitness,
        )


def random_genes(rng: random.Random) -> Dict[str, float]:
    genes = {}
    for name in GENE_NAMES:
        if name == "distance_weight":
            genes[name] = rng.uniform(0.0, 5.0)
        elif name == "low_battery_weight":
            genes[name] = rng.uniform(0.0, 3.0)
        else:
            genes[name] = rng.uniform(0.1, 5.0)
    return genes


def mutate_genes(
    genes: Dict[str, float],
    rng: random.Random,
    mutation_rate: float = 0.18,
    sigma: float = 0.35,
) -> Dict[str, float]:
    child = dict(genes)
    for name in GENE_NAMES:
        if rng.random() < mutation_rate:
            child[name] += rng.gauss(0.0, sigma)
            child[name] = max(0.0, min(8.0, child[name]))
    return child


def crossover(
    parent_a: Dict[str, float],
    parent_b: Dict[str, float],
    rng: random.Random,
) -> Dict[str, float]:
    child = {}
    for name in GENE_NAMES:
        if rng.random() < 0.5:
            child[name] = parent_a[name]
        else:
            alpha = rng.random()
            child[name] = alpha * parent_a[name] + (1.0 - alpha) * parent_b[name]
    return child


def tournament_select(
    evaluated: List[Tuple[float, Dict[str, float], SimulationResult]],
    rng: random.Random,
    size: int = 3,
) -> Dict[str, float]:
    contestants = rng.sample(evaluated, k=min(size, len(evaluated)))
    contestants.sort(key=lambda item: item[0])
    return contestants[0][1]


def evaluate_genes(
    genes: Dict[str, float],
    target_good: int,
    agv_count: int,
    seed: int,
    max_time_min: float,
    dt_min: float,
    stochastic: bool,
    initial_material_fill_ratio: float,
    initial_wip_fill_ratio: float,
) -> SimulationResult:
    simulator = WorkshopSimulator(
        target_good=target_good,
        agv_count=agv_count,
        genes=genes,
        seed=seed,
        max_time_min=max_time_min,
        dt_min=dt_min,
        stochastic=stochastic,
        initial_material_fill_ratio=initial_material_fill_ratio,
        initial_wip_fill_ratio=initial_wip_fill_ratio,
    )
    return simulator.run()


def run_ga(
    args: argparse.Namespace,
    progress_callback: Optional[Callable[[GenerationRecord], None]] = None,
) -> Tuple[Dict[str, float], SimulationResult, List[GenerationRecord]]:
    rng = random.Random(args.seed)
    population = [random_genes(rng) for _ in range(args.population)]
    best_genes: Optional[Dict[str, float]] = None
    best_result: Optional[SimulationResult] = None
    history: List[GenerationRecord] = []

    for generation in range(1, args.generations + 1):
        evaluated: List[Tuple[float, Dict[str, float], SimulationResult]] = []
        for idx, genes in enumerate(population):
            # Use a repeatable seed per generation/member. This keeps GA runs
            # reproducible while still allowing stochastic production cycles.
            sim_seed = args.seed + generation * 1009 + idx * 9176
            result = evaluate_genes(
                genes=genes,
                target_good=args.target_good,
                agv_count=args.agv_count,
                seed=sim_seed,
                max_time_min=args.max_time,
                dt_min=args.dt,
                stochastic=not args.deterministic,
                initial_material_fill_ratio=args.initial_material_fill_ratio,
                initial_wip_fill_ratio=args.initial_wip_fill_ratio,
            )
            evaluated.append((result.fitness, genes, result))

        evaluated.sort(key=lambda item: item[0])
        generation_best_fitness, generation_best_genes, generation_best_result = evaluated[0]

        if best_result is None or generation_best_fitness < best_result.fitness:
            best_genes = copy.deepcopy(generation_best_genes)
            best_result = generation_best_result

        record = GenerationRecord(
            generation=generation,
            fitness=generation_best_fitness,
            completion_time=generation_best_result.completion_time,
            good_stock=generation_best_result.good_stock,
        )
        history.append(record)
        if progress_callback is not None:
            progress_callback(record)

        if args.verbose or generation == 1 or generation == args.generations:
            print(
                f"Generation {generation:03d} | "
                f"best fitness={generation_best_fitness:.2f} | "
                f"time={generation_best_result.completion_time:.2f} min | "
                f"good={generation_best_result.good_stock} | "
                f"blocked={generation_best_result.line_blocked_minutes:.2f} min"
            )

        elite_count = max(1, int(args.population * 0.15))
        new_population = [copy.deepcopy(item[1]) for item in evaluated[:elite_count]]
        while len(new_population) < args.population:
            parent_a = tournament_select(evaluated, rng)
            parent_b = tournament_select(evaluated, rng)
            child = crossover(parent_a, parent_b, rng)
            child = mutate_genes(child, rng)
            new_population.append(child)
        population = new_population

    assert best_genes is not None and best_result is not None

    # Re-evaluate the best chromosome with the base seed for a clean report.
    final_result = evaluate_genes(
        genes=best_genes,
        target_good=args.target_good,
        agv_count=args.agv_count,
        seed=args.seed,
        max_time_min=args.max_time,
        dt_min=args.dt,
        stochastic=not args.deterministic,
        initial_material_fill_ratio=args.initial_material_fill_ratio,
        initial_wip_fill_ratio=args.initial_wip_fill_ratio,
    )
    return best_genes, final_result, history


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Genetic algorithm AGV scheduler for the CKD production-logistics model."
    )
    parser.add_argument("--target-good", type=int, default=200, help="target qualified stock count")
    parser.add_argument("--agv-count", type=int, default=4, help="number of AGVs")
    parser.add_argument("--population", type=int, default=30, help="GA population size")
    parser.add_argument("--generations", type=int, default=40, help="GA iteration count")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--max-time", type=float, default=8 * 60.0, help="max simulation time in minutes")
    parser.add_argument("--dt", type=float, default=0.05, help="simulation step in minutes")
    parser.add_argument(
        "--initial-material-fill-ratio",
        type=float,
        default=1.0,
        help="initial raw-material station buffer fill ratio, from 0 to 1",
    )
    parser.add_argument(
        "--initial-wip-fill-ratio",
        type=float,
        default=0.0,
        help="initial semi-finished/pre-QC station buffer fill ratio, from 0 to 1",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="use mean production cycles and a deterministic 19/20 quality pass pattern",
    )
    parser.add_argument("--ui", action="store_true", help="open a browser-based UI")
    parser.add_argument("--host", default="127.0.0.1", help="UI server host")
    parser.add_argument("--port", type=int, default=8000, help="UI server port")
    parser.add_argument("--verbose", action="store_true", help="print every generation")
    return parser.parse_args()


def print_report(genes: Dict[str, float], result: SimulationResult) -> None:
    print("\n=== Best dispatch genes ===")
    for name in GENE_NAMES:
        print(f"{name:20s}: {genes[name]:.4f}")

    print("\n=== Simulation result ===")
    print(f"Reached target       : {result.reached_target}")
    print(f"Completion time      : {result.completion_time:.2f} min")
    print(f"Qualified stock      : {result.good_stock}")
    print(f"Defective stock      : {result.bad_stock}")
    print(f"Completed AGV tasks  : {result.completed_tasks}")
    print(f"Total AGV distance   : {result.total_distance:.2f} m")
    print(f"Total AGV energy     : {result.total_energy:.2f} kWh")
    print(f"Charging count       : {result.charge_count}")
    print(f"Line blocked time    : {result.line_blocked_minutes:.2f} station-min")
    print(f"Pending tasks        : {result.pending_tasks}")
    print(f"Fitness              : {result.fitness:.2f}")


def format_result_text(genes: Dict[str, float], result: SimulationResult) -> str:
    lines = [
        "=== Simulation result ===",
        f"Reached target       : {result.reached_target}",
        f"Completion time      : {result.completion_time:.2f} min",
        f"Qualified stock      : {result.good_stock}",
        f"Defective stock      : {result.bad_stock}",
        f"Completed AGV tasks  : {result.completed_tasks}",
        f"Total AGV distance   : {result.total_distance:.2f} m",
        f"Total AGV energy     : {result.total_energy:.2f} kWh",
        f"Charging count       : {result.charge_count}",
        f"Line blocked time    : {result.line_blocked_minutes:.2f} station-min",
        f"Pending tasks        : {result.pending_tasks}",
        f"Fitness              : {result.fitness:.2f}",
        "",
        "=== Best dispatch genes ===",
    ]
    lines.extend(f"{name:20s}: {genes[name]:.4f}" for name in GENE_NAMES)
    return "\n".join(lines)


WEB_UI_HTML = """<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <title>CKD AGV Genetic Algorithm Scheduler</title>
  <style>
    body { font-family: Arial, sans-serif; margin: 24px; background: #f6f8fa; color: #222; }
    .panel { background: white; border: 1px solid #d0d7de; border-radius: 8px; padding: 16px; margin-bottom: 16px; }
    .grid { display: grid; grid-template-columns: repeat(6, minmax(120px, 1fr)); gap: 12px; align-items: end; }
    label { display: block; font-size: 13px; margin-bottom: 4px; color: #555; }
    input { width: 100%; box-sizing: border-box; padding: 8px; border: 1px solid #c8d0d8; border-radius: 6px; }
    button { padding: 9px 16px; border: 0; border-radius: 6px; background: #0969da; color: white; cursor: pointer; }
    button:disabled { background: #8c959f; cursor: not-allowed; }
    #status { font-weight: bold; margin-bottom: 10px; }
    canvas { width: 100%; height: 280px; border: 1px solid #d0d7de; background: white; border-radius: 8px; }
    pre { white-space: pre-wrap; max-height: 300px; overflow: auto; background: #0d1117; color: #e6edf3; padding: 12px; border-radius: 8px; }
    table { width: 100%; border-collapse: collapse; font-size: 13px; }
    th, td { border-bottom: 1px solid #d8dee4; padding: 6px; text-align: right; }
    th:first-child, td:first-child { text-align: left; }
  </style>
</head>
<body>
  <h2>CKD AGV 遗传算法调度 UI</h2>
  <div class="panel">
    <div class="grid">
      <div><label>目标入库合格品数</label><input id="target_good" type="number" value="200" min="1"></div>
      <div><label>AGV 数量</label><input id="agv_count" type="number" value="4" min="1"></div>
      <div><label>迭代代数</label><input id="generations" type="number" value="40" min="1"></div>
      <div><label>种群规模</label><input id="population" type="number" value="30" min="2"></div>
      <div><label>最大仿真时间/min</label><input id="max_time" type="number" value="480" min="1"></div>
      <div>
        <label><input id="deterministic" type="checkbox" style="width:auto"> 确定性模式</label>
        <button id="runBtn" onclick="runGa()">运行遗传算法</button>
      </div>
    </div>
  </div>
  <div class="panel">
    <div id="status">等待运行</div>
    <canvas id="chart" width="1000" height="280"></canvas>
  </div>
  <div class="panel">
    <h3>每代结果</h3>
    <table>
      <thead><tr><th>代数</th><th>适应度</th><th>完成时间/min</th><th>合格品数量</th></tr></thead>
      <tbody id="historyBody"></tbody>
    </table>
  </div>
  <div class="panel">
    <h3>最终结果</h3>
    <pre id="resultBox">暂无结果</pre>
  </div>
<script>
function getNumber(id) { return Number(document.getElementById(id).value); }

function drawChart(history) {
  const canvas = document.getElementById("chart");
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.fillStyle = "#333";
  ctx.font = "16px Arial";
  ctx.fillText("每代最佳适应度曲线", 20, 24);
  if (!history.length) {
    ctx.fillStyle = "#777";
    ctx.fillText("运行后显示曲线", 420, 140);
    return;
  }
  const values = history.map(x => x.fitness);
  const minV = Math.min(...values);
  const maxV = Math.max(...values);
  const span = Math.max(maxV - minV, 1e-9);
  const left = 55, right = 20, top = 40, bottom = 35;
  const w = canvas.width - left - right;
  const h = canvas.height - top - bottom;
  ctx.strokeStyle = "#999";
  ctx.beginPath();
  ctx.moveTo(left, top);
  ctx.lineTo(left, top + h);
  ctx.lineTo(left + w, top + h);
  ctx.stroke();
  ctx.fillStyle = "#555";
  ctx.font = "12px Arial";
  ctx.fillText(maxV.toFixed(1), 5, top + 4);
  ctx.fillText(minV.toFixed(1), 5, top + h);
  ctx.strokeStyle = "#1f77b4";
  ctx.lineWidth = 2;
  ctx.beginPath();
  history.forEach((item, index) => {
    const x = left + (history.length === 1 ? 0 : index * w / (history.length - 1));
    const y = top + h - (item.fitness - minV) * h / span;
    if (index === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  });
  ctx.stroke();
  ctx.fillStyle = "#1f77b4";
  history.forEach((item, index) => {
    const x = left + (history.length === 1 ? 0 : index * w / (history.length - 1));
    const y = top + h - (item.fitness - minV) * h / span;
    ctx.beginPath();
    ctx.arc(x, y, 3, 0, Math.PI * 2);
    ctx.fill();
  });
}

function renderHistory(history) {
  const body = document.getElementById("historyBody");
  body.innerHTML = "";
  history.forEach(item => {
    const row = document.createElement("tr");
    row.innerHTML = `<td>${item.generation}</td><td>${item.fitness.toFixed(2)}</td><td>${item.completion_time.toFixed(2)}</td><td>${item.good_stock}</td>`;
    body.appendChild(row);
  });
}

async function runGa() {
  const runBtn = document.getElementById("runBtn");
  const status = document.getElementById("status");
  const resultBox = document.getElementById("resultBox");
  const payload = {
    target_good: getNumber("target_good"),
    agv_count: getNumber("agv_count"),
    generations: getNumber("generations"),
    population: getNumber("population"),
    max_time: getNumber("max_time"),
    deterministic: document.getElementById("deterministic").checked
  };
  runBtn.disabled = true;
  status.textContent = "正在运行，请等待...";
  resultBox.textContent = "运行中...";
  renderHistory([]);
  drawChart([]);
  try {
    const response = await fetch("/api/run", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || "运行失败");
    renderHistory(data.history);
    drawChart(data.history);
    status.textContent = `完成 | 最终合格品 ${data.result.good_stock} | 完成时间 ${data.result.completion_time.toFixed(2)} min | 适应度 ${data.result.fitness.toFixed(2)}`;
    resultBox.textContent = data.report;
  } catch (err) {
    status.textContent = "运行出错";
    resultBox.textContent = String(err);
  } finally {
    runBtn.disabled = false;
  }
}
drawChart([]);
</script>
</body>
</html>
"""


def _result_to_dict(result: SimulationResult) -> Dict[str, object]:
    return {
        "reached_target": result.reached_target,
        "completion_time": result.completion_time,
        "good_stock": result.good_stock,
        "bad_stock": result.bad_stock,
        "total_distance": result.total_distance,
        "total_energy": result.total_energy,
        "charge_count": result.charge_count,
        "completed_tasks": result.completed_tasks,
        "line_blocked_minutes": result.line_blocked_minutes,
        "pending_tasks": result.pending_tasks,
        "fitness": result.fitness,
    }


def _record_to_dict(record: GenerationRecord) -> Dict[str, object]:
    return {
        "generation": record.generation,
        "fitness": record.fitness,
        "completion_time": record.completion_time,
        "good_stock": record.good_stock,
    }


def launch_ui(default_args: argparse.Namespace) -> None:
    class SchedulerHandler(BaseHTTPRequestHandler):
        def _send_bytes(self, status: int, body: bytes, content_type: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _send_json(self, status: int, payload: Dict[str, object]) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self._send_bytes(status, body, "application/json; charset=utf-8")

        def log_message(self, format: str, *args: object) -> None:
            return

        def do_GET(self) -> None:
            if self.path in ("/", "/index.html"):
                self._send_bytes(200, WEB_UI_HTML.encode("utf-8"), "text/html; charset=utf-8")
                return
            self._send_json(404, {"error": "Not found"})

        def do_POST(self) -> None:
            if self.path != "/api/run":
                self._send_json(404, {"error": "Not found"})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                payload = json.loads(self.rfile.read(length).decode("utf-8") or "{}")
                args = copy.copy(default_args)
                args.target_good = int(payload.get("target_good", args.target_good))
                args.agv_count = int(payload.get("agv_count", args.agv_count))
                args.generations = int(payload.get("generations", args.generations))
                args.population = int(payload.get("population", args.population))
                args.max_time = float(payload.get("max_time", args.max_time))
                args.deterministic = bool(payload.get("deterministic", args.deterministic))
                args.verbose = False
                if args.population < 2 or args.generations < 1 or args.agv_count < 1 or args.target_good < 1:
                    raise ValueError("参数不合法：种群规模>=2，迭代代数/AGV数量/目标合格品数均需>=1。")
                genes, result, history = run_ga(args)
                self._send_json(
                    200,
                    {
                        "result": _result_to_dict(result),
                        "history": [_record_to_dict(record) for record in history],
                        "report": format_result_text(genes, result),
                    },
                )
            except Exception as exc:
                self._send_json(400, {"error": str(exc)})

    server = ThreadingHTTPServer((default_args.host, default_args.port), SchedulerHandler)
    url = f"http://{default_args.host}:{default_args.port}"
    print(f"Browser UI is running at {url}")
    print("Open this address in your browser. Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping UI server.")
    finally:
        server.server_close()


def main() -> None:
    args = parse_args()
    if args.ui:
        launch_ui(args)
        return

    if args.population < 2:
        raise ValueError("--population must be at least 2")
    if args.generations < 1:
        raise ValueError("--generations must be at least 1")
    if args.agv_count < 1:
        raise ValueError("--agv-count must be at least 1")
    if args.target_good < 1:
        raise ValueError("--target-good must be at least 1")
    if not 0.0 <= args.initial_material_fill_ratio <= 1.0:
        raise ValueError("--initial-material-fill-ratio must be between 0 and 1")
    if not 0.0 <= args.initial_wip_fill_ratio <= 1.0:
        raise ValueError("--initial-wip-fill-ratio must be between 0 and 1")

    genes, result, _history = run_ga(args)
    print_report(genes, result)


if __name__ == "__main__":
    main()
