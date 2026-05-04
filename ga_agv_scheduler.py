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
import random
import threading
from dataclasses import dataclass
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
    total_task_wait: float
    max_task_wait: float
    pending_tasks: int
    fitness: float


@dataclass
class GenerationRecord:
    generation: int
    fitness: float
    avg_fitness: float
    worst_fitness: float
    completion_time: float
    good_stock: int
    total_task_wait: float
    line_blocked_minutes: float


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
        self.total_task_wait = 0.0
        self.max_task_wait = 0.0

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
        task_wait = max(0.0, self.time - task.created_at)
        self.total_task_wait += task_wait
        self.max_task_wait = max(self.max_task_wait, task_wait)

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
                + 0.010 * total_distance
                + 0.80 * blocked
                + 0.50 * self.total_task_wait
                + 2.00 * charge_count
            )
        else:
            shortage = self.target_good - self.good_stock
            fitness = (
                self.max_time_min
                + 10000.0
                + 50.0 * shortage
                + 0.80 * blocked
                + 0.50 * self.total_task_wait
                + 0.010 * total_distance
                + 2.00 * charge_count
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
            total_task_wait=self.total_task_wait,
            max_task_wait=self.max_task_wait,
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
        generation_worst_fitness = evaluated[-1][0]
        generation_avg_fitness = sum(item[0] for item in evaluated) / len(evaluated)

        if best_result is None or generation_best_fitness < best_result.fitness:
            best_genes = copy.deepcopy(generation_best_genes)
            best_result = generation_best_result

        record = GenerationRecord(
            generation=generation,
            fitness=generation_best_fitness,
            avg_fitness=generation_avg_fitness,
            worst_fitness=generation_worst_fitness,
            completion_time=generation_best_result.completion_time,
            good_stock=generation_best_result.good_stock,
            total_task_wait=generation_best_result.total_task_wait,
            line_blocked_minutes=generation_best_result.line_blocked_minutes,
        )
        history.append(record)
        if progress_callback is not None:
            progress_callback(record)

        if args.verbose or generation == 1 or generation == args.generations:
            print(
                f"Generation {generation:03d} | "
                f"best={generation_best_fitness:.2f} | "
                f"avg={generation_avg_fitness:.2f} | "
                f"worst={generation_worst_fitness:.2f} | "
                f"time={generation_best_result.completion_time:.2f} min | "
                f"good={generation_best_result.good_stock} | "
                f"blocked={generation_best_result.line_blocked_minutes:.2f} min | "
                f"wait={generation_best_result.total_task_wait:.2f} min"
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
    parser.add_argument("--ui", action="store_true", help="open a Tkinter GUI")
    parser.add_argument("--cli", action="store_true", help="run in command-line mode instead of GUI")
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
    print(f"Total task wait      : {result.total_task_wait:.2f} min")
    print(f"Max task wait        : {result.max_task_wait:.2f} min")
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
        f"Total task wait      : {result.total_task_wait:.2f} min",
        f"Max task wait        : {result.max_task_wait:.2f} min",
        f"Pending tasks        : {result.pending_tasks}",
        f"Fitness              : {result.fitness:.2f}",
        "",
        "=== Best dispatch genes ===",
    ]
    lines.extend(f"{name:20s}: {genes[name]:.4f}" for name in GENE_NAMES)
    return "\n".join(lines)


def launch_ui(default_args: argparse.Namespace) -> None:
    try:
        import tkinter as tk
        from tkinter import messagebox, ttk
    except ImportError as exc:
        raise RuntimeError(
            "当前 Python 环境没有 tkinter。请在本机安装 tkinter 后运行，"
            "Windows/Mac 通常自带，Linux 可安装 python3-tk。"
        ) from exc

    root = tk.Tk()
    root.title("CKD AGV 遗传算法调度")
    root.geometry("1100x780")

    input_frame = ttk.LabelFrame(root, text="参数设置")
    input_frame.pack(fill="x", padx=10, pady=8)

    fields = [
        ("目标入库合格品数", "target_good", default_args.target_good),
        ("AGV 数量", "agv_count", default_args.agv_count),
        ("迭代代数", "generations", default_args.generations),
        ("种群规模", "population", default_args.population),
        ("最大仿真时间/min", "max_time", default_args.max_time),
    ]
    entries: Dict[str, tk.Entry] = {}
    for col, (label, key, value) in enumerate(fields):
        ttk.Label(input_frame, text=label).grid(row=0, column=col, padx=6, pady=4)
        entry = ttk.Entry(input_frame, width=16)
        entry.insert(0, str(value))
        entry.grid(row=1, column=col, padx=6, pady=4)
        entries[key] = entry

    deterministic_var = tk.BooleanVar(value=default_args.deterministic)
    ttk.Checkbutton(input_frame, text="确定性模式", variable=deterministic_var).grid(
        row=1, column=len(fields), padx=8, pady=4
    )

    run_button = ttk.Button(input_frame, text="运行遗传算法")
    run_button.grid(row=1, column=len(fields) + 1, padx=8, pady=4)

    status_var = tk.StringVar(value="等待运行")
    ttk.Label(root, textvariable=status_var).pack(anchor="w", padx=12)

    canvas = tk.Canvas(
        root,
        height=250,
        bg="white",
        highlightthickness=1,
        highlightbackground="#cccccc",
    )
    canvas.pack(fill="x", padx=10, pady=8)

    table_frame = ttk.LabelFrame(root, text="每代结果")
    table_frame.pack(fill="both", expand=True, padx=10, pady=8)
    columns = (
        "generation",
        "best_fitness",
        "avg_fitness",
        "worst_fitness",
        "completion_time",
        "good_stock",
        "task_wait",
        "blocked",
    )
    tree = ttk.Treeview(table_frame, columns=columns, show="headings", height=8)
    tree.heading("generation", text="代数")
    tree.heading("best_fitness", text="最优适应度")
    tree.heading("avg_fitness", text="平均适应度")
    tree.heading("worst_fitness", text="最差适应度")
    tree.heading("completion_time", text="完成时间/min")
    tree.heading("good_stock", text="合格品数量")
    tree.heading("task_wait", text="任务等待/min")
    tree.heading("blocked", text="停线/min")
    for column in columns:
        tree.column(column, anchor="center", width=120)
    tree.pack(side="left", fill="both", expand=True)
    tree_scroll = ttk.Scrollbar(table_frame, orient="vertical", command=tree.yview)
    tree_scroll.pack(side="right", fill="y")
    tree.configure(yscrollcommand=tree_scroll.set)

    result_frame = ttk.LabelFrame(root, text="最终结果")
    result_frame.pack(fill="both", expand=True, padx=10, pady=8)
    result_text = tk.Text(result_frame, height=10, wrap="none")
    result_text.pack(side="left", fill="both", expand=True)
    result_scroll = ttk.Scrollbar(result_frame, orient="vertical", command=result_text.yview)
    result_scroll.pack(side="right", fill="y")
    result_text.configure(yscrollcommand=result_scroll.set)

    history: List[GenerationRecord] = []

    def draw_fitness(records: List[GenerationRecord]) -> None:
        canvas.delete("all")
        width = max(canvas.winfo_width(), 500)
        height = max(canvas.winfo_height(), 220)
        margin = 48
        canvas.create_text(width / 2, 18, text="每代最佳适应度曲线", fill="#333333")
        if not records:
            canvas.create_text(width / 2, height / 2, text="运行后显示曲线", fill="#777777")
            return

        values = [record.fitness for record in records]
        min_v = min(values)
        max_v = max(values)
        span = max(max_v - min_v, 1e-9)
        x_span = max(len(records) - 1, 1)

        canvas.create_line(margin, height - margin, width - margin, height - margin, fill="#999999")
        canvas.create_line(margin, margin, margin, height - margin, fill="#999999")
        canvas.create_text(margin + 4, margin - 14, text=f"{max_v:.1f}", anchor="w", fill="#666666")
        canvas.create_text(margin + 4, height - margin + 16, text=f"{min_v:.1f}", anchor="w", fill="#666666")

        points: List[float] = []
        for idx, value in enumerate(values):
            x = margin + idx * (width - 2 * margin) / x_span
            y = height - margin - (value - min_v) * (height - 2 * margin) / span
            points.extend([x, y])
            canvas.create_oval(x - 3, y - 3, x + 3, y + 3, fill="#1f77b4", outline="")
        if len(points) >= 4:
            canvas.create_line(*points, fill="#1f77b4", width=2)

    def build_args_from_ui() -> argparse.Namespace:
        args = copy.copy(default_args)
        try:
            args.target_good = int(entries["target_good"].get())
            args.agv_count = int(entries["agv_count"].get())
            args.generations = int(entries["generations"].get())
            args.population = int(entries["population"].get())
            args.max_time = float(entries["max_time"].get())
        except ValueError as exc:
            raise ValueError("请输入有效的数字参数。") from exc
        args.deterministic = deterministic_var.get()
        args.verbose = False
        return args

    def validate_ui_args(args: argparse.Namespace) -> None:
        if args.population < 2:
            raise ValueError("种群规模必须至少为 2。")
        if args.generations < 1:
            raise ValueError("迭代代数必须至少为 1。")
        if args.agv_count < 1:
            raise ValueError("AGV 数量必须至少为 1。")
        if args.target_good < 1:
            raise ValueError("目标入库合格品数必须至少为 1。")

    def handle_progress(record: GenerationRecord) -> None:
        history.append(record)
        tree.insert(
            "",
            "end",
            values=(
                record.generation,
                f"{record.fitness:.2f}",
                f"{record.avg_fitness:.2f}",
                f"{record.worst_fitness:.2f}",
                f"{record.completion_time:.2f}",
                record.good_stock,
                f"{record.total_task_wait:.2f}",
                f"{record.line_blocked_minutes:.2f}",
            ),
        )
        tree.yview_moveto(1.0)
        status_var.set(
            f"第 {record.generation} 代 | best {record.fitness:.2f} | "
            f"avg {record.avg_fitness:.2f} | worst {record.worst_fitness:.2f} | "
            f"完成时间 {record.completion_time:.2f} min | 合格品 {record.good_stock}"
        )
        draw_fitness(history)

    def handle_done(
        genes: Dict[str, float],
        result: SimulationResult,
        records: List[GenerationRecord],
    ) -> None:
        run_button.configure(state="normal")
        status_var.set(
            f"完成 | 最终合格品 {result.good_stock} | "
            f"完成时间 {result.completion_time:.2f} min | 适应度 {result.fitness:.2f}"
        )
        result_text.delete("1.0", "end")
        result_text.insert("end", format_result_text(genes, result))
        draw_fitness(records)

    def handle_error(exc: Exception) -> None:
        run_button.configure(state="normal")
        status_var.set("运行出错")
        messagebox.showerror("运行出错", str(exc))

    def worker(args: argparse.Namespace) -> None:
        def on_progress(record: GenerationRecord) -> None:
            root.after(0, handle_progress, record)

        try:
            genes, result, records = run_ga(args, progress_callback=on_progress)
        except Exception as exc:
            root.after(0, handle_error, exc)
            return
        root.after(0, handle_done, genes, result, records)

    def start_run() -> None:
        try:
            args = build_args_from_ui()
            validate_ui_args(args)
        except ValueError as exc:
            messagebox.showerror("参数错误", str(exc))
            return

        history.clear()
        for item in tree.get_children():
            tree.delete(item)
        result_text.delete("1.0", "end")
        draw_fitness(history)
        run_button.configure(state="disabled")
        status_var.set("正在运行...")
        thread = threading.Thread(target=worker, args=(args,), daemon=True)
        thread.start()

    run_button.configure(command=start_run)
    draw_fitness(history)
    root.mainloop()


def main() -> None:
    args = parse_args()
    if args.ui or not args.cli:
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
