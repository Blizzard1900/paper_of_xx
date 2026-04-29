from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

from .models import DecodedSolution, Evaluation, Instance


@dataclass(frozen=True)
class FitnessWeights:
    makespan: float = 1.0
    total_energy: float = 50.0
    congestion_wait: float = 20.0
    late_penalty: float = 500.0
    battery_penalty: float = 2000.0
    capacity_penalty: float = 2000.0
    load_penalty: float = 1500.0


def compute_late_penalty(decoded: DecodedSolution) -> tuple[float, int]:
    late_amount = 0.0
    late_count = 0
    for execution in decoded.task_executions.values():
        if execution.late_amount > 0:
            late_amount += execution.late_amount
            late_count += 1
    return late_amount, late_count


def compute_battery_penalty(instance: Instance, decoded: DecodedSolution) -> float:
    penalty = 0.0
    for agv_id, schedule in decoded.agv_schedules.items():
        agv = instance.agvs[agv_id]
        min_battery = min((record.battery for record in schedule.battery_trace), default=agv.battery_init)
        penalty += max(0.0, agv.battery_min - min_battery)
    return penalty


def compute_capacity_penalty(decoded: DecodedSolution) -> float:
    # Capacity conflicts are resolved by waiting in the decoder, so violations
    # should be zero in the deterministic Day 1 pipeline.
    return 0.0


def compute_load_penalty(instance: Instance, solution: Dict[str, object]) -> float:
    assignment = solution.get("assignment", {})
    penalty = 0.0
    if isinstance(assignment, dict):
        for task_id, agv_id in assignment.items():
            task = instance.tasks[task_id]
            agv = instance.agvs[agv_id]
            penalty += max(0.0, task.qty - agv.load_capacity)
    return penalty


def evaluate_solution(
    solution: Dict[str, object],
    instance: Instance,
    decoded: DecodedSolution,
    weights: FitnessWeights | None = None,
) -> Evaluation:
    weights = weights or FitnessWeights()

    makespan = max((schedule.final_time for schedule in decoded.agv_schedules.values()), default=0.0)
    total_distance = sum(schedule.total_distance for schedule in decoded.agv_schedules.values())
    total_energy = sum(schedule.total_energy for schedule in decoded.agv_schedules.values())
    congestion_wait = sum(schedule.total_wait for schedule in decoded.agv_schedules.values())
    charge_count = sum(len(schedule.charging_events) for schedule in decoded.agv_schedules.values())

    late_penalty, late_task_count = compute_late_penalty(decoded)
    battery_penalty = compute_battery_penalty(instance, decoded)
    capacity_penalty = compute_capacity_penalty(decoded)
    load_penalty = compute_load_penalty(instance, solution)

    fitness = (
        weights.makespan * makespan
        + weights.total_energy * total_energy
        + weights.congestion_wait * congestion_wait
        + weights.late_penalty * late_penalty
        + weights.battery_penalty * battery_penalty
        + weights.capacity_penalty * capacity_penalty
        + weights.load_penalty * load_penalty
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
        capacity_penalty=capacity_penalty,
        load_penalty=load_penalty,
        agv_schedules=decoded.agv_schedules,
    )
