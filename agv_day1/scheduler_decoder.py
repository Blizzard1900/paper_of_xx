from __future__ import annotations

from typing import Dict, List, Mapping, Sequence, Tuple

from .models import (
    AGV,
    AGVDecodedSchedule,
    BatteryRecord,
    ChargingEvent,
    DecodedSolution,
    EdgeTraversal,
    Instance,
    Task,
    TaskExecution,
)
from .shortest_path import shortest_path_metrics


def build_assignment_solution(
    assignment: Mapping[str, Sequence[str]],
) -> Dict[str, object]:
    task_ids: List[str] = []
    assignment_vector: List[int] = []
    priority_vector: List[float] = []

    agv_order = sorted(assignment.keys())
    agv_to_index = {agv_id: index for index, agv_id in enumerate(agv_order)}
    for agv_id in agv_order:
        for priority, task_id in enumerate(assignment[agv_id]):
            task_ids.append(task_id)
            assignment_vector.append(agv_to_index[agv_id])
            priority_vector.append(float(priority))

    return {
        "task_ids": task_ids,
        "assignment": assignment_vector,
        "priority": priority_vector,
        "agv_order": agv_order,
    }


def decode_solution(
    solution: Mapping[str, Sequence[float | int | str]],
    instance: Instance,
) -> DecodedSolution:
    agv_task_map = _decode_assignments(solution, instance)
    resource_availability: Dict[str, float] = {}

    agv_schedules = {
        agv_id: _decode_agv_schedule(
            agv=instance.agvs[agv_id],
            task_ids=task_ids,
            instance=instance,
            resource_availability=resource_availability,
        )
        for agv_id, task_ids in agv_task_map.items()
    }
    task_executions = {
        execution.task_id: execution
        for schedule in agv_schedules.values()
        for execution in schedule.task_executions
    }
    return DecodedSolution(agv_schedules=agv_schedules, task_executions=task_executions)


def _decode_assignments(
    solution: Mapping[str, Sequence[float | int | str]],
    instance: Instance,
) -> Dict[str, List[str]]:
    task_ids = list(solution["task_ids"])
    assignment = list(solution["assignment"])
    priority = list(solution["priority"])
    agv_order = list(solution.get("agv_order", sorted(instance.agvs.keys())))

    if not (len(task_ids) == len(assignment) == len(priority)):
        raise ValueError("Solution vectors must have the same length.")

    task_priority_pairs: Dict[str, List[Tuple[float, str]]] = {
        agv_id: [] for agv_id in instance.agvs
    }
    for task_id, agv_index, priority_value in zip(task_ids, assignment, priority):
        assigned_agv = agv_order[int(agv_index)]
        task_priority_pairs[assigned_agv].append((float(priority_value), str(task_id)))

    agv_task_map: Dict[str, List[str]] = {agv_id: [] for agv_id in instance.agvs}
    for agv_id, pairs in task_priority_pairs.items():
        pairs.sort(key=lambda item: item[0])
        agv_task_map[agv_id] = [task_id for _, task_id in pairs]
    return agv_task_map


def _decode_agv_schedule(
    *,
    agv: AGV,
    task_ids: Sequence[str],
    instance: Instance,
    resource_availability: Dict[str, float],
) -> AGVDecodedSchedule:
    current_node = agv.start_node
    current_time = 0.0
    current_battery = agv.battery_init

    visited_nodes: List[str] = [current_node]
    traversals: List[EdgeTraversal] = []
    charging_events: List[ChargingEvent] = []
    task_executions: List[TaskExecution] = []
    battery_trace: List[BatteryRecord] = [
        BatteryRecord(time=0.0, battery=current_battery, note="initial_state")
    ]

    for task_id in task_ids:
        task = instance.tasks[task_id]

        (
            current_node,
            current_time,
            current_battery,
            charge_traversals,
            charge_event,
        ) = _ensure_charge_for_task(
            agv=agv,
            task=task,
            current_node=current_node,
            current_time=current_time,
            current_battery=current_battery,
            instance=instance,
            resource_availability=resource_availability,
        )
        if charge_traversals:
            traversals.extend(charge_traversals)
            _extend_visited_nodes(visited_nodes, charge_traversals)
        if charge_event is not None:
            charging_events.append(charge_event)
            battery_trace.append(
                BatteryRecord(
                    time=current_time,
                    battery=current_battery,
                    note=f"charged_at_{charge_event.charger_id}",
                )
            )

        pickup_arrival, current_battery, pickup_traversals = _travel_and_consume(
            agv_id=agv.agv_id,
            source=current_node,
            target=task.pickup_node,
            start_time=current_time,
            battery=current_battery,
            instance=instance,
            reason="to_pickup",
            task_id=task.task_id,
            resource_availability=resource_availability,
        )
        traversals.extend(pickup_traversals)
        _extend_visited_nodes(visited_nodes, pickup_traversals)
        current_time = pickup_arrival

        pickup_start = max(pickup_arrival, task.earliest_start)
        pickup_end = pickup_start + task.pickup_service_time
        current_time = pickup_end

        drop_arrival, current_battery, drop_traversals = _travel_and_consume(
            agv_id=agv.agv_id,
            source=task.pickup_node,
            target=task.drop_node,
            start_time=current_time,
            battery=current_battery,
            instance=instance,
            reason="to_drop",
            task_id=task.task_id,
            resource_availability=resource_availability,
        )
        traversals.extend(drop_traversals)
        _extend_visited_nodes(visited_nodes, drop_traversals)
        current_time = drop_arrival

        drop_start = drop_arrival
        drop_end = drop_start + task.drop_service_time
        current_time = drop_end
        current_node = task.drop_node

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
        battery_trace.append(
            BatteryRecord(
                time=current_time,
                battery=current_battery,
                note=f"after_{task.task_id}",
            )
        )

    return AGVDecodedSchedule(
        agv_id=agv.agv_id,
        assigned_tasks=list(task_ids),
        visited_nodes=visited_nodes,
        traversals=traversals,
        charging_events=charging_events,
        task_executions=task_executions,
        battery_trace=battery_trace,
        total_distance=sum(traversal.distance for traversal in traversals),
        total_energy=sum(traversal.energy_cost for traversal in traversals),
        total_wait=sum(traversal.wait_before for traversal in traversals),
        final_node=current_node,
        final_time=current_time,
        final_battery=current_battery,
    )


def _ensure_charge_for_task(
    *,
    agv: AGV,
    task: Task,
    current_node: str,
    current_time: float,
    current_battery: float,
    instance: Instance,
    resource_availability: Dict[str, float],
) -> Tuple[str, float, float, List[EdgeTraversal], ChargingEvent | None]:
    required_energy = _required_energy_for_task_with_reserve(
        current_node=current_node,
        task=task,
        instance=instance,
    )
    if current_battery >= max(agv.battery_threshold, required_energy):
        return current_node, current_time, current_battery, [], None

    charger_id = _select_nearest_available_charger(current_node, instance)
    charge_arrival, battery_after_travel, traversals = _travel_and_consume(
        agv_id=agv.agv_id,
        source=current_node,
        target=charger_id,
        start_time=current_time,
        battery=current_battery,
        instance=instance,
        reason="to_charge",
        task_id=task.task_id,
        resource_availability=resource_availability,
    )
    battery_after_travel = max(0.0, battery_after_travel)

    charge_needed = max(0.0, agv.battery_max - battery_after_travel)
    charge_duration_seconds = 0.0
    if agv.charge_power > 0:
        charge_duration_seconds = (charge_needed / agv.charge_power) * 3600.0
    charge_event = ChargingEvent(
        charger_id=charger_id,
        start_time=charge_arrival,
        end_time=charge_arrival + charge_duration_seconds,
        battery_before=battery_after_travel,
        battery_after=agv.battery_max,
    )
    return (
        charger_id,
        charge_event.end_time,
        agv.battery_max,
        traversals,
        charge_event,
    )


def _required_energy_for_task_with_reserve(
    *,
    current_node: str,
    task: Task,
    instance: Instance,
) -> float:
    energy_to_pickup = shortest_path_metrics(instance, current_node, task.pickup_node)[3]
    energy_to_drop = shortest_path_metrics(instance, task.pickup_node, task.drop_node)[3]
    reserve_energy = min(
        shortest_path_metrics(instance, task.drop_node, charger_id)[3]
        for charger_id in instance.chargers
    )
    return energy_to_pickup + energy_to_drop + reserve_energy


def _select_nearest_available_charger(current_node: str, instance: Instance) -> str:
    return min(
        instance.chargers,
        key=lambda charger_id: shortest_path_metrics(instance, current_node, charger_id)[1],
    )


def _travel_and_consume(
    *,
    agv_id: str,
    source: str,
    target: str,
    start_time: float,
    battery: float,
    instance: Instance,
    reason: str,
    task_id: str,
    resource_availability: Dict[str, float],
) -> Tuple[float, float, List[EdgeTraversal]]:
    if source == target:
        return start_time, battery, []

    path = shortest_path_metrics(instance, source, target)[0]
    traversals: List[EdgeTraversal] = []
    current_time = start_time
    current_battery = battery

    for from_node, to_node in zip(path, path[1:]):
        edge = instance.edges[(from_node, to_node)]
        available_time = resource_availability.get(edge.resource_id, 0.0)
        wait_before = max(0.0, available_time - current_time)
        traversal_start = current_time + wait_before
        traversal_end = traversal_start + edge.base_time
        resource_availability[edge.resource_id] = traversal_end
        traversals.append(
            EdgeTraversal(
                agv_id=agv_id,
                from_node=from_node,
                to_node=to_node,
                resource_id=edge.resource_id,
                start_time=traversal_start,
                end_time=traversal_end,
                duration=edge.base_time,
                distance=edge.distance,
                energy_cost=edge.energy_cost,
                wait_before=wait_before,
                edge_type=edge.edge_type,
            )
        )
        current_time = traversal_end
        current_battery -= edge.energy_cost

    return current_time, current_battery, traversals


def _extend_visited_nodes(
    visited_nodes: List[str],
    traversals: Sequence[EdgeTraversal],
) -> None:
    for traversal in traversals:
        if not visited_nodes or visited_nodes[-1] != traversal.to_node:
            visited_nodes.append(traversal.to_node)
