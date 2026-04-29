from __future__ import annotations

from pprint import pprint

from agv_day1.data_loader import build_default_instance
from agv_day1.fitness_function import evaluate_solution
from agv_day1.scheduler_decoder import build_assignment_solution, decode_solution


def build_manual_solution():
    return build_assignment_solution(
        {
            "AGV_1": ["T1", "T3"],
            "AGV_2": ["T2", "T4"],
        }
    )


def main():
    instance = build_default_instance()
    solution = build_manual_solution()
    decoded = decode_solution(solution, instance)
    evaluation = evaluate_solution(solution, instance, decoded)

    print("=== Evaluation Summary ===")
    pprint(
        {
            "fitness": evaluation.fitness,
            "makespan": evaluation.makespan,
            "total_distance": evaluation.total_distance,
            "total_energy": evaluation.total_energy,
            "late_penalty": evaluation.late_penalty,
            "late_task_count": evaluation.late_task_count,
            "charge_count": evaluation.charge_count,
            "congestion_wait": evaluation.congestion_wait,
            "battery_penalty": evaluation.battery_penalty,
            "capacity_penalty": evaluation.capacity_penalty,
            "load_penalty": evaluation.load_penalty,
        }
    )

    print("\n=== AGV Details ===")
    for agv_id, schedule in evaluation.agv_schedules.items():
        print(f"\n--- {agv_id} ---")
        print("Assigned tasks:", schedule.assigned_tasks)
        print("Visited nodes:", schedule.visited_nodes)
        print("Final node:", schedule.final_node)
        print("Final time:", schedule.final_time)
        print("Final battery:", schedule.final_battery)
        print("Charging events:", len(schedule.charging_events))
        for charge_event in schedule.charging_events:
            print(
                "  Charge:",
                charge_event.charger_id,
                charge_event.start_time,
                charge_event.end_time,
                charge_event.battery_before,
                charge_event.battery_after,
            )

        print("Task executions:")
        for execution in schedule.task_executions:
            print(
                " ",
                execution.task_id,
                {
                    "pickup_arrival": execution.pickup_arrival,
                    "pickup_start": execution.pickup_start,
                    "pickup_end": execution.pickup_end,
                    "drop_arrival": execution.drop_arrival,
                    "drop_start": execution.drop_start,
                    "drop_end": execution.drop_end,
                    "late_amount": execution.late_amount,
                },
            )

        print("Battery trace:")
        for record in schedule.battery_trace:
            print(" ", record.time, record.battery, record.note)


if __name__ == "__main__":
    main()
