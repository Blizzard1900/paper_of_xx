from pprint import pprint
import random

from data_loader import build_default_instance
from fitness_function import evaluate_solution
from genetic_algorithm import (
    GAConfig,
    build_random_solution,
    solve_genetic_algorithm,
)
from scheduler_decoder import decode_solution


def evaluate_random_baseline(instance, seed: int = 7):
    rng = random.Random(seed)
    random_solution = build_random_solution(instance, rng)
    decoded = decode_solution(random_solution, instance)
    evaluation = evaluate_solution(random_solution, instance, decoded)
    return random_solution, evaluation


def solution_to_assignment_map(solution):
    task_ids = list(solution["task_ids"])
    assignment = list(solution["assignment"])
    agv_order = list(solution["agv_order"])
    assignment_map = {agv_id: [] for agv_id in agv_order}
    priority_pairs = {agv_id: [] for agv_id in agv_order}
    for task_id, agv_index, priority in zip(task_ids, assignment, solution["priority"]):
        agv_id = agv_order[int(agv_index)]
        priority_pairs[agv_id].append((float(priority), task_id))
    for agv_id, items in priority_pairs.items():
        items.sort(key=lambda item: item[0])
        assignment_map[agv_id] = [task_id for _, task_id in items]
    return assignment_map


def main():
    instance = build_default_instance()

    random_solution, random_evaluation = evaluate_random_baseline(instance)

    config = GAConfig(
        population_size=60,
        generations=120,
        crossover_rate=0.85,
        mutation_rate=0.15,
        elite_count=2,
        tournament_size=3,
        heuristic_seed_count=10,
    )
    result = solve_genetic_algorithm(instance=instance, config=config, seed=42)
    best_solution = result.best_individual.solution
    best_evaluation = result.best_individual.evaluation

    print("=== Random Baseline ===")
    pprint(
        {
            "fitness": random_evaluation.fitness,
            "makespan": random_evaluation.makespan,
            "late_penalty": random_evaluation.late_penalty,
            "charge_count": random_evaluation.charge_count,
            "assignment": solution_to_assignment_map(random_solution),
            "priority": [round(value, 4) for value in random_solution["priority"]],
        }
    )

    print("\n=== Genetic Algorithm Best Result ===")
    pprint(
        {
            "fitness": best_evaluation.fitness,
            "makespan": best_evaluation.makespan,
            "total_distance": best_evaluation.total_distance,
            "total_energy": best_evaluation.total_energy,
            "late_penalty": best_evaluation.late_penalty,
            "late_task_count": best_evaluation.late_task_count,
            "charge_count": best_evaluation.charge_count,
            "congestion_wait": best_evaluation.congestion_wait,
            "battery_penalty": best_evaluation.battery_penalty,
            "capacity_penalty": best_evaluation.capacity_penalty,
            "load_penalty": best_evaluation.load_penalty,
        }
    )

    print("\n=== Best Solution Encoding ===")
    pprint(
        {
            "task_ids": best_solution["task_ids"],
            "assignment_vector": best_solution["assignment"],
            "priority": [round(value, 4) for value in best_solution["priority"]],
            "agv_order": best_solution["agv_order"],
            "assignment_map": solution_to_assignment_map(best_solution),
        }
    )

    print("\n=== Best AGV Schedules ===")
    for agv_id, schedule in best_evaluation.agv_schedules.items():
        print(f"\n--- {agv_id} ---")
        print("Assigned tasks:", schedule.assigned_tasks)
        print("Final time:", schedule.final_time)
        print("Final battery:", schedule.final_battery)
        print("Charge count:", len(schedule.charging_events))
        for execution in schedule.task_executions:
            print(
                " ",
                execution.task_id,
                {
                    "pickup_start": execution.pickup_start,
                    "drop_end": execution.drop_end,
                    "late_amount": execution.late_amount,
                },
            )

    print("\n=== Convergence History (first 10 / last 10) ===")
    history = [round(stat.best_fitness, 4) for stat in result.history]
    print("first_10:", history[:10])
    print("last_10:", history[-10:])


if __name__ == "__main__":
    main()
