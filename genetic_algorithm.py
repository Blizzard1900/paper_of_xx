import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from data_loader import build_default_instance
from fitness_function import Evaluation, evaluate_solution
from models import Instance
from scheduler_decoder import decode_solution


@dataclass(frozen=True)
class GAConfig:
    population_size: int = 60
    generations: int = 120
    crossover_rate: float = 0.8
    mutation_rate: float = 0.15
    elite_count: int = 4
    tournament_size: int = 3
    heuristic_seed_count: int = 10
    rebalance_rate: float = 0.2
    priority_jitter: float = 0.25


@dataclass
class GAIndividual:
    solution: Dict[str, object]
    evaluation: Optional[Evaluation] = None

    @property
    def fitness(self) -> float:
        if self.evaluation is None:
            raise ValueError("Individual has not been evaluated yet.")
        return self.evaluation.fitness


@dataclass
class GAGenerationStats:
    generation: int
    best_fitness: float
    average_fitness: float
    best_makespan: float
    best_late_penalty: float


@dataclass
class GAResult:
    config: GAConfig
    best_individual: GAIndividual
    history: List[GAGenerationStats] = field(default_factory=list)


def build_random_solution(instance: Instance, rng: random.Random) -> Dict[str, object]:
    task_ids = list(instance.tasks.keys())
    agv_order = sorted(instance.agvs.keys())
    assignment = [rng.randrange(len(agv_order)) for _ in task_ids]
    priority = [rng.random() for _ in task_ids]
    return repair_solution(
        {
            "task_ids": task_ids,
            "assignment": assignment,
            "priority": priority,
            "agv_order": agv_order,
        },
        instance,
        rng,
    )


def build_heuristic_solution(instance: Instance, rng: random.Random) -> Dict[str, object]:
    task_items = sorted(
        instance.tasks.values(),
        key=lambda task: (task.earliest_start, task.latest_finish, task.task_id),
    )
    agv_order = sorted(instance.agvs.keys())
    agv_states = {
        agv_id: {
            "current_node": instance.agvs[agv_id].start_node,
            "assigned_count": 0,
            "last_priority": 0.0,
        }
        for agv_id in agv_order
    }

    task_ids: List[str] = []
    assignment: List[int] = []
    priority: List[float] = []

    for task in task_items:
        ranked_agvs = []
        for agv_id in agv_order:
            state = agv_states[agv_id]
            distance_bias = _distance_hint(instance, state["current_node"], task.pickup_node)
            time_window_bias = max(0.0, task.earliest_start * 0.01)
            load_bias = 1000.0 if task.qty > instance.agvs[agv_id].load_capacity else 0.0
            balance_bias = state["assigned_count"] * 5.0
            total_score = distance_bias + time_window_bias + load_bias + balance_bias
            ranked_agvs.append((total_score, agv_id))
        ranked_agvs.sort(key=lambda item: item[0])
        chosen_agv = ranked_agvs[0][1]
        chosen_index = agv_order.index(chosen_agv)
        state = agv_states[chosen_agv]
        state["assigned_count"] += 1
        state["current_node"] = task.drop_node
        state["last_priority"] += 1.0 + rng.random() * 0.01

        task_ids.append(task.task_id)
        assignment.append(chosen_index)
        priority.append(state["last_priority"])

    return repair_solution(
        {
            "task_ids": task_ids,
            "assignment": assignment,
            "priority": priority,
            "agv_order": agv_order,
        },
        instance,
        rng,
    )


def repair_solution(
    solution: Dict[str, object],
    instance: Instance,
    rng: Optional[random.Random] = None,
) -> Dict[str, object]:
    rng = rng or random.Random()
    task_ids = list(solution.get("task_ids", instance.tasks.keys()))
    agv_order = list(solution.get("agv_order", sorted(instance.agvs.keys())))
    assignment = list(solution.get("assignment", []))
    priority = list(solution.get("priority", []))

    if len(assignment) != len(task_ids):
        assignment = [rng.randrange(len(agv_order)) for _ in task_ids]
    if len(priority) != len(task_ids):
        priority = [rng.random() for _ in task_ids]

    for index, task_id in enumerate(task_ids):
        assignment[index] = int(assignment[index]) % len(agv_order)
        if task_id not in instance.tasks:
            task_ids[index] = list(instance.tasks.keys())[index % len(instance.tasks)]
        agv = instance.agvs[agv_order[assignment[index]]]
        task = instance.tasks[task_ids[index]]
        if task.qty > agv.load_capacity:
            feasible_indices = [
                agv_index
                for agv_index, agv_id in enumerate(agv_order)
                if task.qty <= instance.agvs[agv_id].load_capacity
            ]
            if feasible_indices:
                assignment[index] = feasible_indices[0]

    if rng.random() < 0.2:
        _rebalance_assignments(assignment, len(agv_order), rng)

    return {
        "task_ids": task_ids,
        "assignment": assignment,
        "priority": priority,
        "agv_order": agv_order,
    }


def evaluate_individual(individual: GAIndividual, instance: Instance) -> GAIndividual:
    decoded = decode_solution(individual.solution, instance)
    individual.evaluation = evaluate_solution(individual.solution, instance, decoded)
    return individual


def initialize_population(
    instance: Instance,
    config: GAConfig,
    rng: random.Random,
) -> List[GAIndividual]:
    population: List[GAIndividual] = []
    for _ in range(min(config.heuristic_seed_count, config.population_size)):
        population.append(GAIndividual(build_heuristic_solution(instance, rng)))
    while len(population) < config.population_size:
        population.append(GAIndividual(build_random_solution(instance, rng)))
    return population


def tournament_select(
    population: Sequence[GAIndividual],
    tournament_size: int,
    rng: random.Random,
) -> GAIndividual:
    candidates = rng.sample(list(population), k=min(tournament_size, len(population)))
    return min(candidates, key=lambda individual: individual.fitness)


def crossover(
    parent_a: GAIndividual,
    parent_b: GAIndividual,
    instance: Instance,
    config: GAConfig,
    rng: random.Random,
) -> Tuple[GAIndividual, GAIndividual]:
    task_ids = list(parent_a.solution["task_ids"])
    agv_order = list(parent_a.solution["agv_order"])

    assignment_a = list(parent_a.solution["assignment"])
    assignment_b = list(parent_b.solution["assignment"])
    priority_a = list(parent_a.solution["priority"])
    priority_b = list(parent_b.solution["priority"])

    child_assignment_a: List[int] = []
    child_assignment_b: List[int] = []
    child_priority_a: List[float] = []
    child_priority_b: List[float] = []

    for index in range(len(task_ids)):
        if rng.random() < 0.5:
            child_assignment_a.append(assignment_a[index])
            child_assignment_b.append(assignment_b[index])
        else:
            child_assignment_a.append(assignment_b[index])
            child_assignment_b.append(assignment_a[index])

        mix = rng.random()
        child_priority_a.append(priority_a[index] * mix + priority_b[index] * (1.0 - mix))
        child_priority_b.append(priority_b[index] * mix + priority_a[index] * (1.0 - mix))

    child_a = repair_solution(
        {
            "task_ids": task_ids,
            "assignment": child_assignment_a,
            "priority": child_priority_a,
            "agv_order": agv_order,
        },
        instance,
        rng,
    )
    child_b = repair_solution(
        {
            "task_ids": task_ids,
            "assignment": child_assignment_b,
            "priority": child_priority_b,
            "agv_order": agv_order,
        },
        instance,
        rng,
    )
    return GAIndividual(child_a), GAIndividual(child_b)


def mutate(
    individual: GAIndividual,
    instance: Instance,
    config: GAConfig,
    rng: random.Random,
) -> GAIndividual:
    solution = {
        "task_ids": list(individual.solution["task_ids"]),
        "assignment": list(individual.solution["assignment"]),
        "priority": list(individual.solution["priority"]),
        "agv_order": list(individual.solution["agv_order"]),
    }
    assignment = solution["assignment"]
    priority = solution["priority"]
    agv_order = solution["agv_order"]

    for index in range(len(assignment)):
        if rng.random() < config.mutation_rate:
            assignment[index] = rng.randrange(len(agv_order))
        if rng.random() < config.mutation_rate:
            priority[index] = float(priority[index]) + rng.uniform(
                -config.priority_jitter,
                config.priority_jitter,
            )

    if rng.random() < config.rebalance_rate:
        _rebalance_assignments(assignment, len(agv_order), rng)

    individual.solution = repair_solution(solution, instance, rng)
    individual.evaluation = None
    return individual


def solve_genetic_algorithm(
    instance: Optional[Instance] = None,
    config: Optional[GAConfig] = None,
    seed: int = 42,
) -> GAResult:
    instance = instance or build_default_instance()
    config = config or GAConfig()
    rng = random.Random(seed)

    population = initialize_population(instance, config, rng)
    population = [evaluate_individual(individual, instance) for individual in population]

    history: List[GAGenerationStats] = []
    best_individual = min(population, key=lambda individual: individual.fitness)

    for generation in range(config.generations):
        population.sort(key=lambda individual: individual.fitness)
        generation_best = population[0]
        generation_average = sum(ind.fitness for ind in population) / len(population)
        history.append(
            GAGenerationStats(
                generation=generation,
                best_fitness=generation_best.fitness,
                average_fitness=generation_average,
                best_makespan=generation_best.evaluation.makespan,
                best_late_penalty=generation_best.evaluation.late_penalty,
            )
        )
        if generation_best.fitness < best_individual.fitness:
            best_individual = _clone_individual(generation_best)

        next_population = [_clone_individual(ind) for ind in population[: config.elite_count]]

        while len(next_population) < config.population_size:
            parent_a = tournament_select(population, config.tournament_size, rng)
            parent_b = tournament_select(population, config.tournament_size, rng)

            if rng.random() < config.crossover_rate:
                child_a, child_b = crossover(parent_a, parent_b, instance, config, rng)
            else:
                child_a = _clone_individual(parent_a)
                child_b = _clone_individual(parent_b)

            mutate(child_a, instance, config, rng)
            evaluate_individual(child_a, instance)
            next_population.append(child_a)

            if len(next_population) < config.population_size:
                mutate(child_b, instance, config, rng)
                evaluate_individual(child_b, instance)
                next_population.append(child_b)

        population = next_population

    population.sort(key=lambda individual: individual.fitness)
    if population[0].fitness < best_individual.fitness:
        best_individual = _clone_individual(population[0])

    return GAResult(
        config=config,
        best_individual=best_individual,
        history=history,
    )


def _rebalance_assignments(
    assignment: List[int],
    agv_count: int,
    rng: random.Random,
) -> None:
    counts = {index: assignment.count(index) for index in range(agv_count)}
    busiest = max(counts, key=counts.get)
    idlest = min(counts, key=counts.get)
    if counts[busiest] - counts[idlest] <= 1:
        return
    candidate_indices = [idx for idx, agv_index in enumerate(assignment) if agv_index == busiest]
    if candidate_indices:
        move_index = rng.choice(candidate_indices)
        assignment[move_index] = idlest


def _distance_hint(instance: Instance, source: str, target: str) -> float:
    source_node = instance.nodes[source]
    target_node = instance.nodes[target]
    return abs(source_node.x - target_node.x) + abs(source_node.y - target_node.y)


def _clone_individual(individual: GAIndividual) -> GAIndividual:
    cloned_solution = {
        "task_ids": list(individual.solution["task_ids"]),
        "assignment": list(individual.solution["assignment"]),
        "priority": list(individual.solution["priority"]),
        "agv_order": list(individual.solution["agv_order"]),
    }
    return GAIndividual(solution=cloned_solution, evaluation=individual.evaluation)
