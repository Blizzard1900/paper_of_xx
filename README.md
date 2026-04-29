# paper_of_xx

## Day 1 AGV scheduling framework

This repository now contains a Day 1 implementation scaffold for AGV scheduling:

- `agv_day1/data_loader.py`: builds the instance, graph, tasks, and AGV definitions
- `agv_day1/shortest_path.py`: shortest-path distance, time, and energy utilities
- `agv_day1/scheduler_decoder.py`: decodes a scheduling solution into paths, timing, battery, and charging events
- `agv_day1/fitness_function.py`: aggregates makespan, energy, late penalties, congestion wait, and fitness
- `test_small_scenario.py`: runs the small manual validation scenario described in `agv_small_test_scenario.md`

Run the Day 1 validation with:

```bash
python3 test_small_scenario.py
```
