# paper_of_xx

## AGV scheduling framework

This repository now contains a flat-file implementation scaffold for AGV scheduling:

- `models.py`: shared data structures
- `data_loader.py`: builds the instance, graph, tasks, and AGV definitions
- `shortest_path.py`: shortest-path distance, time, and energy utilities
- `scheduler_decoder.py`: decodes a scheduling solution into paths, timing, battery, and charging events
- `fitness_function.py`: aggregates makespan, energy, late penalties, congestion wait, and fitness
- `test_small_scenario.py`: runs the small manual validation scenario described in `agv_small_test_scenario.md`

Run the validation with:

```bash
python3 test_small_scenario.py
```
