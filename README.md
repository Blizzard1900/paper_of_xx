# paper_of_xx

## GA AGV 调度仿真

运行示例：

```bash
python3 ga_agv_scheduler.py --target-good 200 --agv-count 4 --population 30 --generations 40
```

常用参数：

- `--target-good`：目标合格品入库数量，达到后停止仿真。
- `--agv-count`：AGV 数量。
- `--population`：遗传算法种群规模。
- `--generations`：遗传算法迭代次数。
- `--deterministic`：使用确定性生产节拍和 19/20 合格模式。
- `--max-time`：单次仿真最大运行时间，单位分钟。
