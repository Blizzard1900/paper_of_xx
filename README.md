# paper_of_xx

## GA AGV 调度仿真

默认启动 Tkinter 图形界面：

```bash
python3 ga_agv_scheduler.py
```

也可以显式启动 Tkinter 图形界面：

```bash
python3 ga_agv_scheduler.py --ui
```

运行后会弹出本机 GUI 窗口，可设置目标入库合格品数、AGV 数量、迭代代数、种群规模和最大仿真时间，并显示每代最优/平均/最差适应度、完成时间、合格品数量、任务等待时间、停线时间和适应度曲线。

命令行模式需要显式添加 `--cli`：

```bash
python3 ga_agv_scheduler.py --cli --target-good 200 --agv-count 4 --population 30 --generations 40
```

常用参数：

- `--target-good`：目标合格品入库数量，达到后停止仿真。
- `--agv-count`：AGV 数量。
- `--population`：遗传算法种群规模。
- `--generations`：遗传算法迭代次数。
- `--deterministic`：使用确定性生产节拍和 19/20 合格模式。
- `--max-time`：单次仿真最大运行时间，单位分钟。

适应度当前包含最大完成时间、总行驶距离、产线停线时间、任务等待时间和充电次数；其中停线和任务等待权重较高，用于放大不同调度策略之间的差异。
