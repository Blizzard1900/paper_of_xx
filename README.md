# paper_of_xx

## GA AGV 调度仿真

运行示例：

```bash
python3 ga_agv_scheduler.py --target-good 200 --agv-count 4 --population 30 --generations 40
```

启动 Web 图形界面：

```bash
python3 ga_agv_scheduler.py --ui
```

启动后终端会显示访问地址，默认打开：

```text
http://127.0.0.1:8000
```

如果需要从外部访问，可指定监听地址：

```bash
python3 ga_agv_scheduler.py --ui --host 0.0.0.0 --port 8000
```

常用参数：

- `--target-good`：目标合格品入库数量，达到后停止仿真。
- `--agv-count`：AGV 数量。
- `--population`：遗传算法种群规模。
- `--generations`：遗传算法迭代次数。
- `--deterministic`：使用确定性生产节拍和 19/20 合格模式。
- `--max-time`：单次仿真最大运行时间，单位分钟。
