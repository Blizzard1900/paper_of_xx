import matplotlib

# Use a non-interactive backend so the script works in headless environments.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import networkx as nx


def build_agv_topology():
    """Build and return the complete AGV directed topology graph."""
    graph = nx.DiGraph()

    # 1.1 路口节点
    intersections = {
        "N1": (-5, 32),  # 主干线1右端点
        "N2": (-60, 32),  # 主干线1左端点
        "N3": (-5, 12),  # 分支线下端点
        "N4": (-5, 32),  # 分支线上端点（同 N1）
        "N5": (1, 12),  # 通道干路右端点
        "N6": (-5, 12),  # 通道干路左端点（同 N3）
        "N7": (-5, 0),  # 主干线2右端点
        "N8": (-39, 0),  # 主干线2左端点
        "N9": (1, 34.0),  # 三车道上端点
        "N10": (1, -31.8),  # 三车道下端点
        "N11": (2.5, 34.5),  # 存储区网格右上角
        "N12": (2.5, 2.5),  # 存储区网格右下角
        "N13": (40.9, 2.5),  # 存储区网格左下角
        "N14": (40.9, 34.5),  # 存储区网格左上角
        "N15": (2.5, -0.3),  # 成品区网格右上角
        "N16": (2.5, -32.3),  # 成品区网格右下角
        "N17": (40.9, -32.3),  # 成品区网格左下角
        "N18": (40.9, -0.3),  # 成品区网格左上角
    }

    # 1.2 装配线1工位（12个）
    line1_stations = {f"L1_{j}": (-34 - 2 * (j - 1), 32) for j in range(1, 13)}

    # 1.3 装配线2工位（4个）
    line2_stations = {f"L2_{j}": (-17 - 3 * (j - 1), 32) for j in range(1, 5)}

    # 1.4 装配线3工位（3个）
    line3_stations = {f"L3_{j}": (-43 - 8 * (j - 1), 32) for j in range(1, 4)}

    # 1.5 包装线工位（13个）
    line4_stations = {f"L4_{j}": (-20 - 2 * (j - 1), 1) for j in range(1, 14)}

    # 1.6 装配线2物料投放点（4个）
    drop2_points = {f"D2_{j}": (-17 - 3 * (j - 1), 31) for j in range(1, 5)}

    # 1.7 装配线3物料投放点（3个）
    drop3_points = {f"D3_{j}": (-43 - 8 * (j - 1), 31) for j in range(1, 4)}

    # 1.8 暂存区节点
    buffers = {
        "B1": (32.5, -60),  # 装配线1成品库存
        "B2": (-27, 32),  # 装配线2成品暂存点
        "B3": (-38, 30.5),  # 装配线3成品暂存区
        "B4": (-36, 0),  # 成品暂存
        "B5": (-39, 0),  # 不良品暂存
    }

    # 1.9 充电位（20个：两排，Y=34 和 Y=35）
    chargers = {}
    for j in range(1, 11):
        chargers[f"C_{j}_low"] = (4.2 + 3.2 * (j - 1), 34)
        chargers[f"C_{j}_high"] = (4.2 + 3.2 * (j - 1), 35)

    # 1.10 物料存储区网格节点（10行 × 12列）
    storage_nodes = {}
    for r in range(1, 11):
        for c in range(1, 13):
            x = 4.2 + 3.2 * (c - 1)
            y = 5.2 + 3.2 * (r - 1)
            storage_nodes[f"S_{r}_{c}"] = (x, y)

    # 1.11 成品存储区网格节点（9行 × 12列）
    finish_nodes = {}
    for r in range(1, 10):
        for c in range(1, 13):
            x = 4.2 + 3.2 * (c - 1)
            y = -3.0 - 3.2 * (r - 1)
            finish_nodes[f"F_{r}_{c}"] = (x, y)

    # 1.12 不良品区节点（1行 × 12列）
    defect_nodes = {}
    for c in range(1, 13):
        x = 4.2 + 3.2 * (c - 1)
        y = -31.8
        defect_nodes[f"D_{c}"] = (x, y)

    all_nodes = {
        **intersections,
        **line1_stations,
        **line2_stations,
        **line3_stations,
        **line4_stations,
        **drop2_points,
        **drop3_points,
        **buffers,
        **chargers,
        **storage_nodes,
        **finish_nodes,
        **defect_nodes,
    }

    for node, pos in all_nodes.items():
        graph.add_node(node, pos=pos)

    edges = []

    # 2.1 主干道和支路
    main_roads = [
        ("N1", "N2"),
        ("N2", "N1"),  # 主干线1
        ("N3", "N1"),
        ("N1", "N3"),  # 分支线
        ("N5", "N6"),
        ("N6", "N5"),  # 通道干路
        ("N7", "N8"),
        ("N8", "N7"),  # 主干线2
        ("N5", "N9"),
        ("N9", "N5"),  # 三车道连接
        ("N6", "N7"),
        ("N7", "N6"),  # 主干线连接
    ]
    edges.extend(main_roads)

    # 2.2 三车道（4条，间距 0.5m，从 x=1 开始）
    for lane in range(4):
        x = 1 + lane * 0.5
        top_node = f"Lane_{lane}_top"
        bottom_node = f"Lane_{lane}_bottom"
        graph.add_node(top_node, pos=(x, 34.0))
        graph.add_node(bottom_node, pos=(x, -31.8))

        edges.append((top_node, bottom_node))
        edges.append((bottom_node, top_node))
        edges.append((top_node, "N9"))
        edges.append(("N9", top_node))
        edges.append((bottom_node, "N10"))
        edges.append(("N10", bottom_node))

    # 2.3 存储区网格边
    for c in range(1, 13):
        for r in range(1, 10):
            up_node = f"S_{r}_{c}"
            down_node = f"S_{r + 1}_{c}"
            edges.append((up_node, down_node))
            edges.append((down_node, up_node))

        edges.append((f"S_1_{c}", "N11"))
        edges.append(("N11", f"S_1_{c}"))
        edges.append((f"S_10_{c}", "N12"))
        edges.append(("N12", f"S_10_{c}"))

    for r in range(1, 11):
        for c in range(1, 12):
            left_node = f"S_{r}_{c}"
            right_node = f"S_{r}_{c + 1}"
            edges.append((left_node, right_node))
            edges.append((right_node, left_node))

    # 2.4 成品区网格边
    for c in range(1, 13):
        for r in range(1, 9):
            up_node = f"F_{r}_{c}"
            down_node = f"F_{r + 1}_{c}"
            edges.append((up_node, down_node))
            edges.append((down_node, up_node))

        edges.append((f"F_1_{c}", "N15"))
        edges.append(("N15", f"F_1_{c}"))
        edges.append((f"F_9_{c}", "N16"))
        edges.append(("N16", f"F_9_{c}"))

    for r in range(1, 10):
        for c in range(1, 12):
            left_node = f"F_{r}_{c}"
            right_node = f"F_{r}_{c + 1}"
            edges.append((left_node, right_node))
            edges.append((right_node, left_node))

    # 2.5 不良品区横向边及其与成品区连接
    for c in range(1, 12):
        edges.append((f"D_{c}", f"D_{c + 1}"))
        edges.append((f"D_{c + 1}", f"D_{c}"))

    for c in range(1, 13):
        edges.append((f"D_{c}", f"F_9_{c}"))
        edges.append((f"F_9_{c}", f"D_{c}"))

    # 2.6 充电位连接
    for j in range(1, 11):
        edges.append((f"C_{j}_high", "N9"))
        edges.append(("N9", f"C_{j}_high"))
        edges.append((f"C_{j}_low", "N9"))
        edges.append(("N9", f"C_{j}_low"))
        edges.append((f"C_{j}_low", f"C_{j}_high"))
        edges.append((f"C_{j}_high", f"C_{j}_low"))

    # 2.7 工位连接
    for j in range(1, 13):
        edges.append((f"L1_{j}", "N1"))
        edges.append(("N1", f"L1_{j}"))
    for j in range(1, 5):
        edges.append((f"L2_{j}", "N1"))
        edges.append(("N1", f"L2_{j}"))
    for j in range(1, 4):
        edges.append((f"L3_{j}", "N1"))
        edges.append(("N1", f"L3_{j}"))
    for j in range(1, 14):
        edges.append((f"L4_{j}", "N7"))
        edges.append(("N7", f"L4_{j}"))

    # 2.8 物料投放点连接
    for j in range(1, 5):
        edges.append((f"D2_{j}", f"L2_{j}"))
        edges.append((f"L2_{j}", f"D2_{j}"))
    for j in range(1, 4):
        edges.append((f"D3_{j}", f"L3_{j}"))
        edges.append((f"L3_{j}", f"D3_{j}"))

    # 2.9 暂存区连接
    edges.extend(
        [
            ("B1", "N8"),
            ("N8", "B1"),
            ("B2", "N1"),
            ("N1", "B2"),
            ("B3", "N1"),
            ("N1", "B3"),
            ("B4", "N7"),
            ("N7", "B4"),
            ("B5", "N8"),
            ("N8", "B5"),
        ]
    )

    # 2.10 边界节点连接
    edges.extend(
        [
            ("N11", "N9"),
            ("N9", "N11"),
            ("N12", "N10"),
            ("N10", "N12"),
            ("N15", "N9"),
            ("N9", "N15"),
            ("N16", "N10"),
            ("N10", "N16"),
        ]
    )

    graph.add_edges_from(edges)
    return graph, main_roads


def draw_agv_topology(graph, main_roads, output_path="agv_complete_topology.png"):
    """Draw the graph and save the figure to disk."""
    pos = nx.get_node_attributes(graph, "pos")
    plt.figure(figsize=(20, 16))

    nx.draw_networkx_edges(graph, pos, edge_color="lightgray", width=0.5, alpha=0.6)
    nx.draw_networkx_edges(graph, pos, edgelist=main_roads, edge_color="red", width=2.5)

    lane_edges = [(f"Lane_{i}_top", f"Lane_{i}_bottom") for i in range(4)]
    lane_edges += [(f"Lane_{i}_bottom", f"Lane_{i}_top") for i in range(4)]
    nx.draw_networkx_edges(graph, pos, edgelist=lane_edges, edge_color="blue", width=2)

    nx.draw_networkx_nodes(graph, pos, node_size=20, node_color="black", alpha=0.5)

    key_nodes = ["N1", "N2", "N3", "N5", "N6", "N7", "N8", "N9", "N10"]
    key_pos = {k: pos[k] for k in key_nodes if k in pos}
    nx.draw_networkx_labels(graph, key_pos, font_size=8, font_weight="bold")

    plt.title("AGV 完整路径拓扑图", fontsize=14)
    plt.axis("equal")
    plt.grid(True, linestyle=":", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def main():
    graph, main_roads = build_agv_topology()
    output_path = "agv_complete_topology.png"
    draw_agv_topology(graph, main_roads, output_path=output_path)

    print(f"节点总数: {graph.number_of_nodes()}")
    print(f"边总数: {graph.number_of_edges()}")
    print(f"拓扑图已保存至: {output_path}")


if __name__ == "__main__":
    main()
