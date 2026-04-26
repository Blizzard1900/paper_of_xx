import math
from collections import OrderedDict

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import networkx as nx
from matplotlib import font_manager

AGV_SPEED_MPS = 0.5
ENERGY_RATIO_PER_M = 5e-4


def generate_linear_nodes(prefix, start_x, start_y, step_x, count):
    return {
        f"{prefix}_{index}": (start_x + step_x * (index - 1), start_y)
        for index in range(1, count + 1)
    }


def build_node_groups():
    node_groups = OrderedDict()

    node_groups["junctions"] = {
        "N1": (-5, 32),
        "N2": (-60, 32),
        "N3": (-5, 12),
        "N5": (1, 12),
        "N7": (-5, 0),
        "N8": (-39, 0),
        "N9": (1, 34.0),
        "N10": (1, -31.8),
        "N11": (2.5, 34.5),
        "N12": (2.5, 2.5),
        "N15": (2.5, -0.3),
        "N16": (2.5, -32.3),
    }

    node_groups["line1_nodes"] = generate_linear_nodes("L1", -34, 32, -2, 12)
    node_groups["line1_drops"] = generate_linear_nodes("D1", -34, 32.5, -2, 12)
    node_groups["line2_nodes"] = generate_linear_nodes("L2", -17, 32, -3, 4)
    node_groups["line2_drops"] = generate_linear_nodes("D2", -17, 31, -3, 4)
    node_groups["line3_nodes"] = generate_linear_nodes("L3", -43, 32, -8, 3)
    node_groups["line3_drops"] = generate_linear_nodes("D3", -43, 31, -8, 3)
    node_groups["pack_nodes"] = generate_linear_nodes("P", -20, 1, -2, 13)

    buffers = {
        "A_STOCK": (32.5, -60),
        "B_STOCK": (30.5, -33),
        "B_TEMP": (-27, 32),
        "C_TEMP": (-38, 30.5),
        "C_GATE": (-38, 32),
        "PACK_BUFFER": (-36, 0),
    }
    node_groups["buffers"] = buffers

    chargers = {}
    for index in range(1, 11):
        x_coord = 4.2 + 3.2 * (index - 1)
        chargers[f"C_{index}_LOW"] = (x_coord, 34)
        chargers[f"C_{index}_HIGH"] = (x_coord, 35)
    node_groups["chargers"] = chargers

    lanes = {}
    for index in range(1, 5):
        x_coord = 1 + 0.5 * (index - 1)
        lanes[f"LANE_{index}_TOP"] = (x_coord, 34.0)
        lanes[f"LANE_{index}_BOTTOM"] = (x_coord, -31.8)
    node_groups["lanes"] = lanes

    storage_nodes = {}
    for row in range(1, 11):
        for col in range(1, 13):
            storage_nodes[f"S_{row}_{col}"] = (
                2.5 + 3.2 * (col - 1),
                2.5 + 3.2 * (row - 1),
            )
    node_groups["storage_nodes"] = storage_nodes

    finish_nodes = {}
    for row in range(1, 11):
        for col in range(1, 13):
            finish_nodes[f"F_{row}_{col}"] = (
                2.5 + 3.2 * (col - 1),
                -0.3 - 3.2 * (row - 1),
            )
    node_groups["finish_nodes"] = finish_nodes

    defect_nodes = {}
    for col in range(1, 13):
        defect_nodes[f"Q_{col}"] = (2.5 + 3.2 * (col - 1), -32.3)
    node_groups["defect_nodes"] = defect_nodes

    return node_groups


def add_nodes_to_graph(graph, node_groups):
    for group_name, nodes in node_groups.items():
        for node_id, position in nodes.items():
            graph.add_node(node_id, pos=position, group=group_name)


def add_bidirectional_edge(graph, source, target):
    x1, y1 = graph.nodes[source]["pos"]
    x2, y2 = graph.nodes[target]["pos"]
    length = math.hypot(x2 - x1, y2 - y1)
    travel_time = length / AGV_SPEED_MPS
    energy = length * ENERGY_RATIO_PER_M
    edge_attributes = {
        "length_m": round(length, 4),
        "travel_time_s": round(travel_time, 4),
        "energy_ratio": round(energy, 6),
    }
    graph.add_edge(source, target, **edge_attributes)
    graph.add_edge(target, source, **edge_attributes)


def connect_chain(graph, node_ids):
    for left, right in zip(node_ids, node_ids[1:]):
        add_bidirectional_edge(graph, left, right)


def connect_grid(graph, prefix, rows, cols):
    for row in range(1, rows + 1):
        row_nodes = [f"{prefix}_{row}_{col}" for col in range(1, cols + 1)]
        connect_chain(graph, row_nodes)

    for col in range(1, cols + 1):
        col_nodes = [f"{prefix}_{row}_{col}" for row in range(1, rows + 1)]
        connect_chain(graph, col_nodes)


def build_graph():
    graph = nx.DiGraph()
    node_groups = build_node_groups()
    add_nodes_to_graph(graph, node_groups)

    main_road_1 = sorted(
        [
            "N2",
            *node_groups["line1_nodes"].keys(),
            *node_groups["line2_nodes"].keys(),
            *node_groups["line3_nodes"].keys(),
            "B_TEMP",
            "C_GATE",
            "N1",
        ],
        key=lambda node_id: graph.nodes[node_id]["pos"][0],
    )
    connect_chain(graph, main_road_1)

    add_bidirectional_edge(graph, "N1", "N3")
    add_bidirectional_edge(graph, "N3", "N5")
    add_bidirectional_edge(graph, "N3", "N7")

    main_road_2 = sorted(
        ["N8", "PACK_BUFFER", "N7"],
        key=lambda node_id: graph.nodes[node_id]["pos"][0],
    )
    connect_chain(graph, main_road_2)

    for index in range(1, 13):
        add_bidirectional_edge(graph, f"L1_{index}", f"D1_{index}")

    for index in range(1, 5):
        add_bidirectional_edge(graph, f"L2_{index}", f"D2_{index}")

    for index in range(1, 4):
        add_bidirectional_edge(graph, f"L3_{index}", f"D3_{index}")

    pack_chain = sorted(
        node_groups["pack_nodes"].keys(),
        key=lambda node_id: graph.nodes[node_id]["pos"][0],
    )
    connect_chain(graph, pack_chain)
    add_bidirectional_edge(graph, "P_1", "N7")

    add_bidirectional_edge(graph, "C_TEMP", "C_GATE")
    add_bidirectional_edge(graph, "A_STOCK", "N10")
    add_bidirectional_edge(graph, "B_STOCK", "N10")
    add_bidirectional_edge(graph, "N5", "N9")

    for index in range(1, 5):
        top_node = f"LANE_{index}_TOP"
        bottom_node = f"LANE_{index}_BOTTOM"
        add_bidirectional_edge(graph, top_node, bottom_node)

    add_bidirectional_edge(graph, "N9", "LANE_1_TOP")
    add_bidirectional_edge(graph, "N10", "LANE_1_BOTTOM")
    for index in range(1, 4):
        add_bidirectional_edge(graph, f"LANE_{index}_TOP", f"LANE_{index + 1}_TOP")
        add_bidirectional_edge(
            graph, f"LANE_{index}_BOTTOM", f"LANE_{index + 1}_BOTTOM"
        )

    add_bidirectional_edge(graph, "N9", "C_1_LOW")
    for index in range(1, 11):
        add_bidirectional_edge(graph, f"C_{index}_LOW", f"C_{index}_HIGH")
    for index in range(1, 10):
        add_bidirectional_edge(graph, f"C_{index}_LOW", f"C_{index + 1}_LOW")
        add_bidirectional_edge(graph, f"C_{index}_HIGH", f"C_{index + 1}_HIGH")

    connect_grid(graph, "S", 10, 12)
    connect_grid(graph, "F", 10, 12)
    connect_chain(graph, [f"Q_{col}" for col in range(1, 13)])

    add_bidirectional_edge(graph, "N11", "N9")
    add_bidirectional_edge(graph, "N12", "S_1_1")
    add_bidirectional_edge(graph, "N11", "S_10_1")

    add_bidirectional_edge(graph, "N15", "F_1_1")
    add_bidirectional_edge(graph, "N16", "Q_1")
    add_bidirectional_edge(graph, "N10", "N16")

    for col in range(1, 13):
        add_bidirectional_edge(graph, f"F_10_{col}", f"Q_{col}")

    return graph, node_groups


def configure_plot_font():
    preferred_fonts = [
        "Noto Sans CJK SC",
        "Source Han Sans SC",
        "WenQuanYi Zen Hei",
        "Microsoft YaHei",
        "SimHei",
        "PingFang SC",
        "Arial Unicode MS",
    ]
    available_fonts = {font.name for font in font_manager.fontManager.ttflist}

    for font_name in preferred_fonts:
        if font_name in available_fonts:
            plt.rcParams["font.sans-serif"] = [font_name]
            plt.rcParams["axes.unicode_minus"] = False
            return "AGV 路径拓扑图与数学表达"

    plt.rcParams["axes.unicode_minus"] = False
    return "AGV Topology with Mathematical Expressions"


def draw_graph(graph, output_path="agv_complete_topology.png"):
    pos = nx.get_node_attributes(graph, "pos")
    title = configure_plot_font()

    plt.figure(figsize=(22, 17))

    nx.draw_networkx_edges(graph, pos, edge_color="lightgray", width=0.45, alpha=0.55)

    highlighted_edges = []
    for left, right in zip(
        ["N2", "C_GATE", "B_TEMP", "L2_4", "L2_3", "L2_2", "L2_1", "N1"],
        ["C_GATE", "B_TEMP", "L2_4", "L2_3", "L2_2", "L2_1", "N1", "N3"],
    ):
        if graph.has_edge(left, right):
            highlighted_edges.append((left, right))
        if graph.has_edge(right, left):
            highlighted_edges.append((right, left))

    main_lane_edges = []
    for index in range(1, 5):
        main_lane_edges.append((f"LANE_{index}_TOP", f"LANE_{index}_BOTTOM"))
        main_lane_edges.append((f"LANE_{index}_BOTTOM", f"LANE_{index}_TOP"))

    nx.draw_networkx_edges(
        graph,
        pos,
        edgelist=highlighted_edges,
        edge_color="red",
        width=2.2,
        alpha=0.9,
    )
    nx.draw_networkx_edges(
        graph,
        pos,
        edgelist=main_lane_edges,
        edge_color="blue",
        width=1.8,
        alpha=0.9,
    )

    node_colors = []
    for _, attrs in graph.nodes(data=True):
        group = attrs["group"]
        if group == "junctions":
            node_colors.append("#d62728")
        elif group in {"line1_nodes", "line2_nodes", "line3_nodes", "pack_nodes"}:
            node_colors.append("#2ca02c")
        elif "drop" in group:
            node_colors.append("#ff7f0e")
        elif group in {"storage_nodes", "finish_nodes", "defect_nodes"}:
            node_colors.append("#7f7f7f")
        elif group == "chargers":
            node_colors.append("#9467bd")
        else:
            node_colors.append("#1f77b4")

    nx.draw_networkx_nodes(graph, pos, node_size=22, node_color=node_colors, alpha=0.75)

    key_nodes = [
        "N1",
        "N2",
        "N3",
        "N5",
        "N7",
        "N8",
        "N9",
        "N10",
        "A_STOCK",
        "B_STOCK",
        "B_TEMP",
        "C_TEMP",
        "PACK_BUFFER",
    ]
    labels = {node_id: node_id for node_id in key_nodes if node_id in pos}
    nx.draw_networkx_labels(graph, pos, labels=labels, font_size=8, font_weight="bold")

    plt.title(title, fontsize=15)
    plt.axis("equal")
    plt.grid(True, linestyle=":", alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def build_math_description(graph):
    lines = [
        "# AGV 路径拓扑图数学表达",
        "",
        "## 1. 节点坐标表达",
        "",
        "- 主干线1端点：",
        "  - N1 = (-5, 32)",
        "  - N2 = (-60, 32)",
        "- 分支与主干连接节点：",
        "  - N3 = (-5, 12)",
        "  - N5 = (1, 12)",
        "  - N7 = (-5, 0)",
        "  - N8 = (-39, 0)",
        "  - N9 = (1, 34.0)",
        "  - N10 = (1, -31.8)",
        "  - N11 = (2.5, 34.5)",
        "  - N12 = (2.5, 2.5)",
        "  - N15 = (2.5, -0.3)",
        "  - N16 = (2.5, -32.3)",
        "",
        "- 装配线1节点与投放点（j = 1, 2, ..., 12）：",
        "  - L1_j = (-34 - 2(j-1), 32)",
        "  - D1_j = (-34 - 2(j-1), 32.5)",
        "",
        "- 装配线2节点与投放点（j = 1, 2, 3, 4）：",
        "  - L2_j = (-17 - 3(j-1), 32)",
        "  - D2_j = (-17 - 3(j-1), 31)",
        "",
        "- 装配线3节点与投放点（j = 1, 2, 3）：",
        "  - L3_j = (-43 - 8(j-1), 32)",
        "  - D3_j = (-43 - 8(j-1), 31)",
        "",
        "- 打包质检线节点（j = 1, 2, ..., 13）：",
        "  - P_j = (-20 - 2(j-1), 1)",
        "  - 对应 AGV 放料点与节点重合，即 D4_j ≡ P_j",
        "",
        "- 线边库存与暂存区节点：",
        "  - A_STOCK = (32.5, -60)",
        "  - B_STOCK = (30.5, -33)",
        "  - B_TEMP = (-27, 32)",
        "  - C_TEMP = (-38, 30.5)",
        "  - C_GATE = (-38, 32)",
        "  - PACK_BUFFER = (-36, 0)",
        "  - 不良品暂存与主干线2左端重合，即 DEFECT_BUFFER ≡ N8 = (-39, 0)",
        "",
        "- 三车道节点（i = 1, 2, 3, 4）：",
        "  - LANE_i_TOP = (1 + 0.5(i-1), 34.0)",
        "  - LANE_i_BOTTOM = (1 + 0.5(i-1), -31.8)",
        "",
        "- 充电位节点（j = 1, 2, ..., 10）：",
        "  - C_j_LOW = (4.2 + 3.2(j-1), 34)",
        "  - C_j_HIGH = (4.2 + 3.2(j-1), 35)",
        "",
        "- 物料存储区交叉节点（r = 1, 2, ..., 10；c = 1, 2, ..., 12）：",
        "  - S_{r,c} = (2.5 + 3.2(c-1), 2.5 + 3.2(r-1))",
        "",
        "- 成品存储区交叉节点（r = 1, 2, ..., 10；c = 1, 2, ..., 12）：",
        "  - F_{r,c} = (2.5 + 3.2(c-1), -0.3 - 3.2(r-1))",
        "",
        "- 不良品区底边节点（c = 1, 2, ..., 12）：",
        "  - Q_c = (2.5 + 3.2(c-1), -32.3)",
        "",
        "## 2. 路径集合表达",
        "",
        "- 生产区主干线1：R_1 = {(x, 32) | -60 <= x <= -5}",
        "- 分支线1：R_2 = {(-5, y) | 12 <= y <= 32}",
        "- 通道干路：R_3 = {(x, 12) | -5 <= x <= 1}",
        "- 生产区主干线2：R_4 = {(x, 0) | -39 <= x <= -5}",
        "- 打包质检线：R_5 = {(x, 1) | -44 <= x <= -20}",
        "",
        "- 三车道（i = 1, 2, 3, 4）：",
        "  - T_i = {(1 + 0.5(i-1), y) | -31.8 <= y <= 34.0}",
        "",
        "- 物料存储区纵向路径（c = 1, 2, ..., 12）：",
        "  - V_c^S = {(2.5 + 3.2(c-1), y) | 2.5 <= y <= 34.5}",
        "- 物料存储区横向路径（r = 1, 2, ..., 10）：",
        "  - H_r^S = {(x, 2.5 + 3.2(r-1)) | 2.5 <= x <= 40.9}",
        "",
        "- 成品区纵向路径（c = 1, 2, ..., 12）：",
        "  - V_c^F = {(2.5 + 3.2(c-1), y) | -32.3 <= y <= -0.3}",
        "- 成品区横向路径（r = 1, 2, ..., 10）：",
        "  - H_r^F = {(x, -0.3 - 3.2(r-1)) | 2.5 <= x <= 40.9}",
        "",
        "## 3. 路径代价数学表达",
        "",
        "- 若边 e = (u, v)，且节点坐标分别为 p_u = (x_u, y_u)、p_v = (x_v, y_v)，",
        "  - 距离：d_uv = sqrt((x_u - x_v)^2 + (y_u - y_v)^2)",
        f"  - 速度：v = {AGV_SPEED_MPS} m/s",
        "  - 行驶时间：t_uv = d_uv / v = 2 d_uv",
        f"  - 单位距离能耗：eta = {ENERGY_RATIO_PER_M}",
        "  - 边能耗：E_uv = eta * d_uv",
        "",
        "## 4. 图论表达",
        "",
        f"- 本拓扑可表示为有向图 G = (V, E)，其中 |V| = {graph.number_of_nodes()}，|E| = {graph.number_of_edges()}。",
        "- 任意边均附带以下属性：",
        "  - length_m：边长度（m）",
        "  - travel_time_s：在 0.5 m/s 下的通行时间（s）",
        "  - energy_ratio：按每米万分之五估算的电量消耗占比",
        "",
    ]
    return "\n".join(lines)


def save_math_description(graph, output_path="agv_topology_math.md"):
    with open(output_path, "w", encoding="utf-8") as file_obj:
        file_obj.write(build_math_description(graph))


def main():
    graph, _ = build_graph()
    draw_graph(graph)
    save_math_description(graph)

    print(f"节点总数: {graph.number_of_nodes()}")
    print(f"边总数: {graph.number_of_edges()}")
    print("拓扑图已保存为: agv_complete_topology.png")
    print("数学表达已保存为: agv_topology_math.md")


if __name__ == "__main__":
    main()
