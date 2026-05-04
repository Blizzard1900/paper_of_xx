from __future__ import annotations

from typing import Dict, Iterable, Tuple

import networkx as nx

from models import AGV, Edge, GlobalParams, Instance, Node, Task


def build_global_params() -> GlobalParams:
    return GlobalParams()


def _add_node(
    nodes: Dict[str, Node],
    node_id: str,
    node_type: str,
    region: str,
    x: float,
    y: float,
    *,
    service_capable: bool = False,
) -> None:
    nodes[node_id] = Node(
        node_id=node_id,
        node_type=node_type,
        region=region,
        x=x,
        y=y,
        service_capable=service_capable,
    )


def _generate_linear_nodes(
    nodes: Dict[str, Node],
    prefix: str,
    start_x: float,
    start_y: float,
    step_x: float,
    count: int,
    node_type: str,
    region: str,
) -> None:
    for index in range(1, count + 1):
        _add_node(
            nodes,
            f"{prefix}_{index}",
            node_type,
            region,
            start_x + step_x * (index - 1),
            start_y,
            service_capable=True,
        )


def load_nodes(params: GlobalParams | None = None) -> Dict[str, Node]:
    _ = params or build_global_params()
    nodes: Dict[str, Node] = {}

    _add_node(nodes, "J1", "junction", "road", -5.0, 32.0)
    _add_node(nodes, "J2", "junction", "road", -60.0, 32.0)
    _add_node(nodes, "J3", "junction", "road", -5.0, 12.0)
    _add_node(nodes, "J4", "junction", "road", 1.0, 12.0)
    _add_node(nodes, "J5", "junction", "road", -5.0, 0.0)
    _add_node(nodes, "J6", "junction", "road", -39.0, 0.0)
    _add_node(nodes, "J7", "junction", "road", 2.5, 34.5)
    _add_node(nodes, "J8", "junction", "road", 2.5, 2.5)
    _add_node(nodes, "J9", "junction", "road", 2.5, -0.3)
    _add_node(nodes, "J10", "junction", "road", 2.5, -31.8)

    _generate_linear_nodes(nodes, "L1", -34.0, 32.5, -2.0, 13, "line1", "production")
    _generate_linear_nodes(nodes, "L2", -17.0, 31.0, -3.0, 5, "line2", "production")
    _generate_linear_nodes(nodes, "L3", -33.0, 31.0, -8.0, 3, "line3", "production")
    _generate_linear_nodes(nodes, "L4", -20.0, 0.0, -2.0, 9, "line4", "production")

    _add_node(nodes, "A", "buffer", "buffer", 32.5, -60.0, service_capable=True)
    _add_node(nodes, "B", "buffer", "buffer", 30.5, -27.0, service_capable=True)
    _add_node(nodes, "C", "buffer", "buffer", 31.0, -33.0, service_capable=True)
    _add_node(nodes, "Bf", "buffer", "buffer", -36.0, 0.0, service_capable=True)
    _add_node(nodes, "Df", "buffer", "buffer", -39.0, 0.0, service_capable=True)

    for index in range(1, 11):
        _add_node(
            nodes,
            f"C{index}",
            "charger",
            "charging",
            10.5 + 1.5 * (index - 1),
            43.5,
        )

    for index in range(1, 4):
        lane_x = 1.0 + 0.5 * (index - 1)
        _add_node(nodes, f"LANE_{index}_TOP", "lane", "lane", lane_x, 34.0)
        _add_node(nodes, f"LANE_{index}_BOTTOM", "lane", "lane", lane_x, -31.8)

    for row in range(1, 11):
        for col in range(1, 13):
            _add_node(
                nodes,
                f"S_{row}_{col}",
                "storage",
                "storage",
                4.2 + 3.2 * (col - 1),
                5.2 + 3.2 * (row - 1),
                service_capable=True,
            )

    for row in range(1, 10):
        for col in range(1, 13):
            _add_node(
                nodes,
                f"F_{row}_{col}",
                "finish",
                "finish",
                4.2 + 3.2 * (col - 1),
                -3.0 - 3.2 * (row - 1),
                service_capable=True,
            )

    for col in range(1, 13):
        _add_node(
            nodes,
            f"D_{col}",
            "defect",
            "defect",
            4.2 + 3.2 * (col - 1),
            -31.8,
            service_capable=True,
        )

    return nodes


def _manhattan_distance(source: Node, target: Node) -> float:
    return abs(source.x - target.x) + abs(source.y - target.y)


def _resource_id(source: str, target: str, edge_type: str) -> str:
    if edge_type == "charger_link":
        return f"charger-chain:{tuple(sorted((source, target)))}"
    if edge_type.startswith("lane"):
        return f"lane:{tuple(sorted((source, target)))}"
    return f"edge:{source}->{target}"


def _make_edge(
    nodes: Dict[str, Node],
    params: GlobalParams,
    source: str,
    target: str,
    *,
    edge_type: str,
    capacity: int = 1,
    is_production_edge: bool = False,
) -> Edge:
    distance = _manhattan_distance(nodes[source], nodes[target])
    return Edge(
        edge_id=f"E_{source}__{target}",
        from_node=source,
        to_node=target,
        distance=round(distance, 4),
        base_time=round(distance / params.speed_mps, 4),
        energy_cost=round(distance * params.energy_kwh_per_m, 6),
        capacity=capacity,
        edge_type=edge_type,
        is_production_edge=is_production_edge,
        resource_id=_resource_id(source, target, edge_type),
    )


def _add_bidirectional_edge(
    edge_map: Dict[Tuple[str, str], Edge],
    nodes: Dict[str, Node],
    params: GlobalParams,
    source: str,
    target: str,
    *,
    edge_type: str,
    capacity: int = 1,
    is_production_edge: bool = False,
) -> None:
    edge_map[(source, target)] = _make_edge(
        nodes,
        params,
        source,
        target,
        edge_type=edge_type,
        capacity=capacity,
        is_production_edge=is_production_edge,
    )
    edge_map[(target, source)] = _make_edge(
        nodes,
        params,
        target,
        source,
        edge_type=edge_type,
        capacity=capacity,
        is_production_edge=is_production_edge,
    )


def _connect_chain(
    edge_map: Dict[Tuple[str, str], Edge],
    nodes: Dict[str, Node],
    params: GlobalParams,
    ordered_nodes: Iterable[str],
    *,
    edge_type: str,
    capacity: int = 1,
    is_production_edge: bool = False,
) -> None:
    node_list = list(ordered_nodes)
    for left, right in zip(node_list, node_list[1:]):
        _add_bidirectional_edge(
            edge_map,
            nodes,
            params,
            left,
            right,
            edge_type=edge_type,
            capacity=capacity,
            is_production_edge=is_production_edge,
        )


def load_edges(
    nodes: Dict[str, Node],
    params: GlobalParams | None = None,
) -> Dict[Tuple[str, str], Edge]:
    params = params or build_global_params()
    edge_map: Dict[Tuple[str, str], Edge] = {}

    _connect_chain(
        edge_map,
        nodes,
        params,
        ["J2"] + [f"L1_{idx}" for idx in range(13, 0, -1)] + ["J1"],
        edge_type="main_road",
        is_production_edge=True,
    )
    _add_bidirectional_edge(edge_map, nodes, params, "J1", "J3", edge_type="branch_road", is_production_edge=True)
    _add_bidirectional_edge(edge_map, nodes, params, "J3", "J4", edge_type="access_road", is_production_edge=True)
    _add_bidirectional_edge(edge_map, nodes, params, "J3", "J5", edge_type="branch_road", is_production_edge=True)
    _connect_chain(
        edge_map,
        nodes,
        params,
        ["J6", "Bf", "J5"],
        edge_type="main_road",
        is_production_edge=True,
    )

    for node_id in [f"L2_{idx}" for idx in range(1, 6)]:
        _add_bidirectional_edge(edge_map, nodes, params, node_id, "J1", edge_type="service_link", is_production_edge=True)
    for node_id in [f"L3_{idx}" for idx in range(1, 4)]:
        _add_bidirectional_edge(edge_map, nodes, params, node_id, "J1", edge_type="service_link", is_production_edge=True)
    _connect_chain(
        edge_map,
        nodes,
        params,
        [f"L4_{idx}" for idx in range(1, 10)],
        edge_type="service_line",
        is_production_edge=True,
    )
    _add_bidirectional_edge(edge_map, nodes, params, "L4_1", "J5", edge_type="service_link", is_production_edge=True)

    _add_bidirectional_edge(edge_map, nodes, params, "A", "J10", edge_type="buffer_link")
    _add_bidirectional_edge(edge_map, nodes, params, "B", "J10", edge_type="buffer_link")
    _add_bidirectional_edge(edge_map, nodes, params, "C", "J10", edge_type="buffer_link")
    _add_bidirectional_edge(edge_map, nodes, params, "Bf", "J5", edge_type="buffer_link", is_production_edge=True)
    _add_bidirectional_edge(edge_map, nodes, params, "Df", "J6", edge_type="buffer_link", is_production_edge=True)

    _add_bidirectional_edge(edge_map, nodes, params, "J4", "LANE_1_TOP", edge_type="lane_access")
    _add_bidirectional_edge(edge_map, nodes, params, "J10", "LANE_1_BOTTOM", edge_type="lane_access")
    for index in range(1, 4):
        _add_bidirectional_edge(
            edge_map,
            nodes,
            params,
            f"LANE_{index}_TOP",
            f"LANE_{index}_BOTTOM",
            edge_type="lane",
            capacity=params.lane_capacity,
        )
    for index in range(1, 3):
        _add_bidirectional_edge(
            edge_map,
            nodes,
            params,
            f"LANE_{index}_TOP",
            f"LANE_{index + 1}_TOP",
            edge_type="lane_top_link",
        )
        _add_bidirectional_edge(
            edge_map,
            nodes,
            params,
            f"LANE_{index}_BOTTOM",
            f"LANE_{index + 1}_BOTTOM",
            edge_type="lane_bottom_link",
        )

    _connect_chain(
        edge_map,
        nodes,
        params,
        [f"C{index}" for index in range(1, 11)],
        edge_type="charger_link",
    )
    _add_bidirectional_edge(edge_map, nodes, params, "C1", "J7", edge_type="charger_access")
    _add_bidirectional_edge(edge_map, nodes, params, "J7", "J4", edge_type="storage_to_road")

    for row in range(1, 11):
        _connect_chain(
            edge_map,
            nodes,
            params,
            [f"S_{row}_{col}" for col in range(1, 13)],
            edge_type="storage_grid",
        )
    for col in range(1, 13):
        _connect_chain(
            edge_map,
            nodes,
            params,
            [f"S_{row}_{col}" for row in range(1, 11)],
            edge_type="storage_grid",
        )
    _add_bidirectional_edge(edge_map, nodes, params, "J8", "S_1_1", edge_type="storage_access")
    _add_bidirectional_edge(edge_map, nodes, params, "J7", "S_10_1", edge_type="storage_access")

    for row in range(1, 10):
        _connect_chain(
            edge_map,
            nodes,
            params,
            [f"F_{row}_{col}" for col in range(1, 13)],
            edge_type="finish_grid",
        )
    for col in range(1, 13):
        _connect_chain(
            edge_map,
            nodes,
            params,
            [f"F_{row}_{col}" for row in range(1, 10)],
            edge_type="finish_grid",
        )
    _connect_chain(
        edge_map,
        nodes,
        params,
        [f"D_{col}" for col in range(1, 13)],
        edge_type="defect_grid",
    )
    for col in range(1, 13):
        _add_bidirectional_edge(edge_map, nodes, params, f"F_9_{col}", f"D_{col}", edge_type="defect_link")
    _add_bidirectional_edge(edge_map, nodes, params, "J9", "F_1_1", edge_type="finish_access")
    _add_bidirectional_edge(edge_map, nodes, params, "J10", "D_1", edge_type="defect_access")
    _add_bidirectional_edge(edge_map, nodes, params, "J10", "J9", edge_type="finish_to_road")

    return edge_map


def build_graph(nodes: Dict[str, Node], edges: Dict[Tuple[str, str], Edge]) -> nx.DiGraph:
    graph = nx.DiGraph()
    for node in nodes.values():
        graph.add_node(
            node.node_id,
            x=node.x,
            y=node.y,
            node_type=node.node_type,
            region=node.region,
            service_capable=node.service_capable,
        )
    for edge in edges.values():
        graph.add_edge(
            edge.from_node,
            edge.to_node,
            distance=edge.distance,
            base_time=edge.base_time,
            energy_cost=edge.energy_cost,
            capacity=edge.capacity,
            edge_type=edge.edge_type,
            is_production_edge=edge.is_production_edge,
            resource_id=edge.resource_id,
        )
    return graph


def load_tasks() -> Dict[str, Task]:
    return {
        "T1": Task(
            task_id="T1",
            task_type="raw_material",
            pickup_node="S_1_1",
            drop_node="L1_1",
            qty=20.0,
            earliest_start=0.0,
            latest_finish=2000.0,
            pickup_service_time=30.0,
            drop_service_time=30.0,
        ),
        "T2": Task(
            task_id="T2",
            task_type="raw_material",
            pickup_node="S_2_3",
            drop_node="L2_2",
            qty=15.0,
            earliest_start=0.0,
            latest_finish=2200.0,
            pickup_service_time=30.0,
            drop_service_time=30.0,
        ),
        "T3": Task(
            task_id="T3",
            task_type="finished_goods",
            pickup_node="L4_2",
            drop_node="F_1_2",
            qty=10.0,
            earliest_start=200.0,
            latest_finish=2600.0,
            pickup_service_time=20.0,
            drop_service_time=20.0,
        ),
        "T4": Task(
            task_id="T4",
            task_type="defect_transfer",
            pickup_node="L4_4",
            drop_node="D_3",
            qty=8.0,
            earliest_start=300.0,
            latest_finish=2600.0,
            pickup_service_time=20.0,
            drop_service_time=20.0,
        ),
    }


def load_agvs(params: GlobalParams | None = None) -> Dict[str, AGV]:
    params = params or build_global_params()
    return {
        "AGV_1": AGV(
            agv_id="AGV_1",
            start_node="C1",
            battery_init=6.0,
            battery_max=params.battery_capacity_kwh,
            battery_threshold=params.battery_threshold_kwh,
            battery_min=params.battery_min_kwh,
            charge_power=params.charge_power_kw,
            load_capacity=100.0,
            status="idle",
        ),
        "AGV_2": AGV(
            agv_id="AGV_2",
            start_node="J5",
            battery_init=3.2,
            battery_max=params.battery_capacity_kwh,
            battery_threshold=params.battery_threshold_kwh,
            battery_min=params.battery_min_kwh,
            charge_power=params.charge_power_kw,
            load_capacity=100.0,
            status="idle",
        ),
    }


def build_instance() -> Instance:
    params = build_global_params()
    nodes = load_nodes(params)
    edges = load_edges(nodes, params)
    graph = build_graph(nodes, edges)
    tasks = load_tasks()
    agvs = load_agvs(params)
    return Instance(
        global_params=params,
        nodes=nodes,
        edges=edges,
        tasks=tasks,
        agvs=agvs,
        graph=graph,
        chargers=[f"C{index}" for index in range(1, 11)],
        service_nodes=[node_id for node_id, node in nodes.items() if node.service_capable],
    )


def build_default_instance() -> Instance:
    return build_instance()
