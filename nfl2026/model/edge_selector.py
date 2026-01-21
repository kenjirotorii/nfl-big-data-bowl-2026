from jaxtyping import Bool, Float, Int
from torch import Tensor
from torch_cluster import knn, knn_graph, nearest, radius, radius_graph

from ..data.dataclass import EdgeArray


class EdgeSelector:
    def __call__(
        self,
        target_position: Float[Tensor, "... XY"],
        target_partition: Int[Tensor, "..."],
        target_valid: Bool[Tensor, "..."] | None,
        source_position: Float[Tensor, "... XY"],
        source_partition: Int[Tensor, "..."],
        source_valid: Bool[Tensor, "..."] | None,
    ) -> EdgeArray:
        raise NotImplementedError("This method should be implemented by subclasses.")


class RadiusEdgeSelector(EdgeSelector):
    def __init__(self, search_radius: float, max_num_neighbors: int) -> None:
        self.search_radius = search_radius
        self.max_num_neighbors = max_num_neighbors

    def __call__(
        self,
        target_position: Float[Tensor, "... XY"],
        target_partition: Int[Tensor, "..."],
        target_valid: Bool[Tensor, "..."] | None,
        source_position: Float[Tensor, "... XY"],
        source_partition: Int[Tensor, "..."],
        source_valid: Bool[Tensor, "..."] | None,
    ) -> EdgeArray:
        edge_index_t2s: Tensor = radius(
            x=source_position,
            y=target_position,
            r=self.search_radius,
            batch_x=source_partition,
            batch_y=target_partition,
            max_num_neighbors=self.max_num_neighbors,
        )
        if target_valid is not None:
            edge_index_t2s = edge_index_t2s[:, target_valid[edge_index_t2s[0]]]
        if source_valid is not None:
            edge_index_t2s = edge_index_t2s[:, source_valid[edge_index_t2s[1]]]
        return edge_index_t2s


class KNNEdgeSelector(EdgeSelector):
    def __init__(self, k: int) -> None:
        self.k = k

    def __call__(
        self,
        target_position: Float[Tensor, "... XY"],
        target_partition: Int[Tensor, "..."],
        target_valid: Bool[Tensor, "..."] | None,
        source_position: Float[Tensor, "... XY"],
        source_partition: Int[Tensor, "..."],
        source_valid: Bool[Tensor, "..."] | None,
    ) -> EdgeArray:
        edge_index_t2s: Tensor = knn(
            x=source_position,
            y=target_position,
            k=self.k,
            batch_x=source_partition,
            batch_y=target_partition,
        )
        if target_valid is not None:
            edge_index_t2s = edge_index_t2s[:, target_valid[edge_index_t2s[0]]]
        if source_valid is not None:
            edge_index_t2s = edge_index_t2s[:, source_valid[edge_index_t2s[1]]]
        return edge_index_t2s


class NearestEdgeSelector(EdgeSelector):
    def __call__(
        self,
        target_position: Float[Tensor, "... XY"],
        target_partition: Int[Tensor, "..."],
        target_valid: Bool[Tensor, "..."] | None,
        source_position: Float[Tensor, "... XY"],
        source_partition: Int[Tensor, "..."],
        source_valid: Bool[Tensor, "..."] | None,
    ) -> EdgeArray:
        edge_index_t2s: Tensor = nearest(
            x=source_position,
            y=target_position,
            batch_x=source_partition,
            batch_y=target_partition,
        )
        if target_valid is not None:
            edge_index_t2s = edge_index_t2s[:, target_valid[edge_index_t2s[0]]]
        if source_valid is not None:
            edge_index_t2s = edge_index_t2s[:, source_valid[edge_index_t2s[1]]]
        return edge_index_t2s


class KNNGraphEdgeSelector(EdgeSelector):
    def __init__(self, k: int) -> None:
        self.k = k

    def __call__(
        self,
        target_position: Float[Tensor, "... XY"],
        target_partition: Int[Tensor, "..."],
        target_valid: Bool[Tensor, "..."] | None,
        source_position: Float[Tensor, "... XY"],
        source_partition: Int[Tensor, "..."],
        source_valid: Bool[Tensor, "..."] | None,
    ) -> EdgeArray:
        edge_index_s2t: Tensor = knn_graph(
            x=target_position, k=self.k, batch=target_partition
        )
        if target_valid is not None:
            edge_index_s2t = edge_index_s2t[:, target_valid[edge_index_s2t[1]]]
        return edge_index_s2t


class RadiusGraphEdgeSelector(EdgeSelector):
    def __init__(self, search_radius: float, max_num_neighbors: int) -> None:
        self.search_radius = search_radius
        self.max_num_neighbors = max_num_neighbors

    def __call__(
        self,
        target_position: Float[Tensor, "... XY"],
        target_partition: Int[Tensor, "..."],
        target_valid: Bool[Tensor, "..."] | None,
        source_position: Float[Tensor, "... XY"],
        source_partition: Int[Tensor, "..."],
        source_valid: Bool[Tensor, "..."] | None,
    ) -> EdgeArray:
        edge_index_s2t: Tensor = radius_graph(
            x=target_position,
            r=self.search_radius,
            batch=target_partition,
            max_num_neighbors=self.max_num_neighbors,
        )
        if target_valid is not None:
            edge_index_s2t = edge_index_s2t[:, target_valid[edge_index_s2t[1]]]
        return edge_index_s2t
