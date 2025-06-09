import numpy as np
import torch
from collections import defaultdict
from typing import List


def compute_similarity_matrix(embeddings: torch.Tensor) -> np.ndarray:
    """
    Compute cosine similarity between all embeddings.

    Args:
        embeddings (Tensor): (N, D) normalized image embeddings

    Returns:
        np.ndarray: (N, N) similarity matrix
    """
    return (embeddings @ embeddings.T).cpu().numpy()


def group_duplicates(sim_matrix: np.ndarray, paths: List[str], threshold: float = 0.95) -> List[List[str]]:
    """
    Group duplicates using Union-Find based on cosine similarity.

    Args:
        sim_matrix (np.ndarray): Cosine similarity matrix
        paths (List[str]): Corresponding image paths
        threshold (float): Cosine similarity threshold to consider as duplicate

    Returns:
        List[List[str]]: Groups of duplicate file paths
    """
    parent = {}

    def find(x):
        if parent.get(x, x) != x:
            parent[x] = find(parent[x])
        return parent.get(x, x)

    def union(x, y):
        px, py = find(x), find(y)
        if px != py:
            parent[py] = px

    n = len(paths)
    for i in range(n):
        for j in range(i + 1, n):
            if sim_matrix[i, j] >= threshold:
                union(paths[i], paths[j])

    groups = defaultdict(list)
    for path in paths:
        root = find(path)
        groups[root].append(path)

    return [group for group in groups.values() if len(group) > 1]