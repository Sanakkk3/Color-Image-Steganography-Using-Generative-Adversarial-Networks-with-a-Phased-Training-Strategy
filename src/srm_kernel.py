import numpy as np

filter_class_3 = [
  np.array([
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
    [0, 0, -1, 1, 0],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0]
  ], dtype=np.float32),
  np.array([
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
    [0, 1, -1, 0, 0],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0]
  ], dtype=np.float32),
  np.array([
    [0, 0, 0, 0, 0],
    [0, 0, 1, 0, 0],
    [0, 0, -1, 0, 0],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0]
  ], dtype=np.float32),
  np.array([
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
    [0, 0, -1, 0, 0],
    [0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0]
  ], dtype=np.float32),
  np.array([
    [0, 0, 0, 0, 0],
    [0, 0, 0, 1, 0],
    [0, 0, -1, 0, 0],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0]
  ], dtype=np.float32),
  np.array([
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
    [0, 0, -1, 0, 0],
    [0, 0, 0, 1, 0],
    [0, 0, 0, 0, 0]
  ], dtype=np.float32),
  np.array([
    [0, 0, 0, 0, 0],
    [0, 1, 0, 0, 0],
    [0, 0, -1, 0, 0],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0]
  ], dtype=np.float32),
  np.array([
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
    [0, 0, -1, 0, 0],
    [0, 1, 0, 0, 0],
    [0, 0, 0, 0, 0]
  ], dtype=np.float32),
  np.array([
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
    [0, 0.5, -1, 0.5, 0],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0]
  ], dtype=np.float32),
  np.array([
    [0, 0, 0, 0, 0],
    [0, 0, 0.5, 0, 0],
    [0, 0, -1, 0, 0],
    [0, 0, 0.5, 0, 0],
    [0, 0, 0, 0, 0]
  ], dtype=np.float32),
  np.array([
    [0, 0, 0, 0, 0],
    [0, 0.5, 0, 0, 0],
    [0, 0, -1, 0, 0],
    [0, 0, 0, 0.5, 0],
    [0, 0, 0, 0, 0]
  ], dtype=np.float32),
  np.array([
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0.5, 0],
    [0, 0, -1, 0, 0],
    [0, 0.5, 0, 0, 0],
    [0, 0, 0, 0, 0]
  ], dtype=np.float32)
]


filter_edge_5x5 = [
  np.array([
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
    [0, 1, -3, 3, -1],
    [0, 0.5, 0, 0, 0],
    [0, 0, 0, 0, 0]
  ], dtype=np.float32),  # /3
  np.array([
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
    [-1, 3, -3, 1, 0],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0]
  ], dtype=np.float32),
  np.array([
    [0, 0, -1, 0, 0],
    [0, 0, 3, 0, 0],
    [0, 0, -3, 0, 0],
    [0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0]
  ], dtype=np.float32),
  np.array([
    [0, 0, 0, 0, 0],
    [0, 0, 1, 0, 0],
    [0, 0, -3, 0, 0],
    [0, 0, 3, 0, 0],
    [0, 0, -1, 0, 0]
  ], dtype=np.float32),
  np.array([
    [0, 0, 0, 0, -1],
    [0, 0, 0, 3, 0],
    [0, 0, -3, 0, 0],
    [0, 1, 0, 0, 0],
    [0, 0, 0, 0, 0]
  ], dtype=np.float32),
  np.array([
    [0, 0, 0, 0, 0],
    [0, 1, 0, 0, 0],
    [0, 0, -3, 0, 0],
    [0, 0, 0, 3, 0],
    [0, 0, 0, 0, -1]
  ], dtype=np.float32),
  np.array([
    [-1, 0, 0, 0, 0],
    [0, 3, 0, 0, 0],
    [0, 0, -3, 0, 0],
    [0, 0, 0, 3, 0],
    [0, 0, 0, 0, 0]
  ], dtype=np.float32),
  np.array([
    [0, 0, 0, 0, 0],
    [0, 0, 0, 1, 0],
    [0, 0, -3, 0, 0],
    [0, 3, 0, 0, 0],
    [-1, 0, 0, 0, 0]
  ], dtype=np.float32)
]

square_5x5 = [     # /4
  np.array([
    [0, 0, 0, 0, 0],
    [0, 0, 2, -1, 0],
    [0, 0, -4, 2, 0],
    [0, 0, 2, -1, 0],
    [0, 0, 0, 0, 0]
  ], dtype=np.float32),
  np.array([
    [0, 0, 0, 0, 0],
    [0, -1, 2, 0, 0],
    [0, 2, -4, 0, 0],
    [0, -1, 2, 0, 0],
    [0, 0, 0, 0, 0]
  ], dtype=np.float32),
  np.array([
    [0, 0, 0, 0, 0],
    [0, -1, 2, -1, 0],
    [0, 2, -4, 2, 0],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0]
  ], dtype=np.float32),
  np.array([
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
    [0, 2, -4, 2, 0],
    [0, -1, 2, -1, 0],
    [0, 0, 0, 0, 0]
  ], dtype=np.float32),
  np.array([
    [0, 0, 0, 0, 0],
    [0, -1, 2, -1, 0],
    [0, 2, -4, 2, 0],
    [0, -1, 2, -1, 0],
    [0, 0, 0, 0, 0]
  ], dtype=np.float32),
]

filter_class_5x5 = [     # /6
  np.array([
    [0, 0, -1, 1, -0.5],
    [0, 0, 4, -3, 1],
    [0, 0, -6, 4, -1],
    [0, 0, 4, -3, 1],
    [0, 0, -1, 1, -0.5]
  ], dtype=np.float32),
  np.array([
    [-0.5, 1, -1, 0, 0],
    [1, -3, 4, 0, 0],
    [-1, 4, -6, 0, 0],
    [1, -3, 4, 0, 0],
    [-0.5, 1, -1, 0, 0]
  ], dtype=np.float32),
  np.array([
    [-0.5, 1, -1, 1, -0.5],
    [1, -3, 4, -3, 1],
    [-1, 4, -6, 4, -1],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0]
  ], dtype=np.float32),
  np.array([
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
    [-1, 4, -6, 4, -1],
    [1, -3, 4, -3, 1],
    [-0.5, 1, -1, 1, -0.5]
  ], dtype=np.float32),
  np.array([
    [-0.5, 1, -1, 1, -0.5],
    [1, -3, 4, -3, 1],
    [-1, 4, -6, 4, -1],
    [1, -3, 4, -3, 1],
    [-0.5, 1, -1, 1, -0.5]
  ], dtype=np.float32),
]


all_hpf_list = filter_class_3 + filter_edge_5x5 + square_5x5 + filter_class_5x5

hpf_5x5_list = filter_class_3 + filter_edge_5x5 + square_5x5 + filter_class_5x5

normalized_filter_class_3 = filter_class_3
normalized_filter_edge_5x5 = [hpf / 3 for hpf in filter_edge_5x5]
normalized_square_5x5 = [hpf / 4 for hpf in square_5x5]
normalized_class_5x5 = [hpf / 6 for hpf in filter_class_5x5]

all_normalized_hpf_list = normalized_filter_class_3 + normalized_filter_edge_5x5 + normalized_square_5x5 \
  + normalized_class_5x5

# normalized_hpf_5x5_list = normalized_filter_class_3 + normalized_filter_edge_5x5 + [normalized_square_5x5]

# normalized_5x5_list = normalized_filter_edge_5x5 + [normalized_square_5x5]