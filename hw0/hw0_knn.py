'''
K Nearest Neighbors

>>> data_NF = np.asarray([
...     [1, 0],
...     [0, 1],
...     [-1, 0],
...     [0, -1]])
>>> query_QF = np.asarray([
...     [0.9, 0],
...     [0, -0.9]])

Example Test K=1
----------------
# Find the single nearest neighbor for each query vector
>>> neighb_QKF = calc_k_nearest_neighbors(data_NF, query_QF, K=1)
>>> neighb_QKF.shape
(2, 1, 2)

# Neighbor of [0.9, 0]
>>> neighb_QKF[0]
array([[1., 0.]])

# Neighbor of [0, -0.9]
>>> neighb_QKF[1]
array([[ 0., -1.]])

Example Test K=3
----------------
# Now find 3 nearest neighbors for the same queries
>>> neighb_QKF = calc_k_nearest_neighbors(data_NF, query_QF, K=3)
>>> neighb_QKF.shape
(2, 3, 2)

# Neighbor of [0.9, 0]
>>> neighb_QKF[0]
array([[ 1.,  0.],
       [ 0.,  1.],
       [ 0., -1.]])

# Neighbor of [0, -0.9]
>>> neighb_QKF[1]
array([[ 0., -1.],
       [ 1.,  0.],
       [-1.,  0.]])
'''
import numpy as np


# 计算两个向量之间的欧几里德距离
def distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))


def calc_k_nearest_neighbors(data_NF, query_QF, K=1):
    ''' Compute and return k-nearest neighbors under Euclidean distance

    Args
    ----
    data_NF : 2D np.array, shape = (n_examples, n_feats) == (N, F)
        Each row is a feature vector for one example in dataset
    query_QF : 2D np.array, shape = (n_queries, n_feats) == (Q, F)
        Each row is a feature vector whose neighbors we want to find
    K : int, must satisfy K >= 1 and K <= n_examples aka N
        Number of neighbors to find per query vector

    Returns
    -------
    neighb_QKF : 3D np.array, (n_queries, n_neighbors, n_feats) == (Q, K, F)
        Entry q,k is feature vector of k-th nearest neighbor of the q-th query
        If two vectors are equally close, then we break ties by taking the one
        appearing first in row order in the original data_NF array
    '''

    # TODO fixme
    """
      计算查询向量的K个最近邻
    """
    # 查询集个数
    num_queries = query_QF.shape[0]
    # 数据集个数
    num_data = data_NF.shape[0]

    # 初始化最近邻索引数组
    neighbours = np.zeros((num_queries, K, data_NF.shape[1]))

    for i in range(num_queries):
        query_vector = query_QF[i]
        distances = np.zeros(num_data)

        # 计算查询点与每个数据点的距离
        for j in range(num_data):
            distances[j] = distance(query_vector, data_NF[j])

        # 找到K个最近邻的索引
        k_nearest_indices = np.argsort(distances)[:K]

        # 提取K个最近邻的数据点
        k_nearest_neighbours = data_NF[k_nearest_indices]

        # 存储在结果数组中
        neighbours[i] = k_nearest_neighbours


    return neighbours


if __name__ == "__main__":
    data_NF = np.asarray([
        [1, 0],
        [0, 1],
        [-1, 0],
        [0, -1]])

    query_QF = np.asarray([
        [0.9, 0],
        [0, -0.9]])

    neighb_QKF = calc_k_nearest_neighbors(data_NF, query_QF, K=3)
    print(neighb_QKF.shape)

    for i in range(query_QF.shape[0]):
        print(query_QF[i]+":")
        print(neighb_QKF[i])
