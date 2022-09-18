import unittest
import gc
import operator as op
import functools
import torch
from torch.autograd import Variable, Function
import knn_pytorch


class KNearestNeighbor(Function):
    """ Compute k nearest neighbors for each query point.
    """

    # 改为静态方法,ref:https://ldzhangyx.github.io/2019/12/04/pytorch-skill-1204/
    @staticmethod
    def forward(self, ref, query, k):
        """
        Args:
            ref: b x dim x n_ref
            query: b x dim x n_query
            k: number of matches
        """
        ref = ref.contiguous().float().cuda()
        query = query.contiguous().float().cuda()
        inds = torch.empty(query.shape[0], k, query.shape[2]).long().cuda()

        knn_pytorch.knn(ref, query, inds)

        return inds


class TestKNearestNeighbor(unittest.TestCase):
    def test_forward(self):
        k_nearest = 2
        while(1):
            D, N, M = 128, 100, 1000
            ref = Variable(torch.rand(2, D, N))
            query = Variable(torch.rand(2, D, M))

            inds = KNearestNeighbor.apply(ref, query,k_nearest)
            for obj in gc.get_objects():
                if torch.is_tensor(obj):
                    print(functools.reduce(op.mul, obj.size()) if len(obj.size()) > 0 else 0, type(obj), obj.size())
            # ref = ref.cpu()
            # query = query.cpu()
            print(inds)


if __name__ == '__main__':
    unittest.main()
