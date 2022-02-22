import numpy

print("yes")
print("test ignore.file")
print("test idea")
from torch.utils.data import Dataset
from torchvision import transforms
import test
test.test_package()
import torch



# dtype=torch.float32
# device=torch.device("cpu")
#
# for sizes, aspect_ratios in zip(anchor_sizes, aspect_ratios):
#     print('')
#     scales = torch.as_tensor(sizes, dtype=dtype, device=device)
#     aspect_ratios = torch.as_tensor(aspect_ratios, dtype=dtype, device=device)
#     h_ratios = torch.sqrt(aspect_ratios)
#     w_ratios = 1.0 / h_ratios
#
#     # [r1, r2, r3]' * [s1, s2, s3]
#     # number of elements is len(ratios)*len(scales)
#     ws = (w_ratios[:, None] * scales[None, :]).view(-1)
#     hs = (h_ratios[:, None] * scales[None, :]).view(-1)
#     base_anchors = torch.stack([-ws, -hs, ws, hs], dim=1) / 2


# c = torch.as_tensor([[1,2],[2,3],[4,5]])
# a = [torch.as_tensor([1,2]),torch.as_tensor([3,4]),torch.as_tensor([5,6])]
# print(a)
# b = torch.cat(a,dim = 1)
# print(b)



# a = torch.randn((4,3))
# print(a)
# print(a[[[1],[2]],[[1,2,0],[1,2,0]]])




class TestDataSet(torch.utils.data.Dataset):
    def __init__(self,data_name):
        self.name = data_name
        self.data = ['a', 'b','c' ,'d', 'e', 'f', 'g']

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        return  self.data[idx]

testDataSet = TestDataSet('test')
torch.utils.data.DataLoader
import bisect
x = [1,2,3,4,5,6]
b = 4
print(bisect.bisect_right(x,b))
print(x)