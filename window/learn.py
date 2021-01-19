import torch
# t = torch.randn(3, 4, 5)
# t = torch.randn(12)
#
# print(t.view(3, 4))
# print(t.data_ptr())
# print(t.view(4, 3))
# print(t.data_ptr())
#
# print(t.view(3, 4).transpose(0, 1).contiguous())

# t = torch.randn(2, 3, 4)
# print(t)
# print(t[:, 1:-1, 1:3])
# print(t[1, 2, 3])
#
# print(t[t > 0])

# t = torch.randn(3, 4)
# print(t)
# print(t.size(0))
# print(torch.argmax(t, 0))
# print(torch.min(t, -1))
# print(t.sort(-1))
#
# a = torch.randn(2, 3, 4)
# b = torch.randn(2, 4, 3)
# print(a)
# print(b)
# print(a.bmm(b))
# # print(torch.einsum("c,a->b", a, b))
#
# t = torch.randn(3, 6)
# print(torch.randn(3, 6))
# print(t.split([1, 2, 3], -1))


# a = dict(type='Shared2FCBBoxHead', in_channels=256)
# c = ('in_channels', 2)
# print(len(c))
# b = []
# for i, j in a.items():
#     b.append((i, j))
# print(b)

from window.main import MainWindow_Model

MainWindow_Model.message_show(22)