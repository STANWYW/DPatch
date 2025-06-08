import torch
from torch.autograd import Function

# 纯PyTorch实现，避免CUDA链接问题
class ReorgFunction(Function):
    @staticmethod
    def forward(ctx, x, stride):
        ctx.stride = stride
        batch_size, channels, height, width = x.size()
        
        out_height = height // stride
        out_width = width // stride
        out_channels = channels * stride * stride
        
        x = x.view(batch_size, channels, out_height, stride, out_width, stride)
        x = x.permute(0, 1, 3, 5, 2, 4).contiguous()
        x = x.view(batch_size, out_channels, out_height, out_width)

        return x
    
    @staticmethod
    def backward(ctx, grad_output):
        stride = ctx.stride
        batch_size, channels, height, width = grad_output.size()

        out_height = height * stride
        out_width = width * stride
        out_channels = channels // (stride * stride)
        
        grad_output = grad_output.view(batch_size, out_channels, stride, stride, height, width)
        grad_output = grad_output.permute(0, 1, 4, 2, 5, 3).contiguous()
        grad_output = grad_output.view(batch_size, out_channels, out_height, out_width)

        return grad_output, None


class ReorgLayer(torch.nn.Module):
    def __init__(self, stride):
        super(ReorgLayer, self).__init__()
        self.stride = stride

    def forward(self, x):
        return ReorgFunction.apply(x, self.stride)
