import torch
import torch.autograd
from torch.autograd import Function
import torch.nn.functional as F

'''
https://pytorch.org/docs/stable/notes/extending.html
'''
class AuxLossForGradFunction(Function):

    # Note that both forward and backward are @staticmethods
    @staticmethod
    def forward(ctx, input, grad):
        ctx.save_for_backward(input, grad.detach())
        return torch.sum(torch.zeros_like(input))

    # This function has only a single output, so it gets only one gradient
    @staticmethod
    def backward(ctx, grad_output):
        # This is a pattern that is very convenient - at the top of backward
        # unpack saved_tensors and initialize all gradients w.r.t. inputs to
        # None. Thanks to the fact that additional trailing Nones are
        # ignored, the return statement is simple even when the function has
        # optional inputs.
        input, grad = ctx.saved_tensors
        grad_input = grad_grad = None

        # These needs_input_grad checks are optional and there only to
        # improve efficiency. If you want to make your code simpler, you can
        # skip them. Returning gradients for inputs that don't require it is
        # not an error.
        if ctx.needs_input_grad[0]:
            grad_input = grad

        return grad_input, grad_grad

aux_loss_for_grad = AuxLossForGradFunction.apply


''' test '''
#import ipdb
if __name__ == '__main__':
    batch_size = 10
    input_dim = 20
    input  = torch.randn(batch_size, input_dim, dtype=torch.double, requires_grad=True)
    grad   = torch.randn(batch_size, input_dim, dtype=torch.double, requires_grad=False)
    target = torch.randn(batch_size, input_dim, dtype=torch.double)

    #loss = F.mse_loss(input, target)
    #loss.backward()
    #ipdb.set_trace()

    loss = aux_loss_for_grad(input, grad)
    loss.backward()
    #ipdb.set_trace()
    print(input.grad)
    print(grad)
    print(torch.allclose(input.grad, grad))
