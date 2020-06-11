'''
Odena, Augustus, et al. "Is generator conditioning causally related to gan performance?." arXiv preprint arXiv:1802.08768 (2018).
Kumar, Abhishek, Ben Poole, and Kevin Murphy. Learning Generative Samplers using Relaxed Injective Flow. Technical report, 2019. (https://invertibleworkshop.github.io/accepted_papers/pdfs/INNF_2019_paper_32.pdf)
copy and modified from https://github.com/MCDM2018/Implementations/blob/master/GAN.py
'''
import torch
import torch.nn.functional as F

def minrelu(x):
    return -F.relu(-x)

def jac_clamping_loss(
        forward,
        x, z,
        num_pert_samples,
        eta_min, p=2, EPS=0.01,
        postprocessing=None,
        ):
    '''
    batch_size == num_z_samples
    forward: f(ctx, z, num_pert_samples)
    x: batch_size x x_dim
    z: batch_size x z_dim
    '''
    # init
    batch_size = x.size(0)
    x_dim = x.size(-1)
    z_dim = z.size(-1)
    numel = batch_size*num_pert_samples
    assert x.size(0) == batch_size
    assert z.size(0) == batch_size
    assert p == 2

    # get perturb and z_bar
    perturb = torch.randn([batch_size, num_pert_samples, z_dim], device=z.device)
    z_bar = z.unsqueeze(1) + EPS*perturb # bsz x psz x zdim

    # forward
    x = x.unsqueeze(1).expand(batch_size, num_pert_samples, -1).contiguous().view(numel, x_dim)
    z_bar = z_bar.view(numel, z_dim)
    x_bar = forward(z_bar) # numel x x_dim
    if postprocessing:
        x = postprocessing(x)
        x_bar = postprocessing(x_bar)

    # get diff
    x_diff = x_bar - x

    # flatten
    x_diff_flattened = x_diff.view(numel, x_dim)
    perturb_flattened = perturb.view(numel, z_dim)

    # get jac
    unjac_l2sq = torch.sum((x_diff_flattened**2), dim=1)/(EPS**2) #torch.norm(x_diff_flattened, dim=1, p=p)
    per_l2sq = torch.sum((perturb_flattened**2), dim=1) #torch.norm(eps_diff_flattened, dim=1, p=p)
    jac_l2sq = unjac_l2sq / per_l2sq

    # get loss
    loss = (minrelu(jac_l2sq-eta_min))**2

    # return
    return loss.mean()

def cond_jac_clamping_loss(
        forward,
        x, ctx, z,
        num_z_samples, num_pert_samples,
        eta_min, p=2, EPS=0.01,
        postprocessing=None,
        ):
    '''
    forward: f(ctx, z, num_z_samples, num_pert_samples)
    x: batch_size x num_z_samples x x_dim
    z: batch_size x num_z_samples x z_dim
    ctx: batch_size x ctx_dim
    '''
    # init
    batch_size = ctx.size(0)
    x_dim = x.size(-1)
    z_dim = z.size(-1)
    numel = batch_size*num_z_samples*num_pert_samples
    assert x.size(0) == batch_size*num_z_samples
    assert z.size(0) == batch_size*num_z_samples
    assert p == 2

    # get perturb and z_bar
    perturb = torch.randn([batch_size*num_z_samples, num_pert_samples, z_dim], device=z.device)
    z_bar = z.unsqueeze(1) + EPS*perturb # bsz*zsz x psz x zdim

    # forward
    x = x.unsqueeze(1).expand(batch_size*num_z_samples, num_pert_samples, -1).contiguous().view(numel, x_dim)
    z_bar = z_bar.view(numel, z_dim)
    x_bar = forward(ctx, z_bar, num_z_samples, num_pert_samples) # numel x x_dim
    if postprocessing:
        x = postprocessing(x)
        x_bar = postprocessing(x_bar)

    # get diff
    x_diff = x_bar - x

    # flatten
    x_diff_flattened = x_diff.view(numel, x_dim)
    perturb_flattened = perturb.view(numel, z_dim)

    # get jac
    unjac_l2sq = torch.sum((x_diff_flattened**2), dim=1)/(EPS**2) #torch.norm(x_diff_flattened, dim=1, p=p)
    per_l2sq = torch.sum((perturb_flattened**2), dim=1) #torch.norm(eps_diff_flattened, dim=1, p=p)
    jac_l2sq = unjac_l2sq / per_l2sq

    # get loss
    loss = (minrelu(jac_l2sq-eta_min))**2

    # return
    return loss.mean()
