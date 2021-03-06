import torch
import torch.nn.functional as F


def top1(output):
    return output.argmax(dim=-1)


def kl_div_loss(output, labels):
    output = F.softmax(output, dim=-1)
    Nb = output.shape[0]
    Ix = torch.arange(Nb)
    if output.is_cuda:
        Ix = Ix.cuda()
    loss = -torch.log(output[Ix, labels])
    return loss


def cw_loss(output, y):
    Ix = torch.arange(output.shape[0], device=output.device)
    p = output.softmax(dim=-1)
    pc = p.clone()
    pc[Ix, y] = 0.0
    return (p[Ix, y] - pc.max(dim=-1)[0]).clamp(min=0.0)


def pgd_attack(model, clf, images, labels, loss_fn, criterion="top1", norm="L2", eps=0.5, alpha=0.1, iters=20):
    """ BATCH-WISE PGD
        model -> the pytorch model
        images -> a batch of images
        labels -> the true labels corresponding to the images
        loss_fn -> use the same loss function that the model was trained on
        criterion -> the classification criterion
        norm -> in which norm are we attacking? (str)
        eps -> the maximum allowed pertubation in the given norm
        alpha -> the PGD step-size
        iters -> the maximum number of perturbations"""

    # check if we are using the GPU
    has_cuda = images.is_cuda

    # initialize the perturbation to zero
    init_images = images.data
    delta = torch.zeros(init_images.shape)
    if has_cuda:
        delta = delta.cuda()

    # get batch size
    sh = init_images.shape
    # print(f"sh = {sh}\n")
    Nb = sh[0]
    # print(f"Nb = {Nb}")
    # print("\n")

    # feel free to comment the following out and set your own alpha
    # some people define alpha this way (e.g., Salman paper)
    alpha = eps / iters * 2

    # perform the iterative attack until maximum iterations are reached OR until all images are misclassified
    for i in range(iters):
        # add the perturbation to the initial images
        delta.requires_grad = True
        input = init_images + delta

        # see which images are still correctly classified
        features, _ = model(input)
        output = clf(features)
        if criterion == "top1":
            pred_labels = top1(output)
        # print(f"labels = {labels}")
        # print("\n")
        # print(f"pred_labels = {pred_labels}")
        # print("\n")
        # TODO: add other classification criteria
        corr = pred_labels == labels
        # print(f"corr = {corr}\n")

        # exit attack if all images are misclassified
        # print(f"corr.sum().item() = {corr.sum().item()}")
        if corr.sum() == 0:
            delta.detach_()
            break

        # perform PGD step on images that are not yet attacked
        model.zero_grad()
        loss = loss_fn(output, labels)
        # print(f"loss = {loss}")
        # exit()
        # TODO: add other loss functions (similarity)

        if norm == "L2":
            gl = torch.autograd.grad(loss.sum(), delta, retain_graph=True)[0]
            # print(f"gl.shape = {gl.shape}\n")
            # print(f"gl = {gl}\n")
            gl = gl[corr]  # only want to perturb correctly classified images
            # print(f"gl[corr] = {gl}")
            # print(f"gl[corr].shape = {gl.shape}")
            gl_norm = gl.view(corr.sum(), -1).norm(p=2, dim=-1)
            zer = torch.zeros(gl_norm.shape)
            if has_cuda:
                zer = zer.cuda()
            gl_nonzero_norm_bool = gl_norm != zer
            gl = gl[gl_nonzero_norm_bool]
            gl_norm = gl_norm[gl_nonzero_norm_bool]
            # print(f"gl after cutting out zero norms: {gl.shape}")
            # print(f"gl_nonzero_norm_bool = {gl_nonzero_norm_bool}")
            # print(f"gl_norm = {gl_norm}")
            # print(f"gl_norm.shape = {gl_norm.shape}")
            gl_scaled = gl.div(
                gl_norm.view(-1, 1, 1, 1))  # need to unsqueeze dimension of gl_norm so that division will work
            # print(f"gl_norm.view(-1, 1, 1, 1) = {gl_norm.view(-1, 1, 1, 1)}")

            # detach from the computation graph to not accumulate gradients and to avoid 'in-place operation' errors
            delta.detach_()
            gl_nonzero_norm_bool_pad = torch.zeros(Nb, dtype=torch.bool)
            if has_cuda:
                gl_nonzero_norm_bool_pad = gl_nonzero_norm_bool_pad.cuda()
            for j in range(gl_nonzero_norm_bool.shape[0]):
                gl_nonzero_norm_bool_pad[j] = gl_nonzero_norm_bool[j]

            delta[gl_nonzero_norm_bool_pad] = delta[gl_nonzero_norm_bool_pad] + alpha * gl_scaled

            # make sure adversarial images have pixel values in (0,1)
            delta.data.add_(init_images)
            delta.data.clamp_(0, 1).sub_(init_images)

            # clamp the L2-norm of the perturbation to the specified max
            delta.data.renorm_(p=2, dim=0, maxnorm=eps)

        # TODO: add the L1 and Linf version

        # detatch from the computation graph to not accumulate gradients (is this necessary?)
        # delta.detach_()

    # if unable to attack after max_iters, put adversarial distance to zero
    features, _ = model(init_images + delta)
    output = clf(features)
    if criterion == "top1":
        pred_labels = top1(output)
    corr = pred_labels == labels
    delta[corr] = torch.zeros(corr.sum(), *sh[1:]).cuda()

    # return the adversarial perturbation
    return delta
