import os
import math

import torch
from torch import nn, Tensor
from torch.nn import functional as F
from torch.autograd import Function
from torch.utils.cpp_extension import load
from torch.utils.checkpoint import checkpoint
from torch import jit
from typing import Any, Dict, List, Optional, Tuple

module_path = os.path.dirname(__file__)
dag_kernel = load(
    "dag_loss_fn",
    sources=[
        os.path.join(module_path, "dag_loss.cpp"),
        os.path.join(module_path, "dag_loss.cu"),
        os.path.join(module_path, "dag_best_alignment.cu"),
        os.path.join(module_path, "logsoftmax_gather.cu"),
    ],
    extra_cflags=['-DOF_SOFTMAX_USE_FAST_MATH'],
    extra_cuda_cflags=['-DOF_SOFTMAX_USE_FAST_MATH'],
)

class DagLossFunc(Function):
    config = 1
    config1 = 2
    config2 = 2

    @staticmethod
    def forward(
        ctx,
        match_all, # bsz * tarlen * prelen
        links, # bsz * prelen * translen
        output_length,
        target_length
    ):
        require_gradient = ctx.needs_input_grad[0] or ctx.needs_input_grad[1]
        match_all = match_all.contiguous()
        links = links.contiguous()
        alpha, beta = dag_kernel.dag_loss(match_all, links, output_length, target_length, require_gradient, DagLossFunc.config) # bsz * prelen * tarlen

        if require_gradient:
            res = beta[:, 0, 0].clone()
        else:
            res = alpha[range(alpha.shape[0]), target_length - 1, output_length - 1]
        ctx.save_for_backward(alpha, beta, match_all, links, output_length, target_length)
        return res

    @staticmethod
    def backward(ctx, grad_output):
        alpha, beta, match_all, links, output_length, target_length = ctx.saved_tensors
        if ctx.needs_input_grad[0] or ctx.needs_input_grad[1]:
            grad_match_all, grad_links = dag_kernel.dag_loss_backward(grad_output, alpha, beta, match_all, links, output_length, target_length, DagLossFunc.config1, DagLossFunc.config2)
            return grad_match_all, grad_links, None, None
        else:
            return None, None, None, None

dag_loss = DagLossFunc.apply

class DagBestAlignmentFunc(Function):
    config = 1

    @staticmethod
    def forward(
        ctx,
        match_all, # bsz * tarlen * prelen
        links, # bsz * prelen * translen
        output_length,
        target_length
    ):
        match_all = match_all.contiguous()
        links = links.contiguous()
        alpha, path = dag_kernel.dag_best_alignment(match_all, links, output_length, target_length, DagBestAlignmentFunc.config) # bsz * prelen * tarlen
        path = path.to(torch.long)
        ctx.mark_non_differentiable(path)
        return path

    @staticmethod
    def backward(ctx, grad_output):
        assert False, "no backward function for best alignment"

dag_best_alignment = DagBestAlignmentFunc.apply

class DagLogsoftmaxGatherFunc(Function):

    @staticmethod
    def forward(
        ctx,
        word_ins_out, # bsz * prelen * vocabsize
        select_idx # bsz * prelen * slen
    ):
        require_gradient = ctx.needs_input_grad[0]
        selected_result = dag_kernel.logsoftmax_gather(word_ins_out, select_idx, require_gradient)
        # Note: the cuda kernel will modify word_ins_out and then reuse it in backward
        ctx.mark_dirty(word_ins_out)
        ctx.set_materialize_grads(False)

        if require_gradient:
            ctx.save_for_backward(word_ins_out, select_idx)
            ctx.has_backward = False
        return word_ins_out, selected_result # bsz * prelen * slen

    @staticmethod
    def backward(ctx, grad_word_ins_out, grad_output):
        if not ctx.needs_input_grad[0]:
            return None, None
        assert grad_word_ins_out is None, "Cannot reuse word_ins_out after logsoftmax_gather"
        if grad_output is None:
            return None, None

        assert not ctx.has_backward, "Cannot backward twice in logsoftmax_gather"
        ctx.has_backward = True

        grad_input, selected_idx = ctx.saved_tensors
        grad_input.mul_(grad_output.sum(-1, keepdim=True).neg_().to(grad_input.dtype))
        grad_input.scatter_add_(-1, selected_idx, grad_output.to(grad_input.dtype))

        return grad_input, None

dag_logsoftmax_gather_inplace = DagLogsoftmaxGatherFunc.apply

####################### torch version ####################


@jit.script
def logsumexp_keepdim(x: Tensor, dim: int) -> Tensor:
    m, _ = x.max(dim=dim, keepdim=True)
    mask = m == -float('inf')
    m = m.detach()

    s = (x - m.masked_fill_(mask, 0)).exp_().sum(dim=dim, keepdim=True)
    return s.masked_fill_(mask, 1).log_() + m.masked_fill_(mask, -float('inf'))

@jit.script
def loop_function_noempty(last_f: Tensor, links: Tensor, match: Tensor) -> Tensor:
    f_next = logsumexp_keepdim(last_f + links, 1) # batch * 1 * prelen
    f_next = logsumexp_keepdim(f_next.transpose(1, 2) + match, -1) # batch * prelen * 1
    return f_next

@jit.script
def loop_function_noempty_max(last_f: Tensor, links: Tensor, match: Tensor) -> Tensor:
    f_next = torch.max(last_f + links, dim=1)[0] # batch * 1 * prelen
    f_next = torch.max(f_next.unsqueeze(-1) + match, dim=-1)[0].unsqueeze(-1) # batch * prelen * 1
    return f_next

def torch_dag_loss(match_all, links, output_length, target_length):
    match_all = match_all.transpose(1, 2)
    batch_size, prelen, tarlen = match_all.shape
    assert links.shape[1] == links.shape[2], "links should be batch_size * prelen * prelen"
    
    f_arr = []
    f_init = torch.zeros(batch_size, prelen, 1, dtype=match_all.dtype, device=match_all.device).fill_(float("-inf"))
    f_init[:, 0, 0] = match_all[:, 0, 0]
    f_arr.append(f_init)

    match_all_add_jump_matrix = match_all # batch * prelen * (1 + tarlen)
    match_all_add_jump_matrix = torch.chunk(match_all_add_jump_matrix, tarlen, -1) # k * [batch * prelen * 1]
    # match_empty_add_jump_empty_matrix = match_all_add_jump_matrix[0]
    match_add_jump_matrix = match_all_add_jump_matrix

    for k in range(1, tarlen):
        f_now = loop_function_noempty(f_arr[-1], links, match_add_jump_matrix[k])
        f_arr.append(f_now)

    loss_result = torch.cat(f_arr, -1)[range(batch_size), output_length - 1, target_length - 1]

    return loss_result


def __torch_max_loss(match_all, links, output_length, target_length):
    match_all = match_all.transpose(1, 2)
    batch_size, prelen, tarlen = match_all.shape
    assert links.shape[1] == links.shape[2], "links should be batch_size * prelen * prelen"
    
    f_arr = []
    f_init = torch.zeros(batch_size, prelen, 1, dtype=match_all.dtype, device=match_all.device).fill_(float("-inf"))
    f_init[:, 0, 0] = match_all[:, 0, 0]
    f_arr.append(f_init)

    match_arr = torch.chunk(match_all, tarlen, -1)
    for i in range(1, tarlen):
        f_now = loop_function_noempty_max(f_arr[-1], links, match_arr[i])
        f_arr.append(f_now)

    alllogprob = torch.cat(f_arr, -1)[range(batch_size), output_length - 1, target_length - 1]
    
    return alllogprob

def torch_dag_best_alignment(match_all, links, output_length, target_length):
    with torch.enable_grad():
        match_all.requires_grad_()
        alllogprob = __torch_max_loss(match_all, links, output_length, target_length)
        matchgrad = torch.autograd.grad(alllogprob.sum(), [match_all])[0] # batch * talen * prelen
    pathvalue, path = matchgrad.max(dim=1)
    path.masked_fill_(pathvalue < 0.5, -1)
    return path

def torch_dag_logsoftmax_gather_inplace(word_ins_out, select_idx):
    logits = torch.log_softmax(word_ins_out, -1, dtype=torch.float32)
    match = logits.gather(dim=-1, index=select_idx)
    return word_ins_out, match

####################### For Config Tuning ################

import numpy as np
import random
from collections import defaultdict
from itertools import product
import tqdm

def restore_valid_links(links):
    # batch * prelen * trans_len
    batch_size, prelen, translen = links.shape
    valid_links_idx = torch.arange(prelen, dtype=torch.long, device=links.device).unsqueeze(1) + \
                torch.arange(translen, dtype=torch.long, device=links.device).unsqueeze(0) + 1
    invalid_idx_mask = valid_links_idx >= prelen
    valid_links_idx.masked_fill_(invalid_idx_mask, prelen)
    res = torch.zeros(batch_size, prelen, prelen + 1, dtype=torch.float, device=links.device).fill_(float("-inf"))
    res.scatter_(2, valid_links_idx.unsqueeze(0).expand(batch_size, -1, -1), links)
    return res[:, :, :prelen]

def random_check_loss(bsz, prelen, tarlen, translen, config=1, config1=1, config2=1):
    # print(bsz, prelen, tarlen, translen)
    DagLossFunc.config = config
    DagLossFunc.config1 = config1
    DagLossFunc.config2 = config2

    match_all = torch.rand(bsz, tarlen, prelen).cuda().requires_grad_()
    links = torch.rand(bsz, prelen, translen).cuda().log_softmax(dim=-1).requires_grad_()

    # easy case
    output_length = torch.ones(bsz, dtype=torch.long).cuda() * prelen
    target_length = torch.ones(bsz, dtype=torch.long).cuda() * tarlen

    output_length -= torch.randint(0, min(5, prelen), output_length.shape, device=output_length.device)
    target_length -= torch.randint(0, min(5, tarlen), target_length.shape, device=target_length.device)

    import time
    torch.cuda.synchronize()
    start = time.time()
    res = dag_loss(match_all, links, output_length, target_length)
    torch.cuda.synchronize()
    atime = time.time() - start
    # print("cuda :", atime)
    start = time.time()
    res2 = torch_dag_loss(match_all, restore_valid_links(links), output_length, target_length)
    torch.cuda.synchronize()
    btime = time.time() - start
    # print("torch:", btime)
    assert torch.allclose(res, res2, rtol=1e-03, atol=1e-04)

    # return atime, btime

    start = time.time()
    gA, gB = torch.autograd.grad(res.mean(), [match_all, links], retain_graph=True)
    torch.cuda.synchronize()
    ctime = time.time() - start
    # print("cuda  grad:", ctime)
    start = time.time()
    rA, rB = torch.autograd.grad(res2.mean(), [match_all, links], retain_graph=True)
    dtime = time.time() - start
    # print("torch grad:", dtime)

    assert torch.allclose(gA, rA)
    assert torch.allclose(gB, rB)

    return atime, btime, ctime, dtime

@torch.no_grad()
def torch_check_best_alignemnt(alpha, path, match_all, links, output_length, target_length):
    batch_size, tarlen, prelen = match_all.shape

    res = alpha[range(batch_size), target_length - 1, output_length - 1]
    pos = torch.zeros(batch_size, device="cuda", dtype=torch.long)
    tid = torch.zeros(batch_size, device="cuda", dtype=torch.long)
    nowres = match_all[range(batch_size), tid, pos]

    for i in range(1, prelen):
        tid += (path[:, i] >= 0).int()
        nextpos = (torch.ones_like(pos) * i).masked_fill(path[:, i] < 0, 0) + pos.masked_fill(path[:, i] >= 0, 0)
        nowres += (links[range(batch_size), pos, (-pos + i - 1).clip(min=0)] + match_all[range(batch_size), tid, nextpos]) * (path[:, i] >= 0).float()
        pos = nextpos

    return torch.allclose(res, nowres)

def random_check_align(bsz, prelen, tarlen, translen, config=1):
    # print(bsz, prelen, tarlen, translen)
    DagBestAlignmentFunc.config = config

    match_all = torch.rand(bsz, tarlen, prelen).cuda().requires_grad_()
    links = torch.rand(bsz, prelen, translen).cuda().log_softmax(dim=-1).requires_grad_()

    # easy case
    output_length = torch.ones(bsz, dtype=torch.long).cuda() * prelen
    target_length = torch.ones(bsz, dtype=torch.long).cuda() * tarlen

    output_length -= torch.randint(0, min(5, prelen), output_length.shape, device=output_length.device)
    target_length -= torch.randint(0, min(5, tarlen), target_length.shape, device=target_length.device)

    import time
    torch.cuda.synchronize()
    start = time.time()
    alpha, path = dag_kernel.dag_best_alignment(match_all, links, output_length, target_length, DagBestAlignmentFunc.config)
    res = alpha[range(bsz), target_length - 1, output_length - 1]
    torch.cuda.synchronize()
    atime = time.time() - start
    # print("cuda :", atime)
    start = time.time()
    path2 = torch_dag_best_alignment(match_all, restore_valid_links(links), output_length, target_length)
    torch.cuda.synchronize()
    btime = time.time() - start
    # print("torch:", btime)
    res2 = __torch_max_loss(match_all, restore_valid_links(links), output_length, target_length)

    assert torch.allclose(res, res2, rtol=1e-03, atol=1e-04)
    assert torch_check_best_alignemnt(alpha, path, match_all, links, output_length, target_length)
    assert torch_check_best_alignemnt(alpha, path2, match_all, links, output_length, target_length)

    return atime, btime

def random_check_gather(bsz, prelen, tarlen, vocabsize):
    word_ins_out = torch.rand(bsz, prelen, vocabsize, dtype=torch.float16, device="cuda").requires_grad_()
    select_idx = torch.randint(0, vocabsize - 1, (bsz, prelen, tarlen), device="cuda")

    import time
    torch.cuda.synchronize()
    start = time.time()
    _, res = dag_logsoftmax_gather_inplace(word_ins_out.clone(), select_idx)
    ga = torch.autograd.grad(res.sum() / res.shape[2], [word_ins_out], retain_graph=True)[0]
    torch.cuda.synchronize()
    atime = time.time() - start
    # print("cuda :", atime)
    start = time.time()
    res2 = torch_dag_logsoftmax_gather_inplace(word_ins_out, select_idx)
    ra = torch.autograd.grad(res2.sum() / res.shape[2], [word_ins_out], retain_graph=True)[0]
    torch.cuda.synchronize()
    btime = time.time() - start
    # print("torch:", btime)
    assert torch.allclose(res, res2, rtol=1e-3, atol=1e-4)
    assert torch.allclose(ga, ra, rtol=1e-3, atol=1e-4)

    return atime, btime

def tune_config(skip_forward=False, skip_backward=False, skip_align=False, skip_gather=False):
    config_list = [1,2,3,4]
    config1_list = [1,2]
    config2_list = [1,2,3]
    configalign_list = [1,2,3,4]

    forward_best = DagLossFunc.config
    backward_best = (DagLossFunc.config1, DagLossFunc.config2)
    align_best = DagBestAlignmentFunc.config

    if not skip_forward:
        print("########### Forward Tuning #############")

        a_res, b_res = defaultdict(list), defaultdict(list)
        for i in tqdm.tqdm(range(100)):
            for config in config_list:
                SEED = i
                random.seed(SEED)
                np.random.seed(SEED)
                torch.manual_seed(SEED)
                torch.cuda.manual_seed(SEED)

                tarlen = random.randint(40, 60)
                bsz = 4096 // tarlen
                factor = 8
                # print(f"run {i}")
                
                a, b, c, d = random_check_loss(bsz, tarlen * factor, tarlen, factor * 4, config=config)
                # a, b = random_check(1, 8, 4, 4)
                if i > 0:
                    a_res[config].append(a)
                    b_res[config].append(b)

        forward_res = []
        for config in config_list:
            forward_res.append(np.mean(b_res[config]) / np.mean(a_res[config]))
            print(f"{config}: {np.mean(a_res[config]):.6f} {np.mean(b_res[config]):.6f} {forward_res[-1]:.2f}")
        forward_best = config_list[np.argmax(forward_res)]

        print(f"Best Choice: {forward_best}")


    if not skip_backward:
        print("########### Backward Tuning #############")

        c_res, d_res = defaultdict(list), defaultdict(list)
        for i in tqdm.tqdm(range(50)):
            for config1, config2 in product(config1_list, config2_list):
                SEED = i
                random.seed(SEED)
                np.random.seed(SEED)
                torch.manual_seed(SEED)
                torch.cuda.manual_seed(SEED)


                tarlen = random.randint(40, 60)
                bsz = 4096 // tarlen
                factor = 8
                
                a, b, c, d = random_check_loss(bsz, tarlen * factor, tarlen, factor * 4, config=forward_best, config1=config1, config2=config2)
                # a, b = random_check(1, 8, 4, 4)
                if i > 0:
                    c_res[(config1, config2)].append(c)
                    d_res[(config1, config2)].append(d)
        
        backward_res = []
        for config1, config2 in product(config1_list, config2_list):
            backward_res.append(np.mean(d_res[(config1, config2)]) / np.mean(c_res[(config1, config2)]))
            print(f"{config1, config2}: {np.mean(c_res[(config1, config2)]):.6f} {np.mean(d_res[(config1, config2)]):.6f} {backward_res[-1]:.2f}")
        backward_best =  list(product(config1_list, config2_list))[np.argmax(backward_res)]

        print(f"Best Choice: {backward_best}")

    if not skip_align:
        print("########### Align Tuning #############")

        a_res, b_res = defaultdict(list), defaultdict(list)
        for i in tqdm.tqdm(range(30)):
            for config in configalign_list:
                SEED = i
                random.seed(SEED)
                np.random.seed(SEED)
                torch.manual_seed(SEED)
                torch.cuda.manual_seed(SEED)

                tarlen = random.randint(40, 60)
                bsz = 4096 // tarlen
                factor = 8
                # print(f"run {i}")

                a, b = random_check_align(bsz, tarlen * factor, tarlen, factor * 4, config=config)
                # a, b = random_check(1, 8, 4, 4)
                if i > 0:
                    a_res[config].append(a)
                    b_res[config].append(b)

        align_res = []
        for config in configalign_list:
            align_res.append(np.mean(b_res[config]) / np.mean(a_res[config]))
            print(f"{config}: {np.mean(a_res[config]):.6f} {np.mean(b_res[config]):.6f} {align_res[-1]:.2f}")
        align_best = configalign_list[np.argmax(align_res)]

        print(f"Best Choice: {align_best}")

    if not skip_gather:
        print("########### Test Gather #############")

        a_res, b_res = defaultdict(list), defaultdict(list)
        for i in tqdm.tqdm(range(100)):
            SEED = i
            random.seed(SEED)
            np.random.seed(SEED)
            torch.manual_seed(SEED)
            torch.cuda.manual_seed(SEED)

            tarlen = random.randint(40, 60)
            bsz = 4096 // tarlen
            factor = 8
            vocabsize = random.randint(12345, 23456)
            a, b = random_check_gather(bsz, tarlen * factor, tarlen, vocabsize)
            if i > 0:
                a_res[0].append(a)
                b_res[0].append(b)

        gather_res = np.mean(b_res[0]) / np.mean(a_res[0])
        print(f"{np.mean(a_res[0]):.6f} {np.mean(b_res[0]):.6f} {gather_res:.2f}")

    DagLossFunc.config = forward_best
    DagLossFunc.config1 = backward_best[0]
    DagLossFunc.config2 = backward_best[1]
    DagBestAlignmentFunc.config = align_best

if __name__ == "__main__":
    # tune_config(skip_forward=True, skip_backward=True, skip_align=False, skip_gather=False)
    tune_config()