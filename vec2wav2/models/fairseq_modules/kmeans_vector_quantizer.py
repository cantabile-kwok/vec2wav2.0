# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
from vec2wav2.models.fairseq_modules.fp32_group_norm import Fp32GroupNorm


class KmeansVectorQuantizer(nn.Module):
    def __init__(
        self, dim, num_vars, groups, combine_groups, vq_dim, time_first, gamma=0.25
    ):
        """Vector quantization using straight pass-through estimator (i.e. kmeans)

        Args:
            dim: input dimension (channels)
            num_vars: number of quantized vectors per group
            groups: number of groups for vector quantization
            combine_groups: whether to use the vectors for all groups
            vq_dim: dimensionality of the resulting quantized vector
            time_first: if true, expect input in BxTxC format, otherwise in BxCxT
            gamma: commitment loss coefficient
        """
        super().__init__()

        self.groups = groups
        self.combine_groups = combine_groups
        self.input_dim = dim
        self.num_vars = num_vars
        self.vq_dim = vq_dim
        self.time_first = time_first

        assert (
            vq_dim % groups == 0
        ), f"dim {vq_dim} must be divisible by groups {groups} for concatenation"

        self.var_dim = vq_dim // groups
        num_groups = groups if not combine_groups else 1

        self.embedding = nn.Parameter(
            0.01 * torch.randn(num_vars, num_groups, self.var_dim)
        )
        self.projection = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size=1, groups=groups, bias=False),
            Fp32GroupNorm(groups, dim),
        )
        self.gamma = gamma
        self.mse_mean = nn.MSELoss(reduction="mean")

    def _pass_grad(self, x, y):
        """Manually set gradient for backward pass.
        for y = f(x), ensure that during the backward pass,
        dL/dy = dL/dx regardless of f(x).
        Returns:
            y, with the gradient forced to be dL/dy = dL/dx.
        """

        return y.detach() + (x - x.detach())

    @property
    def expand_embedding(self):
        if self.combine_groups:
            return self.embedding.expand(self.num_vars, self.groups, self.var_dim)
        return self.embedding

    def forward_idx(self, x):
        res = self.forward(x, produce_targets=True)
        return res["x"], res["targets"]

    def forward_idx_limited(self, x, valid_label2vqidx_mat):
        # mask_mat = convert_valid_label2vqidx_to_mask_mat(valid_label2vqidx)
        res = self.forward_group2(x, mask_mat=valid_label2vqidx_mat, produce_targets=True)
        return res['x'], res['targets']

    def forward(self, x, produce_targets=False):

        result = {"num_vars": self.num_vars}

        if self.time_first:
            x = x.transpose(1, 2)

        bsz, fsz, tsz = x.shape

        ze = self.projection(x)
        ze_ = ze.view(bsz, self.groups, self.var_dim, tsz).permute(0, 3, 1, 2)
        d = (
            (ze_.unsqueeze(0) - self.expand_embedding.unsqueeze(1).unsqueeze(1))
            .view(self.num_vars, bsz, tsz, self.groups, -1)
            .norm(dim=-1, p=2)
        )
        idx = d.argmin(dim=0)
        zq = (
            torch.stack(
                [
                    self.expand_embedding[idx[..., group], group]
                    for group in range(self.groups)
                ],
                dim=-2,
            )
            .view(bsz, tsz, self.groups * self.var_dim)
            .permute(0, 2, 1)
        )
        assert ze.shape == zq.shape, (ze.shape, zq.shape)
        x = self._pass_grad(ze, zq)

        with torch.no_grad():
            hard_x = (
                idx.new_zeros(bsz * tsz * self.groups, self.num_vars)
                .scatter_(-1, idx.view(-1, 1), 1.0)
                .view(bsz * tsz, self.groups, -1)
            )
            hard_probs = torch.mean(hard_x.float(), dim=0)
            result["code_perplexity"] = torch.exp(
                -torch.sum(hard_probs * torch.log(hard_probs + 1e-7), dim=-1)
            ).sum()

        if produce_targets:
            result["targets"] = idx

        if self.time_first:
            x = x.transpose(1, 2)  # BCT -> BTC
        result["x"] = x

        ze = ze.float()
        zq = zq.float()
        latent_loss = self.mse_mean(zq, ze.detach())
        commitment_loss = self.mse_mean(ze, zq.detach())

        result["kmeans_loss"] = latent_loss + self.gamma * commitment_loss

        return result

    def forward_group2(self, x, mask_mat=None, produce_targets=False, inf=999999):
        assert mask_mat is not None

        result = {"num_vars": self.num_vars}

        if self.time_first:
            x = x.transpose(1, 2)

        bsz, fsz, tsz = x.shape

        ze = self.projection(x)

        ze_ = ze.view(bsz, self.groups, self.var_dim, tsz).permute(0, 3, 1, 2)
        ze_0 = ze_[:, :, 0, None, :]
        ze_1 = ze_[:, :, 1, None, :]  # 4 * 100 * 320 * 128
        cb0_expand = self.expand_embedding[:, 0, :]
        cb1_expand = self.expand_embedding[:, 1, :]  # 320 * 128
        dist_0 = ((ze_0 - cb0_expand) ** 2).sum(dim=-1)[:, :, :, None]
        dist_1 = ((ze_1 - cb1_expand) ** 2).sum(dim=-1)[:, :, None, :]
        res_0, res_1 = torch.broadcast_tensors(dist_0, dist_1)
        mask_mat = (1 - mask_mat[None, None, :, :].to(res_0.device) * torch.ones_like(res_0)) * inf
        # mask_mat = mask_mat.to(x.device)
        d_flt = (res_0 + res_1 + mask_mat).view(bsz, tsz, -1)
        idx_flt = torch.argmin(d_flt, dim=-1)
        idx = torch.stack((idx_flt // self.num_vars, idx_flt % self.num_vars), dim=-1)

        zq = (
            torch.stack(
                [
                    self.expand_embedding[idx[..., group], group]
                    for group in range(self.groups)
                ],
                dim=-2,
            )
            .view(bsz, tsz, self.groups * self.var_dim)
            .permute(0, 2, 1)
        )
        assert ze.shape == zq.shape, (ze.shape, zq.shape)
        x = self._pass_grad(ze, zq)

        with torch.no_grad():
            hard_x = (
                idx.new_zeros(bsz * tsz * self.groups, self.num_vars)
                .scatter_(-1, idx.view(-1, 1), 1.0)
                .view(bsz * tsz, self.groups, -1)
            )
            hard_probs = torch.mean(hard_x.float(), dim=0)
            result["code_perplexity"] = torch.exp(
                -torch.sum(hard_probs * torch.log(hard_probs + 1e-7), dim=-1)
            ).sum()

        if produce_targets:
            result["targets"] = idx

        if self.time_first:
            x = x.transpose(1, 2)  # BCT -> BTC
        result["x"] = x

        ze = ze.float()
        zq = zq.float()
        latent_loss = self.mse_mean(zq, ze.detach())
        commitment_loss = self.mse_mean(ze, zq.detach())

        result["kmeans_loss"] = latent_loss + self.gamma * commitment_loss

        return result

if __name__ == "__main__":
    quantizer = KmeansVectorQuantizer(dim=256, num_vars=320, groups=2, combine_groups=False, vq_dim=256, time_first=True)
    x = torch.ones(4, 100, 256)
    result = quantizer.forward_group2(x, mask_mat=torch.randint(0, 2, (320, 320)))
    print(result)
