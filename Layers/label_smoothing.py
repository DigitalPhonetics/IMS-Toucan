# Copyright    2021  Xiaomi Corp.        (authors: Fangjun Kuang)
#
# See ../../../../LICENSE for clarification regarding multiple authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch


class LabelSmoothingLoss(torch.nn.Module):
    """
    Implement the LabelSmoothingLoss proposed in the following paper
    https://arxiv.org/pdf/1512.00567.pdf
    (Rethinking the Inception Architecture for Computer Vision)

    """

    def __init__(
        self,
        ignore_index: int = -1,
        label_smoothing: float = 0.1,
        reduction: str = "sum",
    ) -> None:
        """
        Args:
          ignore_index:
            ignored class id
          label_smoothing:
            smoothing rate (0.0 means the conventional cross entropy loss)
          reduction:
            It has the same meaning as the reduction in
            `torch.nn.CrossEntropyLoss`. It can be one of the following three
            values: (1) "none": No reduction will be applied. (2) "mean": the
            mean of the output is taken. (3) "sum": the output will be summed.
        """
        super().__init__()
        assert 0.0 <= label_smoothing < 1.0
        self.ignore_index = ignore_index
        self.label_smoothing = label_smoothing
        self.reduction = reduction

    def forward(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute loss between x and target.

        Args:
          x:
            prediction of dimension
            (batch_size, input_length, number_of_classes).
          target:
            target masked with self.ignore_index of
            dimension (batch_size, input_length).

        Returns:
          A scalar tensor containing the loss without normalization.
        """
        assert x.ndim == 3
        assert target.ndim == 2
        assert x.shape[:2] == target.shape
        num_classes = x.size(-1)
        x = x.reshape(-1, num_classes)
        # Now x is of shape (N*T, C)

        # We don't want to change target in-place below,
        # so we make a copy of it here
        target = target.clone().reshape(-1)

        ignored = target == self.ignore_index

        # See https://github.com/k2-fsa/icefall/issues/240
        # and https://github.com/k2-fsa/icefall/issues/297
        # for why we don't use target[ignored] = 0 here
        target = torch.where(ignored, torch.zeros_like(target), target)

        true_dist = torch.nn.functional.one_hot(target, num_classes=num_classes).to(x)

        true_dist = (
            true_dist * (1 - self.label_smoothing) + self.label_smoothing / num_classes
        )

        # Set the value of ignored indexes to 0
        #
        # See https://github.com/k2-fsa/icefall/issues/240
        # and https://github.com/k2-fsa/icefall/issues/297
        # for why we don't use true_dist[ignored] = 0 here
        true_dist = torch.where(
            ignored.unsqueeze(1).repeat(1, true_dist.shape[1]),
            torch.zeros_like(true_dist),
            true_dist,
        )

        loss = -1 * (torch.log_softmax(x, dim=1) * true_dist)
        if self.reduction == "sum":
            return loss.sum()
        elif self.reduction == "mean":
            return loss.sum() / (~ignored).sum()
        else:
            return loss.sum(dim=-1)