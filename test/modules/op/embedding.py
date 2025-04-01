# Copyright (c) 2025 Samsung Electronics Co., Ltd. All Rights Reserved
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch


class SimpleEmbedding(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(num_embeddings=10, embedding_dim=3)

    def forward(self, arg):
        return self.embedding(arg)

    def get_example_inputs(self):
        return (torch.LongTensor([[1, 2, 4, 5], [4, 3, 2, 9]]),)


class PaddedEmbedding(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(
            num_embeddings=10, embedding_dim=3, padding_idx=0
        )

    def forward(self, arg):
        return self.embedding(arg)

    def get_example_inputs(self):
        return (torch.LongTensor([[0, 2, 0, 5]]),)


class ScaleGradEmbedding(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(
            num_embeddings=10, embedding_dim=3, scale_grad_by_freq=True
        )

    def forward(self, arg):
        return self.embedding(arg)

    def get_example_inputs(self):
        return (torch.LongTensor([[0, 2, 0, 5]]),)


class SparseEmbedding(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(
            num_embeddings=10, embedding_dim=3, sparse=True
        )

    def forward(self, arg):
        return self.embedding(arg)

    def get_example_inputs(self):
        return (torch.LongTensor([[1, 2, 4, 5], [4, 3, 2, 9]]),)
