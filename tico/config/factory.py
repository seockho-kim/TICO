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

from typing import Type

from tico.config.base import CompileConfigBase
from tico.config.v1 import CompileConfigV1


class CompileConfigFactory:
    _config_classes = {
        "1.0": CompileConfigV1,
        # '2.0': CompileConfigV2,
    }

    @classmethod
    def get_config(cls, version: str) -> Type[CompileConfigBase]:
        if version not in cls._config_classes:
            raise ValueError(f"Unsupported version: {version}")

        return cls._config_classes[version]

    @classmethod
    def create(cls, version: str):
        config_class = cls.get_config(version)
        return config_class()


def get_default_config(version: str = "1.0"):
    return CompileConfigFactory.create(version)
