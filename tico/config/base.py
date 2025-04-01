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

from dataclasses import dataclass


@dataclass
class CompileConfigBase:
    def get(self, name: str):
        return getattr(self, name) if hasattr(self, name) else None

    def set(self, name: str, enabled: bool):
        setattr(self, name, enabled)

    def to_dict(self):
        return {key: value for key, value in self.__dict__.items()}

    @classmethod
    def from_dict(cls, config_dict: dict):
        config = cls()
        for key in config_dict:
            if key in config.to_dict():
                assert type(config.get(key)) == bool
                config.set(key, config_dict[key])

        return config
