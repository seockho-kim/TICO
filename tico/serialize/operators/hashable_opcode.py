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

from circle_schema import circle


class OpCode(circle.OperatorCode.OperatorCodeT):
    """
    Wrapper class for operator code in circle schema
    This implements __eq__ and __hash__ for use with dict()
    """

    def __init__(self):
        super().__init__()

    def __eq__(self, other):
        if self.version != other.version:
            return False

        if self.builtinCode == circle.BuiltinOperator.BuiltinOperator.CUSTOM:
            return self.customCode == other.customCode

        return self.builtinCode == other.builtinCode

    def __hash__(self):
        val = (
            self.deprecatedBuiltinCode,
            self.customCode,
            self.version,
            self.builtinCode,
        )
        return hash(val)
