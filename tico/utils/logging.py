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

import logging
import os


def _loggerLevel():
    TICO_LOG = os.environ.get("TICO_LOG")
    if TICO_LOG == "1":
        log_level = logging.FATAL
    elif TICO_LOG == "2":
        log_level = logging.WARNING
    elif TICO_LOG == "3":
        log_level = logging.INFO
    elif TICO_LOG == "4":
        log_level = logging.DEBUG
    else:
        log_level = logging.WARNING
    return log_level


LOG_LEVEL = _loggerLevel()


def getLogger(name: str):
    """
    Get logger with setting log level according to the `TICO_LOG` environment variable.
    """
    logging.basicConfig()
    logger = logging.getLogger(name)
    logger.setLevel(LOG_LEVEL)

    return logger
