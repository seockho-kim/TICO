#!/bin/bash

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

# This script is called by `ccex`
#
# [EXPORTED VARIABLES]
# - CCEX_PROJECT_PATH

show_help() {
# `cat << EOF` This means that cat should stop reading when EOF is detected
cat << EOF  
Usage: ./ccex configure test [--torch_ver|--help|-h]
--torch_ver     [2.5|2.6|nightly]
                Specify torch version to install family test packages.
                (default: 2.6)
-h | --help     Show help message and exit
EOF
}

SCRIPTS_DIR="${CCEX_PROJECT_PATH}/infra/scripts"
TEST_DIR="${CCEX_PROJECT_PATH}/test"

pushd ${CCEX_PROJECT_PATH} > /dev/null

options=$(getopt -o h --long torch_ver:,help -- "$@")

[ $? -eq 0 ] || { 
    echo "Incorrect options provided"
    exit 1
}

_TORCH_VER="2.6"

eval set -- "$options"

while true; do
    case "$1" in
        --torch_ver)
            _TORCH_VER="$2"
            shift
            ;;
        -h|--help)
            show_help
            exit 0;
            ;;
        --)
            shift
            break;;
        *)  break;;
    esac
    shift
done

if [ "$_TORCH_VER" != "2.5" -a "$_TORCH_VER" != "2.6" -a "$_TORCH_VER" != "nightly" ]; then
    echo "Invalid '$_TORCH_VER'"
    echo "(Use --help to see available options)"
    exit 1
fi

if [ "$_TORCH_VER" == "nightly" ]; then
  echo "Install test package dependencies from nightly version"
  python3 -m pip install -r "${TEST_DIR}/requirements_dev.txt"
elif [ "$_TORCH_VER" == "2.6" ]; then
  echo "Install test package dependencies from stable version"
  python3 -m pip install -r "${TEST_DIR}/requirements_2_6.txt"
elif [ "$_TORCH_VER" == "2.5" ]; then
  echo "Install test package dependencies from stable version"
  python3 -m pip install -r "${TEST_DIR}/requirements_2_5.txt"
fi

popd > /dev/null
