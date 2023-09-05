#!/bin/bash

ADB_DEVICE=$1

python test/op_chain_adb_run_test2.py \
  --dl_model=vgg16 \
  --exec_type=CODL_BUFFER_BASED \
  --rounds=1 \
  --adb_device=$ADB_DEVICE
