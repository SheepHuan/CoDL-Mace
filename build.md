```bash

cd /root/codl/codl-mobile
bash tools/codl/build_executable_files.sh
bash tools/codl/push_executable_files.sh H933CK9N01234567 




cd  /root/codl-eval-tools/codl-lat-collect-and-eval/
bash tools/collect_all_op_latency.sh kirin990  H933CK9N01234567 


cd /root/codl-eval-tools/codl-lat-predict
 bash tools/train_and_eval_lat_predictors.sh /root/codl-eval-tools/codl-lat-collect-and-eval/lat_datasets kirin990



cd /root/codl/codl-eval-tools/codl-lat-collect-and-eval
bash tools/push_lat_predictors_and_configs.sh \
  dimensity8050 \
  /root/codl/codl-eval-tools/codl-lat-predict/saved_models \
  /root/codl/codl-eval-tools/codl-lat-collect-and-eval/configs \
  H933CK9N01234567

cd /root/codl/codl-eval-tools/codl-lat-collect-and-eval
bash tools/eval_codl_and_baselines.sh H933CK9N01234567

# 需要手动调整chanidx

adb -s H933CK9N01234567 shell " MACE_OPENCL_PROFILING=1 CODL_CONFIG_PATH=/data/local/tmp/codl/configs/config_codl.json /data/local/tmp/codl/codl_run --test=vgg16_chain_search --op_idx=0 --op_count=-1 --chain_idx=0 --chain_count=-1 --num_threads=4 --chain_param_hint=11 --gpu_mtype=1 --data_transform --compute --latency_acq=1 --lp_backend=0 --search_method=serial --search_baseline=0 --pratio_hint=0 --rounds=50 --debug_level=1"
adb -s H933CK9N01234567 shell " MACE_OPENCL_PROFILING=1 CODL_CONFIG_PATH=/data/local/tmp/codl/configs/config_codl.json /data/local/tmp/codl/codl_run --test=vgg16_chain_search --op_idx=0 --op_count=-1 --chain_idx=1 --chain_count=-1 --num_threads=4 --chain_param_hint=11 --gpu_mtype=1 --data_transform --compute --latency_acq=1 --lp_backend=0 --search_method=serial --search_baseline=0 --pratio_hint=0 --rounds=50 --debug_level=1"
```