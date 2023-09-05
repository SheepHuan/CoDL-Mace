import os
import re
def get_chain_len_from_pratioes(pratioes):
    chain_len = [1]
    prev = pratioes[0]
    for pratio in pratioes[1:]:
        if pratio == prev:
            chain_len[-1]+=1
        else:
            chain_len.append(1)
            prev = pratio
    return chain_len
    

    
def cpu_only(device_id,model_name,round=50):
    cmd = f"""
    adb -s {device_id} shell \
        "MACE_OPENCL_PROFILING=1 CODL_CONFIG_PATH=/data/local/tmp/codl/configs/config_codl.json /data/local/tmp/codl/codl_run \
            --test={model_name}_chain_search --op_idx=0 --op_count=-1 --chain_idx=-1 --chain_count=-1 --num_threads=4 --chain_param_hint=0 \
            --gpu_mtype=2 --compute --latency_acq=1 --lp_backend=1 --search_method=serial \
            --search_baseline=0 --pratio_hint=0 --rounds={round} --debug_level=0" 
    """
    ret_text = os.popen(cmd).read()
    pattern = r"total (\d+\.\d+)"
    result = re.search(pattern, ret_text)
    if result:
        time = result.group(1)
        print(float(time))

def gpu_only(device_id,model_name,round=50):
    cmd = f"""
    adb -s {device_id} shell " MACE_OPENCL_PROFILING=1 CODL_CONFIG_PATH=/data/local/tmp/codl/configs/config_codl.json /data/local/tmp/codl/codl_run \
        --test={model_name}_chain_search --op_idx=0 --op_count=-1 --chain_idx=-1 --chain_count=-1 --num_threads=4 --chain_param_hint=1 --gpu_mtype=2 \
        --data_transform --compute --latency_acq=1 --lp_backend=1 --search_method=serial --search_baseline=0 --pratio_hint=0 --rounds={round} --debug_level=0" \
        | grep "total" 
    """
    ret_text = os.popen(cmd).read()
    # print(ret_text)
    pattern = r"total (\d+\.\d+)"
    result = re.search(pattern, ret_text)
    if result:
        time = result.group(1)
        print(float(time))
    
    
def tune_vgg16(chain_idx,device_id="H933CK9N01234567",round=50):
    """
    pratio: 这里的切分比如果是0.4，则算子在GPU算0.6，CPU上算0.4
    """
    base_pratioes = [0.6, 0.6, 0.6, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.4, 0.4, 0.3, 0.3, 0.5, 0.5, 0.5, 1, 0.5, 0.2, 0.1]
    base_pdim = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 4, 4, 4, 1, 4, 4, 4]
    
    chain_len = get_chain_len_from_pratioes(base_pratioes)
    print(chain_len)
    cmd = f"""
    adb -s {device_id} shell \
        " MACE_OPENCL_PROFILING=1 CODL_CONFIG_PATH=/data/local/tmp/codl/configs/config_codl.json /data/local/tmp/codl/codl_run \
            --test=vgg16_chain_search --op_idx=0 --op_count=-1 --chain_idx={chain_idx} --chain_count=-1 --num_threads=4 \
            --chain_param_hint=11 --gpu_mtype=1 --data_transform --compute --latency_acq=1 --lp_backend=0 \
            --search_method=serial --search_baseline=0 \
            --pratio_hint=0 --rounds={round} --debug_level=1 \
            --pdim_hint_list={",".join([str(i) for i in base_pdim])} \
            --pratio_hint_list={",".join([str(i) for i in base_pratioes])} \
            --chain_lengths_list={",".join([str(i) for i in chain_len])}"
    
    """
    print(cmd)
    
device_id = "H933CK9N01234567"  
model_name = "vgg16"
# gpu_only(device_id,model_name)
tune_vgg16(0)