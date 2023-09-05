import os
import re
def get_chain_len_from_pratioes(pratioes):
    chain_len = [1]
    prev = pratioes[0]
    g1=[prev]
    for pratio in pratioes[1:]:
        if pratio == prev:
            chain_len[-1]+=1
        else:
            chain_len.append(1)
            prev = pratio
            g1.append(pratio)
    return chain_len,g1
    

    
def cpu_only(device_id,model_name,round=50):
    cmd = f"""
    adb -s {device_id} shell \
        "MACE_OPENCL_PROFILING=1 CODL_CONFIG_PATH=/data/local/tmp/codl/configs/config_codl.json /data/local/tmp/codl/codl_run \
            --test={model_name}_chain_search --op_idx=0 --op_count=-1 --chain_idx=-1 --chain_count=-1 --num_threads=4 --chain_param_hint=0 \
            --gpu_mtype=2 --compute --latency_acq=1 --lp_backend=1 --search_method=serial \
            --search_baseline=0 --pratio_hint=0 --rounds={round} --debug_level=0" | grep "total"
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
    
def tune_run(model_name,chain_idx,base_pdim,base_pratioes,chain_len,device_id="H933CK9N01234567",round=50):
    cmd = f"""
    adb -s {device_id} shell \
        " MACE_OPENCL_PROFILING=1 CODL_CONFIG_PATH=/data/local/tmp/codl/configs/config_codl.json /data/local/tmp/codl/codl_run \
            --test={model_name}_chain_search --op_idx=0 --op_count=-1 --chain_idx={chain_idx} --chain_count=-1 --num_threads=4 \
            --chain_param_hint=11 --gpu_mtype=1 --data_transform --compute --latency_acq=1 --lp_backend=0 \
            --search_method=serial --search_baseline=0 \
            --pratio_hint=0 --rounds={round} --debug_level=1 \
            --pdim_hint_list={",".join([str(i) for i in base_pdim])} \
            --pratio_hint_list={",".join([str(i) for i in base_pratioes])} \
            --chain_lengths_list={",".join([str(i) for i in chain_len])}" | grep "total"
    
    """
    # print(cmd)
    ret_text = os.popen(cmd).read()
    # print(ret_text)
    pattern = r"total (\d+\.\d+)"
    result = re.search(pattern, ret_text)
    if result:
        time = result.group(1)
        # print(float(time))
        return float(time)
    
def tune_vgg16(device_id="H933CK9N01234567",round=50):
    """
    pratio: 这里的切分比如果是0.4，则算子在GPU算0.6，CPU上算0.4
    """
    # base_pratioes = [0.6, 0.6, 0.6, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.4, 0.4, 0.3, 0.3, 0.5, 0.5, 0.5, 1, 0.5, 0.2, 0.1]
    # base_pdim = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 4, 4, 4, 1, 4, 4, 4]
    
    chain_idx = 0
    # base_pratioes = [0.9, 0.9, 0.9, 1, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.4, 0.4, 0.3, 0.3, 0.5, 0.5, 0.5, 1, 0.5, 0.2, 0.1]
    base_pratioes = [0.8 for i in range(21)]
    # base_pdim = [1 for i in base_pratioes]
    base_pdim = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 4, 4, 4, 1, 4, 4, 4]
    
    chain_len = get_chain_len_from_pratioes(base_pratioes)
    tune_run("vgg16",chain_idx,base_pdim,base_pratioes,chain_len)
    
def tune_retinaface():
    # chain_len = [1, 2, 6, 3, 7, 2, 1, 1, 6, 6, 4, 3, 3, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    #                    1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    #                    1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    #                    1, 1, 1, 1, 1, 1, 1]
    base_pdim = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
               1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
               1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
               1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
               1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
               1, 4, 1, 1, 1, 1, 1, 1, 1, 1,
               1, 1, 1, 1, 1, 1, 4, 1, 1, 1,
               1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
               1, 4, 1, 1, 1, 1, 1]
    base_pratioes = [0.9, 0.8, 0.3, 0.8,
                    0.3, 0.3,
                    0.8, 
                    0.3, 0.1, 
                    1, 1, 1,
                    0.8, 
                    0.3, 0.2,0.8,0.5, 0.1, 0.8,0.4, 0.1, 1, 1, 
                    0.2, 0.9,0.3, 0.2, 0.9, 0.3,
                    0.2,1, 1, 0.1,
                    1, 1, 0, 
                    0.9, 0.2, 0.1, 
                    0.9, 0.3, 0.2, 
                    0.9, 0.2, 
                    0.4, 1, 0.3, 0.3, 0.7, 0.7,0.1, 
                    0.7, 0.2, 0.9, 
                    0.4, 0.9, 0.4, 0.5, 1, 1,
                    1, 1, 1, 1, 1, 1, 1, 0.7, 0.7, 1,
                    1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                        1, 1, 0.9, 1, 1, 1, 1]
    # base_pratioes = [0.7 for i in base_pdim]
    chain_len,g1 = get_chain_len_from_pratioes(base_pratioes)
    # chain_idx = 35
    # print(g1[chain_idx])
    # time1 = tune_run("retinaface",chain_idx,base_pdim,base_pratioes,chain_len)
    # print(chain_idx,time1)
    
    time = 0
    for chain_idx in range(len(chain_len)):
        time1 = tune_run("retinaface",chain_idx,base_pdim,base_pratioes,chain_len)
        print(chain_idx,time1)
        time+=time1
    print(time)

lat_dict = {
    "vgg16": {
        "cpu_only": 173.665,
        "gpu_only": 213.436,
        "codl_buffer": {
            "pratioes":[0.8 for i in range(21)],
            "pdim":[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 4, 4, 4, 1, 4, 4, 4],
            "chain_len":[0],
            "lat": 67.5758
        }
    },
    "retinaface":{
        "cpu_only": 1323.06,
        "gpu_only": 924.678,
        "codl_buffer": {
            "pratioes":[0.8 for i in range(21)],
            "pdim":[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 4, 4, 4, 1, 4, 4, 4],
            "chain_len":[0],
            "lat": 67.5758
        } 
    }
    
}
    
device_id = "H933CK9N01234567"  
model_name = "retinaface"



# cpu_only(device_id,model_name,round=100) 
# gpu_only(device_id,model_name,round=100)
# tune_vgg16()
tune_retinaface()