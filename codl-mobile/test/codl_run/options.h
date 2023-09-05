#ifndef OPTIONS_H
#define OPTIONS_H

#include "gflags/gflags.h"

DECLARE_string(test);
DECLARE_int32(chain_idx);
DECLARE_int32(chain_count);
DECLARE_int32(chain_param_hint);
DECLARE_int32(op_idx);
DECLARE_int32(op_count);
DECLARE_int32(cpu_dtype);
DECLARE_int32(num_threads);
DECLARE_int32(gpu_dtype);
DECLARE_int32(gpu_mtype);
DECLARE_double(part_ratio);
DECLARE_double(size_ratio);
DECLARE_int32(rounds);
DECLARE_int32(latency_acq);
DECLARE_int32(lp_backend);
DECLARE_bool(make_partition_plan);
DECLARE_bool(profile_data_transform);
DECLARE_bool(profile_compute);
DECLARE_int32(pdim_hint);
DECLARE_int32(pratio_hint);

DECLARE_bool(data_transform);
DECLARE_bool(compute);
DECLARE_string(search_method);
DECLARE_int32(search_baseline);
DECLARE_int32(debug_level);


DECLARE_string(pdim_hint_list);
DECLARE_string(pratio_hint_list);
DECLARE_string(chain_lengths_list);

std::vector<int> split2int(std::string text);
std::vector<float> split2float(std::string text);
#endif