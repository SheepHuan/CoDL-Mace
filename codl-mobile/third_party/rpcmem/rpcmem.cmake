set(RPCMEM_INSTALL_DIR  "${PROJECT_SOURCE_DIR}/third_party/rpcmem")
set(RPCMEM_INCLUDE_DIR  "${RPCMEM_INSTALL_DIR}")

include_directories(SYSTEM "${RPCMEM_INCLUDE_DIR}")

set(RPCMEM_LIB "${RPCMEM_INSTALL_DIR}/${ANDROID_ABI}/rpcmem.a")
add_library(rpcmem STATIC IMPORTED GLOBAL)
set_target_properties(rpcmem PROPERTIES IMPORTED_LOCATION ${RPCMEM_LIB})

install(FILES ${RPCMEM_LIB} DESTINATION lib)