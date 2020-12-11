cmake_minimum_required(VERSION 3.10)

set(SOEM_PATH ./SOEM)

set (CMAKE_CXX_STANDARD 17)

if(WIN32)
	set(OS win32)
elseif(APPLE)
	set(OS macosx)
else()
	set(OS linux)
endif()

include_directories(
	../include
	${SOEM_PATH}/soem
	${SOEM_PATH}/osal
	${SOEM_PATH}/osal/${OS}
	${SOEM_PATH}/oshw/${OS}
)

if(WIN32)
	include_directories(
		${SOEM_PATH}/oshw/${OS}/wpcap/Include
		${SOEM_PATH}/oshw/${OS}/wpcap/Include/pcap
	)
endif()

add_library(soem STATIC
	${SOEM_PATH}/soem/ethercatbase.c
	${SOEM_PATH}/soem/ethercatcoe.c
	${SOEM_PATH}/soem/ethercatconfig.c
	${SOEM_PATH}/soem/ethercatdc.c
	${SOEM_PATH}/soem/ethercateoe.c
	${SOEM_PATH}/soem/ethercatfoe.c
	${SOEM_PATH}/soem/ethercatmain.c
	${SOEM_PATH}/soem/ethercatprint.c
	${SOEM_PATH}/soem/ethercatsoe.c
	
	${SOEM_PATH}/osal/${OS}/osal.c
	${SOEM_PATH}/oshw/${OS}/nicdrv.c
	${SOEM_PATH}/oshw/${OS}/oshw.c
)
