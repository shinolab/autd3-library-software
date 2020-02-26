cmake_minimum_required(VERSION 3.10)

set(SOEM_PATH ../deps/SOEM)

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

add_library(autdsoem STATIC
	libsoem.cpp
	../lib/libsoem.hpp

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

source_group("soem" FILES
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

if(WIN32)
	add_custom_command(TARGET autdsoem
					POST_BUILD
					COMMAND lib /OUT:$(OUTDIR)temp.lib $(TARGETPATH) $(SolutionDir)..\\deps\\SOEM\\oshw\\win32\\wpcap\\Lib\\x64\\*.lib /ignore:4006
					COMMAND del $(TARGETPATH)
					COMMAND rename $(OUTDIR)temp.lib $(TargetFileName)
	)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /wd26451 /wd6385  /wd6386 /wd6011 /wd26495 /wd4996 /wd6001 /wd4200")
endif()

if(NOT WIN32)
  set_target_properties(autdsoem PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif()
