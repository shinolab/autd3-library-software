cmake_minimum_required(VERSION 3.12)

set(SOEM_PATH ../deps/SOEM)

include_directories(
	../include

	${SOEM_PATH}/soem
	${SOEM_PATH}/osal
	${SOEM_PATH}/osal/win32
	${SOEM_PATH}/oshw/win32
	${SOEM_PATH}/oshw/win32/wpcap/Include
	${SOEM_PATH}/oshw/win32/wpcap/Include/pcap
)

add_library(libsoem STATIC
	libsoem.cpp
	../include/libsoem.hpp

	${SOEM_PATH}/soem/ethercatbase.c
	${SOEM_PATH}/soem/ethercatcoe.c
	${SOEM_PATH}/soem/ethercatconfig.c
	${SOEM_PATH}/soem/ethercatdc.c
	${SOEM_PATH}/soem/ethercateoe.c
	${SOEM_PATH}/soem/ethercatfoe.c
	${SOEM_PATH}/soem/ethercatmain.c
	${SOEM_PATH}/soem/ethercatprint.c
	${SOEM_PATH}/soem/ethercatsoe.c
	
	${SOEM_PATH}/osal/win32/osal.c
	${SOEM_PATH}/oshw/win32/nicdrv.c
	${SOEM_PATH}/oshw/win32/oshw.c
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
	
	${SOEM_PATH}/osal/win32/osal.c
	${SOEM_PATH}/oshw/win32/nicdrv.c
	${SOEM_PATH}/oshw/win32/oshw.c
)

add_custom_command(TARGET libsoem
                   POST_BUILD
                   COMMAND lib /OUT:$(OUTDIR)temp.lib $(TARGETPATH) $(SolutionDir)..\\deps\\SOEM\\oshw\\win32\\wpcap\\Lib\\x64\\*.lib
                   COMMAND del $(TARGETPATH)
                   COMMAND rename $(OUTDIR)temp.lib $(TargetFileName)
)

