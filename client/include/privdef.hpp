//
//  privdef.hpp
//  autd3
//
//  Created by Seki Inoue on 6/7/16.
//
//

#ifndef privdef_h
#define privdef_h

#include <stdint.h>
#include <array>

#define NUM_TRANS_IN_UNIT (249)
#define NUM_TRANS_X (18)
#define NUM_TRANS_Y (14)
#define TRANS_SIZE (10.18f) // in milli meters
#define IS_MISSING_TRANSDUCER(X,Y) (Y==1 && (X==1 || X== 2 || X==16))

#define FPGA_CLOCK (25600000)

#define ULTRASOUND_FREQUENCY_DEFAULT (40000)
#define ULTRASOUND_WAVELENGTH (8.5f)
#define M_PIf (3.14159265f)

#define MOD_SAMPLING_FREQ (4000.0)
#define MOD_FRAME_SIZE (124)


typedef enum {
    LOOP_BEGIN = 1 << 0,
    LOOP_END = 1 << 1,
	MOD_BEGIN = 1 << 2,
	SILENT = 1 << 3,
	MOD_RESET = 1 <<5,
}RxGlobalControlFlags;

typedef struct {
    uint8_t msg_id;
    uint8_t control_flags;
	int8_t frequency_shift;
    uint8_t mod_size;
    uint8_t mod[MOD_FRAME_SIZE];
} RxGlobalHeader;


#endif /* privdef_h */
