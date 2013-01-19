The parallel version of the drop detector part of the linescan application by CUDA.

drop_detector_cpu.cpp           the reference serial implementation
drop_detector_v1.cu             each nozzle associated with a CUDA block
drop_detector_v2.cu             each nozzle associated with a CUDA block and CUDA threads parallelize operations within each blocks