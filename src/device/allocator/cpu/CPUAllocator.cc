#include "Allocator_CPU.h"
#include "../RegisterAllocator.h"

static CPUAllocator allocator_cpu;

REGISTER_ALLOCATOR(CPUAllocator, &allocator_cpu);