#include "Worker.h"
#include "CPU/CPUMemoryManager.h"

class CPUWorker : public Worker
{
public:

private:
    Result handle_init() override;
    Result handle_infer() override;

    CPUMemoryManager memory_manager_;
};