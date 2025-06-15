#include "Worker.h"
#include "CPU/CPUMemoryManager.h"

class CPUWorker : public Worker
{
public:

private:
    void handle_init(Command cmd) override;
    void handle_infer(Command cmd) override;

    CPUMemoryManager memory_manager_;
};