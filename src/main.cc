#include "Engine.h"

int main() {
    /**
     * Server 做完tokenize之后，使用 Engine.add_request() 将数据添加到Engine请求队列中
     * Engine 通过 Scheduler 组织 batch 送进 Worker 进行处理
     * Server 使用 Engine.step() 进行迭代 返回 List<SequenceEvent> 请求状态
     * Server 处理之后发送给用户
     */
    
    // input：
    //      world_size, cur_world rank_size
    //      param_filepath
    //      config
    Engine engine(5, 0);
    // inputWarp -> outputWarp
    engine.step();
    return 0;
}
