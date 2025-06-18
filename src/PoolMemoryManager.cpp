#include "PoolMemoryManager.h"

PoolMemoryManager::PoolMemoryManager(void* pool_ptr, size_t pool_size, size_t alignment) 
        :pool_ptr_(pool_ptr), pool_size_(pool_size), alignment_(alignment){
    if (pool_ptr_ == nullptr || pool_size_ == 0) {
        throw std::invalid_argument("Pool pointer is null or pool size is zero.");
    }

    // 初始时，整个池都是一个大的空闲块
    add_free_block(pool_ptr_, pool_size_);
    total_free_memory_ = pool_size_;
    std::cout << "Memory Pool initialized. Total size: " 
            << pool_size_ / (1024 * 1024) << " MB." << std::endl;
}


PoolMemoryManager::~PoolMemoryManager(){
    // 检查是否有内存泄漏
    if (allocated_blocks_.size() > 0) {
        std::cerr << "PoolMemoryManager Warning: Memory leak detected! " 
                  << allocated_blocks_.size() << " blocks were not deallocated." << std::endl;
        std::cerr << "Total leaked memory: " << pool_size_ - total_free_memory_ << " bytes." << std::endl;
    } else if (total_free_memory_ != pool_size_) {
        // 这通常表示内部状态不一致，是一个严重的 bug
        std::cerr << "PoolMemoryManager Error: Inconsistent memory state on destruction. "
                  << "Expected " << pool_size_ << " free bytes, but found " << total_free_memory_ << "." << std::endl;
    }
}

// 核心分配逻辑
void* PoolMemoryManager::allocate(size_t size) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    if (size == 0) return nullptr;

    size_t aligned_size = get_aligned_size(size);

    // Best-Fit: 找到第一个大小 >= aligned_size 的块
    auto it = free_blocks_by_size_.lower_bound(aligned_size);
    if (it == free_blocks_by_size_.end()) {
        std::cerr << "Allocation failed: Not enough contiguous memory. "
                  << "Requested: " << aligned_size << " bytes. "
                  << "Total free: " << total_free_memory_ << " bytes." << std::endl;
        throw std::bad_alloc();
    }

    void* block_ptr = it->second;
    size_t block_size = it->first;

    // 从空闲块列表中移除这个块
    remove_free_block(block_ptr, block_size);

    size_t remaining_size = block_size - aligned_size;
    // 如果剩余部分太小，就避免产生碎片，将其一并分配 (内部碎片)
    if (remaining_size < alignment_) { 
        aligned_size = block_size;
    } else {
        // 否则，将剩余部分作为新的空闲块加回去
        void* new_free_block_ptr = static_cast<char*>(block_ptr) + aligned_size;
        add_free_block(new_free_block_ptr, remaining_size);
    }

    allocated_blocks_.insert({block_ptr, aligned_size});
    total_free_memory_ -= aligned_size;
    return block_ptr;
}

// 核心回收逻辑
void PoolMemoryManager::deallocate(void* ptr) {
    if (ptr == nullptr) return;

    std::lock_guard<std::mutex> lock(mutex_);
    
    auto it_alloc = allocated_blocks_.find(ptr);
    if (it_alloc == allocated_blocks_.end()) {
        std::cerr << "Error: Trying to deallocate an unknown or already deallocated memory pointer." << std::endl;
        return;
    }

    size_t released_size = it_alloc->second;
    void* released_ptr = it_alloc->first;
    // 从已分配map中移除
    allocated_blocks_.erase(it_alloc);

    total_free_memory_ += released_size;

    // --- 空闲块合并 (Coalescing) ---
    // 1. 尝试与后面的空闲块合并
    void* next_block_ptr = static_cast<char*>(released_ptr) + released_size;
    auto it_next = free_blocks_by_addr_.find(next_block_ptr);
    if (it_next != free_blocks_by_addr_.end()) {
        size_t next_block_size = it_next->second;
        // 从空闲列表移除后面的块
        remove_free_block(it_next->first, it_next->second);
        // 合并
        released_size += next_block_size;
    }

    // 2. 尝试与前面的空闲块合并
    auto it_prev = free_blocks_by_addr_.lower_bound(released_ptr);
    if (it_prev != free_blocks_by_addr_.begin()) {
        --it_prev; // lower_bound 找到的是 >= 的，我们需要 < 的，所以向前一个
        void* prev_block_ptr = it_prev->first;
        size_t prev_block_size = it_prev->second;

        // 检查是否真的相邻
        if (static_cast<char*>(prev_block_ptr) + prev_block_size == released_ptr) {
            // 从空闲列表移除前面的块
            remove_free_block(prev_block_ptr, prev_block_size);
            // 合并
            released_size += prev_block_size;
            released_ptr = prev_block_ptr; // 新的合并块的起始地址是前一个块的地址
        }
    }

    // 3. 将最终合并后的块加回空闲列表
    add_free_block(released_ptr, released_size);
}

size_t PoolMemoryManager::get_free_memory() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return total_free_memory_;
}

size_t PoolMemoryManager::get_total_memory() const {
    return pool_size_;
}

size_t PoolMemoryManager::get_aligned_size(size_t size) const {
    return ((size + alignment_ - 1) / alignment_) * alignment_;
}

void PoolMemoryManager::add_free_block(void* ptr, size_t size){
    free_blocks_by_size_.insert({size, ptr});
    free_blocks_by_addr_.insert({ptr, size});
}

void PoolMemoryManager::remove_free_block(void* ptr, size_t size){
    // 从按大小排序的 multimap 中移除的逻辑稍微复杂，因为可能有多个同样大小的块
    auto range = free_blocks_by_size_.equal_range(size);
    for (auto it = range.first; it != range.second; ++it) {
        if (it->second == ptr) {
            free_blocks_by_size_.erase(it);
            break;
        }
    }
    // 从按地址排序的 map 中移除块
    free_blocks_by_addr_.erase(ptr);
}