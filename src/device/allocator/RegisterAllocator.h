// RegisterAllocator.h
#ifndef REGISTER_ALLOCATOR_H
#define REGISTER_ALLOCATOR_H

#include "AllocatorFactory.h"

// 宏定义，用于注册现有Allocator实例
#define REGISTER_ALLOCATOR(class_name, instance_ptr) \
    namespace { \
        struct Register_##class_name { \
            Register_##class_name() { \
                AllocatorFactory::instance().registerAnimal(#class_name, instance_ptr); \
            } \
        }; \
        static Register_##class_name register_##class_name; \
    }

#endif // REGISTER_ALLOCATOR_H
