#pragma once
#include <array>
#include <cstddef>

// Clever little signal system to allow for multiple listeners to be notified of a signal.
// Exetremely useful when there is dynamic memory manipulation, while the GPU is doing something with said memory
// For example, when a MemPool is defragmented, a lot of internal offsets are changed. But the GPU is still using the old offsets. 
// So, we can signal all the tasks using the MemPool to update their VkWriteDescriptorSets.

template<std::size_t MaxListeners>
struct Signal {
    bool isSignaled = false;

    struct ListenerEntry {
        void* ptr = nullptr;
        void (*invoke)(void*);
    };

    std::array<ListenerEntry, MaxListeners> listeners{};
    std::size_t count = 0;

    template<typename T>
    static void invokeFunc(void* p) {
        static_cast<T*>(p)->onSignal();
    }

    template<typename T>
    void addListener(T* listener) {
        if (count < MaxListeners) {
            listeners[count++] = { static_cast<void*>(listener), &invokeFunc<T> };
        }
    }

    template<typename T>
    void removeListener(T* listener) {
        for (std::size_t i = 0; i < count; ++i) {
            if (listeners[i].ptr == listener) {
                listeners[i] = listeners[--count];
                break;
            }
        }
    }

    void trigger() {
        for (std::size_t i = 0; i < count; ++i) {
            listeners[i].invoke(listeners[i].ptr);
        }
        isSignaled = true;
    }
};
