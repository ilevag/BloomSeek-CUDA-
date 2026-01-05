#include "thread_pool.h"

ThreadPool::ThreadPool(size_t num_threads) {
    threads_.reserve(num_threads);
    for (size_t i = 0; i < num_threads; ++i) {
        threads_.emplace_back(&ThreadPool::worker_loop, this);
    }
}

ThreadPool::~ThreadPool() {
    stop_.store(true);
    cv_.notify_all();
    for (auto& t : threads_) if (t.joinable()) t.join();
}

void ThreadPool::parallel_for(size_t begin, size_t end, const std::function<void(size_t,size_t)>& fn) {
    const size_t tasks = threads_.empty() ? 1 : threads_.size();
    const size_t total = end - begin;
    const size_t chunk = (total + tasks - 1) / tasks;

    {
        std::lock_guard<std::mutex> lock(mutex_);
        for (size_t i = 0; i < tasks; ++i) {
            size_t s = begin + i * chunk;
            if (s >= end) break;
            size_t e = std::min(end, s + chunk);
            tasks_.push(Task{ s, e, fn });
        }
    }
    cv_.notify_all();

    // Ждём завершения всех задач без активного ожидания
    std::unique_lock<std::mutex> lock(mutex_);
    done_cv_.wait(lock, [&]{ return tasks_.empty() && active_.load() == 0; });
}

void ThreadPool::worker_loop() {
    for(;;) {
        Task task;
        {
            std::unique_lock<std::mutex> lock(mutex_);
            cv_.wait(lock, [&]{ return stop_.load() || !tasks_.empty(); });
            if (stop_.load() && tasks_.empty()) return;
            task = tasks_.front();
            tasks_.pop();
            active_.fetch_add(1, std::memory_order_relaxed);
        }
        for (size_t i = task.begin; i < task.end; ++i) {
            task.fn(i, task.end);
        }
        {
            std::lock_guard<std::mutex> lock(mutex_);
            active_.fetch_sub(1, std::memory_order_relaxed);
            if (tasks_.empty() && active_.load() == 0) {
                done_cv_.notify_all();
            }
        }
    }
}




