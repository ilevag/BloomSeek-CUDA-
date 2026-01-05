#include "logger.h"
#include <iostream>

Logger* g_logger = nullptr;

Logger::Logger(const std::string& filename, LogLevel level, bool console)
    : current_level(level), console_output(console) {
    log_file.open(filename, std::ios::app);
    if (!log_file.is_open()) {
        std::cerr << "Ошибка: Не удалось открыть лог файл: " << filename << std::endl;
    }
}

Logger::~Logger() {
    if (log_file.is_open()) {
        log_file.close();
    }
}

std::string Logger::get_timestamp() {
    auto now = std::chrono::system_clock::now();
    auto time_t = std::chrono::system_clock::to_time_t(now);
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        now.time_since_epoch()) % 1000;
    
    std::stringstream ss;
    ss << std::put_time(std::localtime(&time_t), "%Y-%m-%d %H:%M:%S");
    ss << '.' << std::setfill('0') << std::setw(3) << ms.count();
    return ss.str();
}

std::string Logger::level_to_string(LogLevel level) {
    switch (level) {
        case LogLevel::DEBUG: return "DEBUG";
        case LogLevel::INFO:  return "INFO ";
        case LogLevel::WARN:  return "WARN ";
        case LogLevel::ERROR: return "ERROR";
        default: return "UNKNOWN";
    }
}

void Logger::log(LogLevel level, const std::string& message) {
    if (level < current_level) {
        return;
    }
    
    std::lock_guard<std::mutex> lock(log_mutex);
    
    std::string log_message = "[" + get_timestamp() + "] [" + 
                             level_to_string(level) + "] " + message;
    
    if (console_output) {
        std::cout << log_message << std::endl;
    }
    
    if (log_file.is_open()) {
        log_file << log_message << std::endl;
        log_file.flush();
    }
}

void Logger::debug(const std::string& message) {
    log(LogLevel::DEBUG, message);
}

void Logger::info(const std::string& message) {
    log(LogLevel::INFO, message);
}

void Logger::warn(const std::string& message) {
    log(LogLevel::WARN, message);
}

void Logger::error(const std::string& message) {
    log(LogLevel::ERROR, message);
}

void Logger::set_level(LogLevel level) {
    current_level = level;
}

void Logger::flush() {
    std::lock_guard<std::mutex> lock(log_mutex);
    if (log_file.is_open()) {
        log_file.flush();
    }
}