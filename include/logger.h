#pragma once

#include <string>
#include <fstream>
#include <mutex>
#include <chrono>
#include <iomanip>
#include <sstream>

enum class LogLevel {
    DEBUG = 0,
    INFO = 1,
    WARN = 2,
    ERROR = 3
};

class Logger {
private:
    std::ofstream log_file;
    std::mutex log_mutex;
    LogLevel current_level;
    bool console_output;
    
    std::string get_timestamp();
    std::string level_to_string(LogLevel level);
    
public:
    Logger(const std::string& filename, LogLevel level = LogLevel::INFO, bool console = true);
    ~Logger();
    
    void log(LogLevel level, const std::string& message);
    void debug(const std::string& message);
    void info(const std::string& message);
    void warn(const std::string& message);
    void error(const std::string& message);
    
    void set_level(LogLevel level);
    void flush();
};

// Глобальный логгер
extern Logger* g_logger;

// Макросы для удобства
#define LOG_DEBUG(msg) if(g_logger) g_logger->debug(msg)
#define LOG_INFO(msg) if(g_logger) g_logger->info(msg)
#define LOG_WARN(msg) if(g_logger) g_logger->warn(msg)
#define LOG_ERROR(msg) if(g_logger) g_logger->error(msg)