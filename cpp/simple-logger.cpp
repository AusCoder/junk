#include <iostream>
#include <memory>

struct Logger {
  // virtual triggers runtime dispatch based on the pointer type we call it with
  virtual void LogMessage(const char *message) = 0;
  virtual ~Logger() = default;
};

struct ConsoleLogger final : Logger {
  void LogMessage(const char *message) override {
    std::cout << message << "\n";
  }
};

struct SurpriseLogger final : Logger {
  void LogMessage(const char *message) override { std::exit(1); }
};

void LogHelloWorld(Logger &logger) { logger.LogMessage("hello world"); }

int main() {
  auto logger{std::make_unique<SurpriseLogger>()};
  LogHelloWorld(*logger.get());
}