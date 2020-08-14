#include <moderngpu/transform.hxx>

using namespace mgpu;

int main(int argc, char **argv) {
  // The context encapsulates things like an allocator and a stream.
  // By default it prints device info to the console.
  standard_context_t context;

  // Launch five threads to greet us.
  transform([] MGPU_DEVICE(
                int index) { printf("Hello GPU from thread %d\n", index); },
            5, context);

  // Synchronize on the context's stream to send the output to the console.
  context.synchronize();

  return 0;
}
