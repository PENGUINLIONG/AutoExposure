#include "gft/log.hpp"
#include "gft/args.hpp"
#include "gft/vk.hpp"
#include "gft/glslang.hpp"

using namespace liong;
using namespace vk;


static const char* APP_NAME = "AutoExposure";
static const char* APP_DESC = "Implementation of various AutoExposure algorithms.";



struct AppConfig {
  bool verbose = false;
} CFG;


scoped::Context CTXT {};


void initialize(int argc, const char** argv) {
  args::init_arg_parse(APP_NAME, APP_DESC);
  args::reg_arg<args::SwitchParser>("-v", "--verbose", CFG.verbose,
    "Produce extra amount of logs for debugging.");
  args::parse_args(argc, argv);

  extern void log_cb(log::LogLevel lv, const std::string& msg);
  log::set_log_callback(log_cb);
  log::LogLevel level = CFG.verbose ?
    log::LogLevel::L_LOG_LEVEL_DEBUG : log::LogLevel::L_LOG_LEVEL_INFO; 
  log::set_log_filter_level(level);

  CTXT = scoped::Context(create_ctxt(ContextConfig { "CTXT", 0 }), false);
}
void finalize() {
  CTXT = {};
}


scoped::Invocation fill(const scoped::Image& img, const vmath::float4& value) {
  static const char* comp_src = R"(
    #version 460
    layout(local_size_x_id=0, local_size_y_id=1, local_size_z_id=2) in;

    layout(binding=0)
    uniform Params {
      vec4 value;
    } u;

    layout(binding=1, rgba16f)
    writeonly image2D out_img;

    void main() {
      ivec3 global_id = gl_GlobalInvocationID;
      ivec2 out_img_size = imageSize(out_img);
      if (global_id.x > out_img_size.x || global_id.y > out_img_size.y) return;

      imageStore(out_img, global_id.xy, u.value);
    }
  )";

  static glslang::ComputeSpirvArtifact art =
    glslang::compile_comp(comp_src, "main");

  static scoped::Task task = CTXT.build_comp_task("fill")
    .comp(art.comp_spv)
    .rsc(L_RESOURCE_TYPE_UNIFORM_BUFFER)
    .rsc(L_RESOURCE_TYPE_STORAGE_IMAGE)
    .build();

  auto params_buf = CTXT.build_buf()
    .streaming_with(value)
    .uniform()
    .build();

  return task.build_comp_invoke()
    .rsc(params_buf.view())
    .rsc(img.view())
    .build();
}

float avg_cpu(
  const std::vector<vmath::float4>& data,
  uint32_t width,
  uint32_t height
) {
  float sum = 0.0f;
  for (uint32_t w = 0; w < width; ++w) {
    for (uint32_t h = 0; h < height; ++h) {
      vmath::float4 c = data[h * width + w];
      sum += 0.2126f * c.x + 0.7152 * c.y + 0.0722 * c.z;
    }
  }
  return sum / (width * height);
}

float avg_gpu(
  const std::vector<vmath::float4>& data,
  uint32_t width,
  uint32_t height
) {
  static const char* comp_src = R"(
    #version 460
    layout(local_size_x_id=0, local_size_y_id=1, local_size_y_id=2) in;

    layout(binding=0)
    uniform Params {
      uvec2 size;
    } u;

    layout(binding=1)
    writeonly buffer Avg { float avg; };
    layout(binding=2)
    uniform sampler2D img;

    float luminance(vec3 c) {
      return 0.2126 * c.r + 0.7152 * c.g + 0.0722 * c.b;
    }

    void main() {
      uvec3 global_id = gl_GlobalInvocationID;
      uint W = u.size.x;
      uint H = u.size.y;

      if (global_id.x > 0 || global_id.y > 0) return;

      float sum = 0.0f;
      for (uint w = 0; w < W; ++w) {
        for (uint h = 0; h < H; ++h) {
          sum += luminance(texelFetch(img, ivec2(w, h), 0).rgb);
        }
      }

      avg = sum / float(W * H);
    }
  )";

  static glslang::ComputeSpirvArtifact art =
    glslang::compile_comp(comp_src, "main");

  static scoped::Task task = CTXT.build_comp_task("avg")
    .comp(art.comp_spv)
    .rsc(L_RESOURCE_TYPE_UNIFORM_BUFFER)
    .rsc(L_RESOURCE_TYPE_STORAGE_BUFFER)
    .rsc(L_RESOURCE_TYPE_SAMPLED_IMAGE)
    .workgrp_size(1, 1, 1)
    .build();

  scoped::GcScope gc();

  vmath::uint2 size(width, height);
  scoped::Buffer params_buf = CTXT.build_buf()
    .streaming_with(size)
    .uniform()
    .build();

  scoped::Image img = CTXT.build_img()
    .width(width)
    .height(height)
    .fmt(fmt::L_FORMAT_R32G32B32A32_SFLOAT)
    .storage()
    .sampled()
    .build();

  scoped::Buffer img_stage_buf = CTXT.build_buf()
    .streaming_with(data)
    .build();

  CTXT.build_trans_invoke()
    .src(img_stage_buf.view())
    .dst(img.view())
    .build()
    .submit()
    .wait();

  float out_avg;
  scoped::Buffer out_buf = CTXT.build_buf()
    .size_like(out_avg)
    .storage()
    .read_back()
    .build();

  auto invoke = task.build_comp_invoke()
    .rsc(params_buf.view())
    .rsc(out_buf.view())
    .rsc(img.view())
    .workgrp_count(1, 1, 1)
    .build();

  invoke.submit().wait();

  out_buf.map_read().read(out_avg);

  return out_avg;
}

void fuzz_avg() {
  auto run = [=](uint32_t WIDTH, uint32_t HEIGHT) {
    std::vector<vmath::float4> data;
    data.resize(WIDTH * HEIGHT);
    for (uint32_t w = 0; w < WIDTH; ++w) {
      for (uint32_t h = 0; h < HEIGHT; ++h) {
        data[h * WIDTH + w] = vmath::float4 {
          float(std::rand() % 1024) / 1024.0f,
          float(std::rand() % 1024) / 1024.0f,
          float(std::rand() % 1024) / 1024.0f,
          float(std::rand() % 1024) / 1024.0f,
        };
      }
    }


    float cpu_res = avg_cpu(data, WIDTH, HEIGHT);
    float gpu_res = avg_gpu(data, WIDTH, HEIGHT);

    if (std::fabs(cpu_res - gpu_res) > 1e-3) {
      log::error("expect=", cpu_res, "; actual=", gpu_res);
      panic();
    }
  };

  run(1, 1);
  run(2, 1);
  run(1, 2);
  run(2, 2);
  run(8, 8);
  run(64, 8);
  run(8, 64);
  run(64, 64);
  run(512, 64);
  run(64, 512);
  run(512, 512);
  run(1024, 768);
  log::info("fuzzed avg");
}

void guarded_main() {
  log::info("hello, world!");
  fuzz_avg();
}



// -----------------------------------------------------------------------------
// Usually you don't need to change things below.

int main(int argc, const char** argv) {
  initialize(argc, argv);
  //try {
    guarded_main();
  //} catch (const std::exception& e) {
  //  liong::log::error("application threw an exception");
  //  liong::log::error(e.what());
  //  liong::log::error("application cannot continue");
  //} catch (...) {
  //  liong::log::error("application threw an illiterate exception");
  //}
  finalize();

  return 0;
}

void log_cb(log::LogLevel lv, const std::string& msg) {
  using log::LogLevel;
  switch (lv) {
  case LogLevel::L_LOG_LEVEL_DEBUG:
    printf("[\x1b[90mDEBUG\x1B[0m] %s\n", msg.c_str());
    break;
  case LogLevel::L_LOG_LEVEL_INFO:
    printf("[\x1B[32mINFO\x1B[0m] %s\n", msg.c_str());
    break;
  case LogLevel::L_LOG_LEVEL_WARNING:
    printf("[\x1B[33mWARN\x1B[0m] %s\n", msg.c_str());
    break;
  case LogLevel::L_LOG_LEVEL_ERROR:
    printf("[\x1B[31mERROR\x1B[0m] %s\n", msg.c_str());
    break;
  }
}