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
    layout(local_size_x=8, local_size_y=8, local_size_z=1) in;

    layout(binding=0)
    uniform Params {
      uvec2 size;
    } u;

    layout(binding=1, r32f)
    writeonly uniform image2D out_img;
    layout(binding=2)
    uniform sampler2D img;

    float luminance(vec3 c) {
      return 0.2126 * c.x + 0.7152 * c.y + 0.0722 * c.z;
    }

    shared float[64] stage_sum;

    void main() {
      uvec3 global_id = gl_GlobalInvocationID;
      uvec3 workgrp_id = gl_WorkGroupID;
      uvec3 local_id = gl_LocalInvocationID;
      uint local_idx = gl_LocalInvocationIndex;
      uint W = u.size.x;
      uint H = u.size.y;

      vec2 size_coe = 1.0f / (vec2(W, H));

      vec3 sum = vec3(0.0f, 0.0f, 0.0f);
      for (uint w = local_id.x * 4; w < W; w += 32) {
        for (uint h = local_id.y * 4; h < H; h += 32) {
          sum += 4.0f * texture(img, vec2(w + 1.0, h + 1.0) * size_coe).rgb;
          sum += 4.0f * texture(img, vec2(w + 3.0, h + 1.0) * size_coe).rgb;
          sum += 4.0f * texture(img, vec2(w + 1.0, h + 3.0) * size_coe).rgb;
          sum += 4.0f * texture(img, vec2(w + 3.0, h + 3.0) * size_coe).rgb;
        }
      }
      stage_sum[local_idx] = luminance(sum.rgb);

      memoryBarrierShared();
      barrier();
      if (local_idx > 8) { return; }
      stage_sum[local_idx] +=
        stage_sum[local_idx + 8] +
        stage_sum[local_idx + 16] +
        stage_sum[local_idx + 24] +
        stage_sum[local_idx + 32] +
        stage_sum[local_idx + 40] +
        stage_sum[local_idx + 48] +
        stage_sum[local_idx + 56];

      memoryBarrierShared();
      barrier();
      if (local_idx > 0) { return; }
      float out_sum =
        stage_sum[0] + stage_sum[1] + stage_sum[2] + stage_sum[3] +
        stage_sum[4] + stage_sum[5] + stage_sum[6] + stage_sum[7];
      out_sum = out_sum / float(W * H);

      imageStore(out_img, ivec2(workgrp_id.xy), out_sum.xxxx);
    }
  )";

  static glslang::ComputeSpirvArtifact art =
    glslang::compile_comp(comp_src, "main");

  static scoped::Task task = CTXT.build_comp_task("avg")
    .comp(art.comp_spv)
    .rsc(L_RESOURCE_TYPE_UNIFORM_BUFFER)
    .rsc(L_RESOURCE_TYPE_STORAGE_IMAGE)
    .rsc(L_RESOURCE_TYPE_SAMPLED_IMAGE)
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

  scoped::Image out_img = CTXT.build_img()
    .width(1)
    .height(1)
    .fmt(fmt::L_FORMAT_R32_SFLOAT)
    .storage()
    .build();

  float out_avg;
  scoped::Buffer out_buf = CTXT.build_buf()
    .size_like(out_avg)
    .storage()
    .read_back()
    .build();

  auto invoke = task.build_comp_invoke()
    .rsc(params_buf.view())
    .rsc(out_img.view())
    .rsc(img.view())
    .workgrp_count(1, 1, 1)
    .is_timed()
    .build();

  invoke.submit().wait();

  CTXT.build_trans_invoke()
    .src(out_img.view())
    .dst(out_buf.view())
    .build()
    .submit()
    .wait();

  log::info("[", width, "x", height, "] ", invoke.get_time_us(), "us");

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

    if (std::fabs(cpu_res - gpu_res) > 0.1) {
      log::error("expect=", cpu_res, "; actual=", gpu_res);
      panic();
    }
  };

  //run(1, 1);
  //run(2, 1);
  //run(1, 2);
  //run(2, 2);
  run(4, 4);
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
  try {
    guarded_main();
  } catch (const std::exception& e) {
    liong::log::error("application threw an exception");
    liong::log::error(e.what());
    liong::log::error("application cannot continue");
  } catch (...) {
    liong::log::error("application threw an illiterate exception");
  }
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
