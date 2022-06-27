#include <iostream>
#include <chrono>

#include <torch/torch.h>

#include "config/config.h"
#include "optimizers/BaseOptimizer.h"

#include <nlohmann/json.hpp>

// We include the logger here because config.h gets included everywhere
#define PLOG_OMIT_LOG_DEFINES
#include <plog/Log.h> // Step1: include the headers
#include "plog/Initializers/RollingFileInitializer.h"
#include "plog/Appenders/ConsoleAppender.h"

// FOr logging:
// #include "tensorboard_logger.h"

int main(int argc, char* argv[]) {

  /////////////////////////////////////////////
  // Initialize the logger
  /////////////////////////////////////////////

  // Create the 1st appender, this one goes to file.
  static plog::RollingFileAppender<plog::CsvFormatter>
    fileAppender("process.csv", 8000, 3);
  // Create the 2nd appender, this one prints to console.
  static plog::ConsoleAppender<plog::TxtFormatter> consoleAppender;
  // Initialize the logger with the both appenders.
  plog::init(plog::debug, &fileAppender).addAppender(&consoleAppender);

  // The only argument this program should accept is a json configuration file.

  if (argc == 1){
    PLOG_ERROR << "Must supply a configuration file on cmdline!  Exiting!";
    return 1;
  }

  // We load it directly with json:
  std::string fname(argv[argc - 1]);

  PLOG_INFO << "config file is: " << fname;

  // Read the configuration from file:
  std::ifstream ifs(fname);

  if (! ifs.good()){
    PLOG_ERROR << "ERROR!  configuration file does not exist.";
    return 2;
  }


  // Init the json object:
  json j_cfg;

  try{
    j_cfg = json::parse(ifs);
  }
  catch (...){
    PLOG_ERROR << "ERROR!  json file might have an error.";
  }



  // Convert the json object into a config object:

  Config cfg = j_cfg.get<Config>();

  // This logic here ensures that the 
  // wavefunctions get populated properly with input/output dimensions.
  cfg.validate_config();
 
  
  json resolved_json = cfg;
  PLOG_INFO << "Config: " <<  resolved_json.dump(2);
  

  // Create default tensor options, passed for all tensor creation:

  auto options =
  torch::TensorOptions()
    .dtype(torch::kFloat64)
    .layout(torch::kStrided);

  // Default device?
  // torch::Device device(torch::kCPU);
  if (torch::cuda::is_available()) {
    PLOG_INFO << "CUDA is available! Training on GPU.";
    options = options.device(torch::kCUDA);
  }

  // Configure a logger:

  // Set number of threads
  // at::set_num_interop_threads(cfg.sampler.n_particles);
  // at::set_num_threads(cfg.sampler.n_particles);


  // TensorBoardLogger logger(cfg.out_dir);

  PLOG_INFO << "Device selected: " << options.device();

  // Create the base algorithm:
  BaseOptimizer optim(cfg, options);

  // // Run the optimization:
  // for (int64_t iteration = 0; iteration < cfg.n_iterations; iteration ++){
  //     auto start = std::chrono::high_resolution_clock::now();
  //     auto metrics = optim.sr_step();
  //     auto stop = std::chrono::high_resolution_clock::now();
  //     std::chrono::duration<double, std::milli>  duration = stop - start;
  //     PLOG_INFO << "energy: " << metrics["energy/energy"];
  //     PLOG_INFO << "Duration: " << duration.count() << "[ms]";
  //     // PLOG_INFO << metrics;
  // }

  return 0;
}
