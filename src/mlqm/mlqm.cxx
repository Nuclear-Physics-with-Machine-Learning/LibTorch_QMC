#include <iostream>
#include <chrono>

#include <torch/torch.h>

#include "config/config.h"
#include "models/models.h"
#include "sampler/sampler.h"
#include "hamiltonians/NuclearHamiltonian.h"
#include "optimizers/BaseOptimizer.h"



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


  PLOG_INFO << "Config: " <<  j_cfg.dump(2);

  // Convert the json object into a config object:

  Config cfg = j_cfg.get<Config>();

  // How many threads?
  // at::set_num_interop_threads(4);
  // at::set_num_threads(1);
  // PLOG_INFO << "Using this many threads for interop: " <<  at::get_num_interop_threads();


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

  PLOG_INFO << "Device selected: " << options.device();

  // Create the base algorithm:
  BaseOptimizer optim(cfg, options); 


  return 0;
}
