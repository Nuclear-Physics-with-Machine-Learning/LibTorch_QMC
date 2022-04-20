#include <iostream>

#include <torch/torch.h>

#include "config/config.h"
#include "models/models.h"
#include "sampler/sampler.h"




int main(int argc, char* argv[]) {

  // The only argument this program should accept is a json configuration file.

  if (argc == 1){
    std::cout << "Must supply a configuration file on cmdline!  Exiting!" << std::endl;
    return 1;
  }

  // We load it directly with json:
  std::string fname(argv[argc - 1]);

  std::cout << "config file is: " << fname << std::endl;

  // Read the configuration from file:
  std::ifstream ifs(fname);

  if (! ifs.good()){
    std::cout << "ERROR!  configuration file does not exist." << std::endl;
    return 2;
  }

  // Init the json object:
  json j_cfg;

  try{
    j_cfg = json::parse(ifs);
  }
  catch (...){
    std::cout << "ERROR!  json file might have an error." << std::endl;
  }


  std::cout << "Config: " <<  j_cfg << std::endl;

  // Convert the json object into a config object:

  Config cfg = j_cfg.get<Config>();


  // Create a sampler:
  MetropolisSampler sampler(cfg.sampler);

  // Create a model:
  DeepSetsCorrelator dsc = DeepSetsCorrelator(cfg.wavefunction);

  // // Initialize random input:
  auto input = sampler.sample();

  // // Run the model forward:

  torch::Tensor acceptance = sampler.kick(500, dsc);


  std::cout << "acceptance : " << acceptance << std::endl;
  

  return 0;
}