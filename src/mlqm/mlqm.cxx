#include <iostream>
#include <chrono>

#include <torch/torch.h>

#include "config/config.h"
#include "models/models.h"
#include "sampler/sampler.h"
#include "hamiltonians/BaseHamiltonian.h"



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

  // Create default tensor options, passed for all tensor creation:

  auto options =
  torch::TensorOptions()
    .dtype(torch::kFloat32)
    .layout(torch::kStrided);

  // Default device?
  // torch::Device device(torch::kCPU);
  if (torch::cuda::is_available()) {
    std::cout << "CUDA is available! Training on GPU." << std::endl;
    options = options.device(torch::kCUDA);
  }

  std::cout << "Device selected: " << options.device() << std::endl;
  std::cout << "Requires grad?: " << options.requires_grad() << std::endl;

  // Create a sampler:
  MetropolisSampler sampler(cfg.sampler, options);

  // Create a model:
  DeepSetsCorrelator dsc = DeepSetsCorrelator(cfg.wavefunction, options);
  dsc -> to(options.device());


  // // Initialize random input:
  auto input = sampler.sample();

  std::cout << "input.sizes(): "<< input.sizes() << std::endl;


  // Run the model forward:

  auto start = std::chrono::high_resolution_clock::now();

  torch::Tensor acceptance = sampler.kick(500, dsc);
  auto stop = std::chrono::high_resolution_clock::now();

  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);

  std::cout << "acceptance : " << acceptance << std::endl;
  std::cout << "Kick time: " << (duration.count() / 1000.) << " seconds" << std::endl;

  start = std::chrono::high_resolution_clock::now();

  acceptance = sampler.kick(500, dsc);
  stop = std::chrono::high_resolution_clock::now();

  duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);

  std::cout << "acceptance : " << acceptance << std::endl;
  std::cout << "Kick time: " << (duration.count() / 1000.) << " seconds" << std::endl;

  BaseHamiltonian h;

  // auto junk = h.energy(dsc, sampler.sample());

  return 0;
}
