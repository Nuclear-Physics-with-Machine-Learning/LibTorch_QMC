#include <iostream>

#include <torch/torch.h>
// #include <torch/extension.h>
// #include <pybind11/pybind11.h>

#include "models/models.h"
#include "sampler/sampler.h"

// int main() {
//   IndividualNet individual_net(3,16);

//   auto input = torch::randn({500,2,3});

//   auto output = individual_net->forward(input);

//   std::cout << "OUtput: "<< output << std::endl;
// }

 
// PYBIND11_MODULE(mlqm, m) {
//   // torch::python::init_bindings(mlqm);
//   // init_models(m);
// }




int main() {

  // Create a sampler:
  SamplerConfig sc;

  MetropolisSampler sampler(sc);

  // Create a model:

  DeepSetsCorrelaterConfig dsc_cfg;

  // int64_t latent_size = 16;

  DeepSetsCorrelator dsc = DeepSetsCorrelator(dsc_cfg);

  // // Initialize random input:
  auto input = sampler.sample();

  // // Run the model forward:

  torch::Tensor acceptance = sampler.kick(500, dsc);


  std::cout << "acceptance : " << acceptance << std::endl;

}