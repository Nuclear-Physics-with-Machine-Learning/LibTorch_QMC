#include <catch2/catch_test_macros.hpp>

#include <torch/torch.h>

#include "config/config.h"
#include "models/models.h"
#include "sampler/sampler.h"
#include "hamiltonians/NuclearHamiltonian.h"


TEST_CASE( "Sampler size is correct", "[sampler]"){

    auto sampler_config = SamplerConfig();
    sampler_config.n_walkers = 100;
    sampler_config.n_particles = 3;
    sampler_config.n_dim = 3;

    auto options = torch::TensorOptions()
        .dtype(torch::kFloat32)
        .layout(torch::kStrided);

    // Default device?
    // torch::Device device(torch::kCPU);
    if (torch::cuda::is_available()) {
        options = options.device(torch::kCUDA);
    }

    // Turn off inference mode at the highest level
    c10::InferenceMode guard(false);

    // Create a sampler:
    MetropolisSampler sampler(sampler_config, options);

    auto x = sampler.sample();

    REQUIRE (x.sizes()[0] == 100);
    REQUIRE (x.sizes()[1] == 3);
    REQUIRE (x.sizes()[2] == 3);
}