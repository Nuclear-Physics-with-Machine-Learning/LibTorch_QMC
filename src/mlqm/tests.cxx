#include <catch2/catch_test_macros.hpp>

#include <torch/torch.h>

#include "config/config.h"
#include "models/models.h"
#include "sampler/sampler.h"
#include "hamiltonians/NuclearHamiltonian.h"

static auto default_config = R"(
{
    "sampler": {
        "n_walkers" : 10,
        "n_particles" : 16,
        "n_dim" : 3,
        "n_spin_up" : 0,
        "n_protons" : 0,
        "use_spin" : false,
        "use_isospin" : false,
        "kick_mean" : 0.0,
        "kick_std" : 0.0
    },
    "wavefunction": {
        "spatial_config": {
            "n_input" : 3,
            "n_output" : 32,
            "n_layers" : 4,
            "n_filters_per_layer" : 32,
            "bias" : false,
            "residual" : false
        },
        "correlator_config" : {
            "individual_config" : {
                "n_input" : 3,
                "n_output" : 32,
                "n_layers" : 4,
                "n_filters_per_layer" : 32,
                "bias" : false,
                "residual" : false
            },
            "aggregate_config"  : {
                "n_input" : 32,
                "n_output" : 1,
                "n_layers" : 4,
                "n_filters_per_layer" : 32,
                "bias" : false,
                "residual" : false
            },
            "confinement"       : 0.1,
            "latent_space"      : 32
        }
    }
}
)"_json;



TEST_CASE( "Sampler size is correct", "[sampler]"){
    Config cfg;
    from_json(default_config, cfg);


    auto options = torch::TensorOptions()
        .dtype(torch::kFloat64)
        .layout(torch::kStrided);

    // Default device?
    // torch::Device device(torch::kCPU);
    if (torch::cuda::is_available()) {
        options = options.device(torch::kCUDA);
    }

    // Turn off inference mode at the highest level
    c10::InferenceMode guard(false);

    // Create a sampler:
    MetropolisSampler sampler(cfg.sampler, options);

    auto x = sampler.sample_x();

    REQUIRE (x.sizes()[0] == cfg.sampler.n_walkers);
    REQUIRE (x.sizes()[1] == cfg.sampler.n_particles);
    REQUIRE (x.sizes()[2] == cfg.sampler.n_dim);
}

TEST_CASE( "Differentiation matches numerical approximations", "[wavefunction]"){


    Config cfg;
    from_json(default_config, cfg);

    auto options = torch::TensorOptions()
        .dtype(torch::kFloat64)
        .layout(torch::kStrided);

    // Default device?
    // torch::Device device(torch::kCPU);
    if (torch::cuda::is_available()) {
        options = options.device(torch::kCUDA);
    }

    // Turn off inference mode at the highest level
    c10::InferenceMode guard(false);

    // Create a sampler:
    MetropolisSampler sampler(cfg.sampler, options);

    NuclearHamiltonian h(options);


    ManyBodyWavefunction wavefunction = ManyBodyWavefunction(cfg.wavefunction, options, cfg.sampler.n_particles);
    wavefunction -> to(torch::kDouble);
    wavefunction -> to(options.device());

    auto x = sampler.sample_x();

    auto derivatives =  h.compute_derivatives(wavefunction, x);

    auto w_of_x  = derivatives[0];
    auto dw_dx   = derivatives[1];
    auto d2w_dx2 = derivatives[2];

    float kick_size = 1e-3;

    for (int64_t i_particle = 0; i_particle < cfg.sampler.n_particles; i_particle ++){
        for (int64_t j_dim = 0; j_dim < cfg.sampler.n_dim; j_dim ++){
            // Numerical differentiation along an axis

            // Create a kick for this
            auto kick = torch::zeros_like(x, options);
            kick.index_put_({Slice(), i_particle, j_dim}, kick_size);

            auto x_up = x + kick;
            auto x_down = x - kick;

            auto w_up = wavefunction(x_up);
            auto w_down = wavefunction(x_down);

            // Compute the numerical derivative:
            auto numerical_dw_dx = (w_up - w_down) / (2*kick_size);

            auto this_dwdx_slice = dw_dx.index({Slice(), i_particle, j_dim});

            // Find the largerst absolute difference between the numerical and analytical deriv:
            auto diff = torch::max(torch::abs(this_dwdx_slice - numerical_dw_dx));
            // std::cout << "diff: " << diff << std::endl;
            auto pass = (diff < 2*kick_size).to(torch::kBool).cpu();
            REQUIRE(pass.data_ptr<bool>()[0]);



            // What about the second derivative?
            ///TODO THIS
            //
            auto numerical_d2w_dx2 = (w_up - 2*w_of_x + w_down) / (kick_size*kick_size);

            auto this_d2wdx2_slice = d2w_dx2.index({Slice(), i_particle, j_dim});

            auto diff_2nd = torch::max(torch::abs(this_d2wdx2_slice - numerical_d2w_dx2));
            // std::cout << "dif2nd: " << diff_2nd << std::endl;
            auto pass2nd = (diff_2nd < kick_size).to(torch::kBool).cpu();
            REQUIRE(pass2nd.data_ptr<bool>()[0]);


        }
    }

}
