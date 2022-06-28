#include <catch2/catch_test_macros.hpp>
#include <catch2/benchmark/catch_benchmark_all.hpp>

#include <torch/torch.h>

#include "config/config.h"
#include "models/models.h"
#include "sampler/sampler.h"
#include "hamiltonians/NuclearHamiltonian.h"

#include "optimizers/JacobianCalculator.h"



TEST_CASE("Derivatives Benchmark", "[derivatives]"){


    Config cfg;

    // We don't want to over-complicate the computations here, so simplify a bit:
    cfg.sampler.n_particles = 50;
    cfg.wavefunction.correlator_config.individual_config.n_layers = 2;
    cfg.wavefunction.correlator_config.individual_config.n_filters_per_layer = 8;

    cfg.wavefunction.correlator_config.aggregate_config.n_layers = 2;
    cfg.wavefunction.correlator_config.aggregate_config.n_filters_per_layer = 8;


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

    NuclearHamiltonian h(cfg.hamiltonian, options);


    ManyBodyWavefunction wavefunction = ManyBodyWavefunction(
        cfg.wavefunction, options);
    wavefunction -> to(torch::kDouble);
    wavefunction -> to(options.device());

    auto x = sampler.sample_x();

    // Warm up
    auto derivatives =  h.compute_derivatives(wavefunction, x);


    BENCHMARK("Hamiltonian Derivatives"){
        return h.compute_derivatives(wavefunction, x);
    };

}

/*

TEST_CASE("Jacobian Benchmark"){


    Config cfg;

    // We don't want to over-complicate the computations here, so simplify a bit:
    cfg.sampler.n_walkers = 5;
    cfg.wavefunction.correlator_config.individual_config.n_layers = 2;
    cfg.wavefunction.correlator_config.individual_config.n_filters_per_layer = 3;

    cfg.wavefunction.correlator_config.aggregate_config.n_layers = 2;
    cfg.wavefunction.correlator_config.aggregate_config.n_filters_per_layer = 3;

    cfg.wavefunction.correlator_config.latent_space = 3;
    cfg.wavefunction.mean_subtract = false;
    cfg.validate_config();


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

    JacobianCalculator jac_calc(options);
    ManyBodyWavefunction wavefunction = ManyBodyWavefunction(
        cfg.wavefunction, options);
    wavefunction -> to(torch::kDouble);
    wavefunction -> to(options.device());

    // Default to parallelization of 1, can increase later
    jac_calc.set_parallelization(wavefunction, 1);

    auto x = sampler.sample_x();

    float KICK_SIZE = 1e-3;

    // The numerical jacobian is what we benchmark against:
    auto jac_numerical = jac_calc.numerical_jacobian(x, wavefunction, KICK_SIZE);


    auto jac_rev = jac_calc.jacobian_reverse(x, wavefunction);

    REQUIRE(jac_numerical.sizes() == jac_rev.sizes());
    // Find the largerst absolute difference between the numerical and analytical deriv:
    auto diff = torch::max(torch::abs(jac_numerical - jac_rev));
    auto pass = (diff < 2*1e-3).to(torch::kBool).cpu();
    REQUIRE(pass.data_ptr<bool>()[0]);


    auto jac_rev_batched = jac_calc.batch_jacobian_reverse(x, wavefunction);
    REQUIRE(jac_numerical.sizes() == jac_rev_batched.sizes());
    diff = torch::max(torch::abs(jac_numerical - jac_rev_batched));
    pass = (diff < 2*1e-3).to(torch::kBool).cpu();
    REQUIRE(pass.data_ptr<bool>()[0]);



    auto jac_fwd = jac_calc.jacobian_forward(x, wavefunction);
    REQUIRE(jac_numerical.sizes() == jac_fwd.sizes());
    diff = torch::max(torch::abs(jac_numerical - jac_fwd));
    pass = (diff < 2*1e-3).to(torch::kBool).cpu();
    REQUIRE(pass.data_ptr<bool>()[0]);

    auto jac_fwd_batched = jac_calc.batch_jacobian_forward(x, wavefunction);

    REQUIRE(jac_numerical.sizes() == jac_fwd_batched.sizes());
    diff = torch::max(torch::abs(jac_numerical - jac_fwd_batched));
    pass = (diff < 2*1e-3).to(torch::kBool).cpu();
    REQUIRE(pass.data_ptr<bool>()[0]);
}



*/