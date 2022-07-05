#include <catch2/catch_test_macros.hpp>
#include <catch2/benchmark/catch_benchmark_all.hpp>

#include <torch/torch.h>

#include "config/config.h"
#include "models/models.h"
#include "sampler/sampler.h"
#include "hamiltonians/NuclearHamiltonian.h"

#include "optimizers/JacobianCalculator.h"

Config benchmark_config(){
    Config cfg;

    // We don't want to over-complicate the computations here, so simplify a bit:
    cfg.sampler.n_particles = 200;
    cfg.wavefunction.correlator_config.individual_config.n_layers = 2;
    cfg.wavefunction.correlator_config.individual_config.n_filters_per_layer = 8;

    cfg.wavefunction.correlator_config.aggregate_config.n_layers = 2;
    cfg.wavefunction.correlator_config.aggregate_config.n_filters_per_layer = 8;

    return cfg;
}

torch::TensorOptions default_options(){

    auto options = torch::TensorOptions()
        .dtype(torch::kFloat64)
        .layout(torch::kStrided);

    // Default device?
    // torch::Device device(torch::kCPU);
    if (torch::cuda::is_available()) {
        options = options.device(torch::kCUDA);
    }
    return options;
}

TEST_CASE("Derivatives Benchmark"){

    auto cfg = benchmark_config();
    auto options = default_options();

    // Turn off inference mode at the highest level
    c10::InferenceMode guard(false);

    // Create a sampler:
    MetropolisSampler sampler(cfg.sampler, options);

    NuclearHamiltonian h(cfg.hamiltonian, options);


    ManyBodyWavefunction wavefunction = ManyBodyWavefunction(
        cfg.wavefunction, cfg.sampler, options);
    wavefunction -> to(torch::kDouble);
    wavefunction -> to(options.device());

    auto x = sampler.sample_x();
    auto spin = sampler.sample_spin();
    auto isospin = sampler.sample_isospin();

    // Warm up
    auto derivatives =  h.compute_derivatives(wavefunction, x, spin, isospin);


    BENCHMARK("Hamiltonian Derivatives"){
        return h.compute_derivatives(wavefunction, x, spin, isospin);
    };

}



TEST_CASE("JacobianBenchmarkForward"){
    auto cfg = benchmark_config();
    auto options = default_options();


    // Turn off inference mode at the highest level
    c10::InferenceMode guard(false);

    // Create a sampler:
    MetropolisSampler sampler(cfg.sampler, options);

    JacobianCalculator jac_calc(options);
    ManyBodyWavefunction wavefunction = ManyBodyWavefunction(
        cfg.wavefunction, cfg.sampler, options);
    wavefunction -> to(torch::kDouble);
    wavefunction -> to(options.device());

    // Default to parallelization of 1, can increase later
    jac_calc.set_parallelization(wavefunction, 1);

    auto x = sampler.sample_x();
    auto spin = sampler.sample_spin();
    auto isospin = sampler.sample_isospin();

    BENCHMARK("JacobianForward"){
        auto jac_rev = jac_calc.jacobian_forward(x, spin, isospin, wavefunction);
        return jac_rev;
    };

}

TEST_CASE("JacobianBenchmarkReverse"){
    auto cfg = benchmark_config();
    auto options = default_options();


    // Turn off inference mode at the highest level
    c10::InferenceMode guard(false);

    // Create a sampler:
    MetropolisSampler sampler(cfg.sampler, options);

    JacobianCalculator jac_calc(options);
    ManyBodyWavefunction wavefunction = ManyBodyWavefunction(
        cfg.wavefunction, cfg.sampler, options);
    wavefunction -> to(torch::kDouble);
    wavefunction -> to(options.device());

    // Default to parallelization of 1, can increase later
    jac_calc.set_parallelization(wavefunction, 1);

    auto x = sampler.sample_x();
    auto spin = sampler.sample_spin();
    auto isospin = sampler.sample_isospin();

    BENCHMARK("JacobianReverse"){
        auto jac_rev = jac_calc.jacobian_reverse(x, spin, isospin, wavefunction);
        return jac_rev;
    };

}

TEST_CASE("JacobianBenchmarkReverseBatched"){
    auto cfg = benchmark_config();
    auto options = default_options();


    // Turn off inference mode at the highest level
    c10::InferenceMode guard(false);

    // Create a sampler:
    MetropolisSampler sampler(cfg.sampler, options);

    JacobianCalculator jac_calc(options);
    ManyBodyWavefunction wavefunction = ManyBodyWavefunction(
        cfg.wavefunction, cfg.sampler, options);
    wavefunction -> to(torch::kDouble);
    wavefunction -> to(options.device());

    // Default to parallelization of 1, can increase later
    jac_calc.set_parallelization(wavefunction, 10);

    auto x = sampler.sample_x();
    auto spin = sampler.sample_spin();
    auto isospin = sampler.sample_isospin();

    BENCHMARK("BatchJacobianReverse"){
        auto jac_rev = jac_calc.batch_jacobian_reverse(x, spin, isospin, wavefunction);
        return jac_rev;
    };

}


TEST_CASE("JacobianBenchmarkForwardBatched"){
    auto cfg = benchmark_config();
    auto options = default_options();


    // Turn off inference mode at the highest level
    c10::InferenceMode guard(false);

    // Create a sampler:
    MetropolisSampler sampler(cfg.sampler, options);

    JacobianCalculator jac_calc(options);
    ManyBodyWavefunction wavefunction = ManyBodyWavefunction(
        cfg.wavefunction, cfg.sampler, options);
    wavefunction -> to(torch::kDouble);
    wavefunction -> to(options.device());

    // Default to parallelization of 1, can increase later
    jac_calc.set_parallelization(wavefunction, 10);

    auto x = sampler.sample_x();
    auto spin = sampler.sample_spin();
    auto isospin = sampler.sample_isospin();

    BENCHMARK("BatchJacobianForward"){
        auto jac_rev = jac_calc.batch_jacobian_forward(x, spin, isospin, wavefunction);
        return jac_rev;
    };

}
