/**
 * @defgroup   METROPOLISSAMPLER Metropolis Sampler
 *
 * @brief      This file implements Metropolis Sampler.
 *
 * @author     Corey.adams
 * @date       2022
 */

#pragma once

#include <torch/torch.h>

#include "mlqm/config/SamplerConfig.h"
#include "mlqm/models/ManyBodyWavefunction.h"

class MetropolisSampler
{
public:
    MetropolisSampler(SamplerConfig cfg, torch::TensorOptions options);
    ~MetropolisSampler(){}

    torch::Tensor kick(int n_kicks, ManyBodyWavefunction wavefunction);

    torch::Tensor sample_x();
    torch::Tensor sample_spin();
    torch::Tensor sample_isospin();

    void reset_history();

private:

    // Holds a state: the sampled tensors
    torch::Tensor x;
    torch::Tensor spin;
    torch::Tensor isospin;

    std::vector<torch::Tensor> x_history;
    std::vector<torch::Tensor> spin_history;
    std::vector<torch::Tensor> isospin_history;

    SamplerConfig cfg;

    torch::TensorOptions opts;

    void init_possilble_swaps();

    std::vector<std::pair<int64_t, int64_t>> possible_swaps;

    int64_t n_walkers_opt_shape;

};
