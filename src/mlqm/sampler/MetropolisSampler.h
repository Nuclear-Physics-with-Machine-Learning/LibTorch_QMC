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
#include "mlqm/models/DeepSetsCorrelator.h"

class MetropolisSampler
{
public:
    MetropolisSampler(SamplerConfig cfg, torch::TensorOptions options);
    ~MetropolisSampler(){}

    torch::Tensor kick(int n_kicks, DeepSetsCorrelator wavefunction);

    torch::Tensor sample(){return torch::clone(x);}

private:

    // Holds a state: the sampled tensors
    torch::Tensor x;

    SamplerConfig cfg;

    torch::TensorOptions opts;

};
