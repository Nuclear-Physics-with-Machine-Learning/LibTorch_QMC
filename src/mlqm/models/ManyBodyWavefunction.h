/**
 * @defgroup   MANYBODYWAVEFUNCTION Many Body Wavefunction
 *
 * @brief      This file implements Many Body Wavefunction.
 *
 * @author     Corey.adams
 * @date       2022
 */

#pragma once

#include "mlqm/config/ModelConfig.h"
#include "DeepSetsCorrelator.h"
#include "MLP.h"

struct ManyBodyWavefunctionImpl : torch::nn::Module {
    ManyBodyWavefunctionImpl(
        ManyBodyWavefunctionConfig _cfg,
        torch::TensorOptions options
    );

    torch::Tensor forward(torch::Tensor x);

    ManyBodyWavefunctionConfig cfg;
    DeepSetsCorrelator dsc;
    // torch::nn::ModuleList spatial_nets;
    torch::TensorOptions opts;
};

TORCH_MODULE(ManyBodyWavefunction);
