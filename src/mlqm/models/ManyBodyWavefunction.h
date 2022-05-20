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
    ManyBodyWavefunctionImpl(ManyBodyWavefunctionConfig _cfg, torch::TensorOptions options, int64_t n_particles)
        : cfg(_cfg),
          dsc(_cfg.correlator_config, options),
          opts(options)
    {
        register_module("dsc", dsc);
        register_module("spatial_nets", spatial_nets);

        // Initialize a spatial net for every particle:
        for (int64_t i = 0; i < n_particles; i++){
            auto spatial_net = MLP(cfg.spatial_config, options);
            spatial_nets->push_back(spatial_net);
        }

    }

    torch::Tensor forward(torch::Tensor x);

    ManyBodyWavefunctionConfig cfg;
    DeepSetsCorrelator dsc;
    torch::nn::ModuleList spatial_nets;
    torch::TensorOptions opts;
};

TORCH_MODULE(ManyBodyWavefunction);
