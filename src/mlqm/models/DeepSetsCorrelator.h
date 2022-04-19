/**
 * @defgroup   DEEPSETSCORRELATOR Deep Sets Correlator
 *
 * @brief      This file implements Deep Sets Correlator.
 *
 * @author     Corey.adams
 * @date       2022
 */

#pragma once

#include <torch/torch.h>

#include "MLP.h"

using namespace torch::indexing;

struct DeepSetsCorrelatorImpl : torch::nn::Module {
    DeepSetsCorrelatorImpl(DeepSetsCorrelaterConfig _cfg): 
    cfg(_cfg),
    individual_net(_cfg.individual_config),
    aggregate_net(_cfg.aggregate_config)
    {
        register_module("individual_net", individual_net);
        register_module("aggregate_net", aggregate_net);

    }

    torch::Tensor forward(torch::Tensor x){

        // The input will be of the shape [N_walkers, n_particles, n_dim]

        int64_t n_walkers = x.sizes()[0];
        auto n_particles = x.sizes()[1];

        torch::Tensor summed_output = torch::zeros({n_walkers, cfg.latent_space});

        // Chunk the tensor into n_particle pieces:
        // std::vector<torch::Tensor> torch::chunk(x, n_particles, 1);


        for (int i = 0; i < n_particles; i++){
            // index = at::indexing::TensorIndex(int(i));
            auto s = x.index({Slice(), i});
            summed_output += individual_net(s);
        }

        summed_output = aggregate_net(summed_output);

        return summed_output;
    }
    DeepSetsCorrelaterConfig cfg;
    MLP individual_net, aggregate_net;
};

TORCH_MODULE(DeepSetsCorrelator);


// #include <pybind11/pybind11.h>
// // Declare python binding function
// void init_DSC(pybind11::module m);
