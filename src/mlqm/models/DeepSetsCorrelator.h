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
    DeepSetsCorrelatorImpl(DeepSetsCorrelaterConfig _cfg, torch::TensorOptions options);

    torch::Tensor forward(torch::Tensor x);

    DeepSetsCorrelaterConfig cfg;
    MLP individual_net, aggregate_net;
    torch::TensorOptions opts;
};

TORCH_MODULE(DeepSetsCorrelator);


// #include <pybind11/pybind11.h>
// // Declare python binding function
// void init_DSC(pybind11::module m);
