/**
 * @defgroup   MLP MLP
 *
 * @brief      This file implements MLP.
 *
 * @author     Corey.adams
 * @date       2022
 */

#pragma once

#include <torch/torch.h>
// #include <torch/extension.h>
using namespace torch::indexing;

#include "mlqm/config/ModelConfig.h"


struct MLPImpl : torch::nn::Module {

    MLPImpl(){}

    MLPImpl(MLPConfig _cfg, torch::TensorOptions options);


    torch::Tensor forward(torch::Tensor x);

    MLPConfig cfg;
    torch::nn::Sequential layers;
    torch::TensorOptions opts;
};

TORCH_MODULE(MLP);



// #include <pybind11/pybind11.h>
// // Declare python binding function
// void init_MLP(pybind11::module m);
