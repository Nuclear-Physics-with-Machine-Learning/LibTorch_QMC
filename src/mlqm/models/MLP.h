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

struct MLPImpl : torch::nn::Module {
    MLPImpl(int64_t input_size, int64_t output_size)
        : output_size(output_size),
          layer1(torch::nn::Linear(input_size, output_size)),
          layer2(torch::nn::Linear(output_size, output_size)),
          layer3(torch::nn::Linear(output_size, output_size)),
          layer4(torch::nn::Linear(output_size, output_size))
    {
        register_module("layer1", layer1);
        register_module("layer2", layer2);
        register_module("layer3", layer3);
        register_module("layer4", layer4);
    }

    torch::Tensor forward(torch::Tensor x){

        auto s = torch::tanh(layer1(x));
        s = torch::tanh(layer2(s));
        s = torch::tanh(layer3(s));
        s = layer4(s);  
        return s;
    }

    int64_t output_size;
    torch::nn::Linear layer1, layer2, layer3, layer4;
};

TORCH_MODULE(MLP);



// #include <pybind11/pybind11.h>
// // Declare python binding function
// void init_MLP(pybind11::module m);

