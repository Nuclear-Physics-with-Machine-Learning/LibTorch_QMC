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

#include "ModelConfig.h"


struct MLPImpl : torch::nn::Module {

    MLPImpl(){}

    MLPImpl(MLPConfig _cfg)
        : cfg(_cfg)
        , layers(torch::nn::ModuleList())
          // layer1(torch::nn::Linear(input_size, output_size)),
          // layer2(torch::nn::Linear(output_size, output_size)),
          // layer3(torch::nn::Linear(output_size, output_size)),
          // layer4(torch::nn::Linear(output_size, output_size))
    {

        register_module("layers", layers);

        // First in/out:
        int in  = cfg.n_input;
        int out = cfg.n_filters_per_layer;

        // Build up a list of layers:
        for (int i = 0; i < cfg.n_layers; i ++){
            std::string name = "layer" + std::to_string(i);
            auto layer = register_module(name, torch::nn::Linear(in, out));
            layers->push_back(layer);
            in = out;
            if (i == cfg.n_layers - 2) out = cfg.n_output;
        }

        // register_module("layer1", layer1);
        // register_module("layer2", layer2);
        // register_module("layer3", layer3);
        // register_module("layer4", layer4);
    }

    torch::Tensor forward(torch::Tensor x){

        // auto s = torch::tanh(layer1(x));
        // s = torch::tanh(layer2(s));
        // s = torch::tanh(layer3(s));
        // s = layer4(s);  
        // return s;
        return x;
    }

    MLPConfig cfg;
    torch::nn::ModuleList layers;
};

TORCH_MODULE(MLP);



// #include <pybind11/pybind11.h>
// // Declare python binding function
// void init_MLP(pybind11::module m);

