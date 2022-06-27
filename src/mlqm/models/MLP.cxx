#include "MLP.h"


MLPImpl::MLPImpl(MLPConfig _cfg, torch::TensorOptions options)
        : cfg(_cfg)
        , layers(torch::nn::Sequential())
        , opts(options)
{

    register_module("layers", layers);

    // First in/out:
    int in  = cfg.n_input;
    int out = cfg.n_filters_per_layer;

    // Build up a list of layers:
    for (int i = 0; i < cfg.n_layers; i ++){
        std::string name = "layer" + std::to_string(i);
        auto layer = torch::nn::Linear(in, out);
        // auto layer = register_module(name, torch::nn::Linear(in, out));
        // Use the specified precision:
        // layer ->to(opts.dtype());
        layers->push_back(layer);
        // skip the activation on the last layer:
        if (i != cfg.n_layers -1){
            layers->push_back(torch::nn::Tanh());
        }
        in = out;
        if (i == cfg.n_layers - 2) out = cfg.n_output;
    }

    // register_module("layer1", layer1);
    // register_module("layer2", layer2);
    // register_module("layer3", layer3);
    // register_module("layer4", layer4);
}


torch::Tensor MLPImpl::forward(torch::Tensor x){
    // c10::InferenceMode guard(true);
    return layers->forward(x);
}
