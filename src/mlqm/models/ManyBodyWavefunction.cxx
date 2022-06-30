#include "ManyBodyWavefunction.h"

ManyBodyWavefunctionImpl::ManyBodyWavefunctionImpl(
    ManyBodyWavefunctionConfig _cfg, 
    torch::TensorOptions options)
    : cfg(_cfg),
      dsc(_cfg.correlator_config, options),
      opts(options)
{
    register_module("dsc", dsc);
    // register_module("spatial_nets", spatial_nets);

    // // Initialize a spatial net for every particle:
    // for (int64_t i = 0; i < n_particles; i++){
    //     auto spatial_net = MLP(cfg.spatial_config, options);
    //     spatial_nets->push_back(spatial_net);
    // }

}

torch::Tensor ManyBodyWavefunctionImpl::forward(torch::Tensor x){

    // The input will be of the shape [N_walkers, n_particles, n_dim]
    // c10::InferenceMode guard(true);
    // mean subtract if needed:
    if (cfg.mean_subtract){
        // Average over particles
        auto mean = torch::mean(x, {1});

        x = x - mean.reshape({x.sizes()[0], 1, x.sizes()[2]});
    }

    auto correlation =  dsc(x);

    return correlation;

}
