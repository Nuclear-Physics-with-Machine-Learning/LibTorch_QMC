#include "DeepSetsCorrelator.h"

#ifdef OPENMP_FOUND
#include <omp.h>
#endif

DeepSetsCorrelatorImpl::DeepSetsCorrelatorImpl(DeepSetsCorrelaterConfig _cfg, torch::TensorOptions options):
    cfg(_cfg),
    individual_net(_cfg.individual_config, options),
    aggregate_net(_cfg.aggregate_config, options),
    opts(options)
{
    register_module("individual_net", individual_net);
    register_module("aggregate_net", aggregate_net);

}

torch::Tensor DeepSetsCorrelatorImpl::forward(torch::Tensor x){

    // The input will be of the shape [N_walkers, n_particles, n_dim]
    // c10::InferenceMode guard(true);

    auto individual_net_response = individual_net(x);

    auto summed_output = individual_net_response.sum(1);
    // std::cout << "summed_output.sizes(): "<< summed_output.sizes() << std::endl;

/*
    int64_t n_walkers = x.sizes()[0];
    int64_t n_particles = x.sizes()[1];
    int64_t n_dim = x.sizes()[2];

    torch::Tensor summed_output = torch::zeros({n_walkers, cfg.individual_config.n_output}, opts);
    for (int i = 0; i < n_particles; i++){
        // index = at::indexing::TensorIndex(int(i));
        auto s = x.index({Slice(), i});
        // std::cout << "s.sizes(): "<< s.sizes() << std::endl;
        // std::cout << "individual_net(s).sizes(): "<< individual_net(s).sizes() << std::endl;

        summed_output += individual_net(s);
    }
*/

    summed_output = aggregate_net(summed_output);

    summed_output = summed_output.reshape({x.sizes()[0],});

    // Calculate the confinement:
    torch::Tensor exp = x.pow(2).sum({1,2});
    auto confinement = -cfg.confinement * exp;


    return torch::exp(summed_output + confinement);

}
