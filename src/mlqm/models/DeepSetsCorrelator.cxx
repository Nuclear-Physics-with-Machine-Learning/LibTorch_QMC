#include "DeepSetsCorrelator.h"

#ifdef OPENMP_FOUND
#include <omp.h>
#endif

// void init_DSC(pybind11::module m){
//     using Class = DeepSetsCorrelatorImpl;
//     pybind11::class_<Class> DSC(m, "DeepSetsCorrelator");
//     DSC.def(pybind11::init<int64_t, int64_t>());

//     DSC.def("forward",        &Class::forward);

// }

torch::Tensor DeepSetsCorrelatorImpl::forward(torch::Tensor x){
    // std::cout << "DSC x device: " << x.device() << std::endl;

    // The input will be of the shape [N_walkers, n_particles, n_dim]
    // c10::InferenceMode guard(true);


    int64_t n_walkers = x.sizes()[0];
    int64_t n_particles = x.sizes()[1];
    int64_t n_dim = x.sizes()[2];

    // std::cout << "summed_output.device(): "<< summed_output.device() << std::endl;

    // Chunk the tensor into n_particle pieces:
    // std::vector<torch::Tensor> torch::chunk(x, n_particles, 1);

    auto individual_net_response = individual_net(x);

    auto summed_output = individual_net_response.sum(1);
    // std::cout << "summed_output.sizes(): "<< summed_output.sizes() << std::endl;
     

    // torch::Tensor summed_output = torch::zeros({n_walkers, cfg.individual_config.n_output}, opts);
    // for (int i = 0; i < n_particles; i++){
    //     // index = at::indexing::TensorIndex(int(i));
    //     auto s = x.index({Slice(), i});
    //     // std::cout << "s.sizes(): "<< s.sizes() << std::endl;
    //     // std::cout << "individual_net(s).sizes(): "<< individual_net(s).sizes() << std::endl;

    //     summed_output += individual_net(s);
    // }

/*
    std::vector<torch::Tensor> individual_outputs;
    individual_outputs.resize(n_particles);
    #pragma omp parallel for num_threads(8)
    for (int i = 0; i < n_particles; i++){
        // index = at::indexing::TensorIndex(int(i));
        auto s = x.index({Slice(), i});
        // std::cout << "s.sizes(): "<< s.sizes() << std::endl;
        // std::cout << "individual_net(s).sizes(): "<< individual_net(s).sizes() << std::endl;

        individual_outputs[i] = individual_net(s);
    }

    // Sum over all the
    // std::cout << "summed_output.sizes(): "<< summed_output.sizes() << std::endl;
    torch::Tensor summed_output = torch::zeros({n_walkers, cfg.individual_config.n_output}, opts);

    // Add all the individual outputs to the summed output:
    for (int i = 0; i < n_particles; i++){
        summed_output += individual_outputs[i];
    }
*/

    summed_output = aggregate_net(summed_output);

    // Calculate the confinement:
    torch::Tensor exp = x.pow(2).sum({1,2});
    auto confinement = -cfg.confinement * exp;
    // std::cout << "confinement sizes: " << confinement.sizes() << std::endl;

    return summed_output;

}
