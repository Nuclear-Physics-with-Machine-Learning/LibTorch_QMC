// /**
//  * @defgroup   MANYBODYWAVEFUNCTION Many Body Wavefunction
//  *
//  * @brief      This file implements Many Body Wavefunction.
//  *
//  * @author     Corey.adams
//  * @date       2022
//  */

// #pragma once


// struct ManyBodyWavefunctionImpl : torch::nn::Module {
//     ManyBodyWavefunctionImpl(int64_t input_size, int64_t latent_size)
//         : input_size(input_size),
//           latent_size(latent_size),
//           individual_net(input_size, latent_size),
//           aggregate_net(latent_size, 1)
//     {
//         register_module("individual_net", individual_net);
//         register_module("aggregate_net", aggregate_net);
//     }

//     torch::Tensor forward(torch::Tensor x){

//         // The input will be of the shape [N_walkers, n_particles, n_dim]

//         int64_t n_walkers = x.sizes()[0];
//         auto n_particles = x.sizes()[1];

//         torch::Tensor summed_output = torch::zeros({n_walkers, latent_size});

//         // Chunk the tensor into n_particle pieces:
//         // std::vector<torch::Tensor> torch::chunk(x, n_particles, 1);


//         for (int i = 0; i < n_particles; i++){
//             // index = at::indexing::TensorIndex(int(i));
//             auto s = x.index({Slice(), i});
//             summed_output += individual_net(s);
//         }

//         summed_output = aggregate_net(summed_output);

//         return summed_output;
//     }

//     int64_t input_size, latent_size;
//     DeepSetsCorrelator dsc;
// };

// TORCH_MODULE(ManyBodyWavefunction);
