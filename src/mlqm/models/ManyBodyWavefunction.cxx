
    // {

    //     // The input will be of the shape [N_walkers, n_particles, n_dim]

    //     int64_t n_walkers = x.sizes()[0];
    //     auto n_particles = x.sizes()[1];

    //     torch::Tensor summed_output = torch::zeros({n_walkers, latent_size});

    //     // Chunk the tensor into n_particle pieces:
    //     // std::vector<torch::Tensor> torch::chunk(x, n_particles, 1);


    //     for (int i = 0; i < n_particles; i++){
    //         // index = at::indexing::TensorIndex(int(i));
    //         auto s = x.index({Slice(), i});
    //         summed_output += individual_net(s);
    //     }

    //     summed_output = aggregate_net(summed_output);

    //     return summed_output;
    // }