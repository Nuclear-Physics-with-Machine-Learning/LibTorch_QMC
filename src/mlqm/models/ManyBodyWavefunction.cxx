#include "ManyBodyWavefunction.h"

#include <string>

ManyBodyWavefunctionImpl::ManyBodyWavefunctionImpl(
    ManyBodyWavefunctionConfig _cfg, 
    SamplerConfig _sampler_cfg,
    torch::TensorOptions options)
    : cfg(_cfg),
      sampler_cfg(_sampler_cfg),
      dsc(_cfg.correlator_config, options),
      opts(options)
{
    register_module("dsc", dsc);


    spatial_nets.clear();
    // Initialize a spatial net for every particle:
    for (int64_t i = 0; i < sampler_cfg.n_particles; i++){
        auto spatial_net = MLP(cfg.spatial_config, options);
        spatial_nets.push_back(spatial_net);
        register_module("spatial_nets_" + std::to_string(i), spatial_net);
    }



    // Here, we construct, up front, the appropriate spinor (spin + isospin)
    // component of the slater determinant.

    // The idea is that for each row (horizontal), the state in question
    // is ADDED to this matrix to get the right values.

    // We have two of these, one for spin and one for isospin. After the
    // addition of the spinors, apply a DOT PRODUCT between these two matrices
    // and then a DOT PRODUCT to the spatial matrix

    // Since the spinors are getting shuffled by the metropolis alg,
    // It's not relevant what the order is as long as we have
    // the total spin and isospin correct.

    // Since we add the spinors directly to this, we need
    // to have the following mappings:
    // spin up -> 1 for the first n_spin_up state
    // spin up -> 0 for the last nparticles - n_spin_up states
    // spin down -> 0 for the first n_spin_up state
    // spin down -> 1 for the last nparticles - n_spin_up states

    // so, set the first n_spin_up entries to 1, which
    // (when added to the spinor) yields 2 for spin up and 0 for spin down

    // Set the last nparticles - n_spin_up states to -1, which
    // yields 0 for spin up and -2 for spin down.



    if (sampler_cfg.use_spin){
        spin_spinor_2d = torch::zeros({sampler_cfg.n_particles, sampler_cfg.n_particles}, options);
        for (size_t i = 0; i < sampler_cfg.n_spin_up; i ++){
            spin_spinor_2d.index_put_({Slice(0, sampler_cfg.n_spin_up), Slice()}, 1.);
            spin_spinor_2d.index_put_({Slice(sampler_cfg.n_spin_up, None), Slice()}, -1.);
        }
    }


    if (sampler_cfg.use_isospin){
        isospin_spinor_2d = torch::zeros({sampler_cfg.n_particles, sampler_cfg.n_particles}, options);
        for (size_t i = 0; i < sampler_cfg.n_protons; i ++){
            isospin_spinor_2d.index_put_({Slice(0, sampler_cfg.n_protons), Slice()}, 1.);
            isospin_spinor_2d.index_put_({Slice(sampler_cfg.n_protons, None), Slice()}, -1.);
        }
    }

}

torch::Tensor ManyBodyWavefunctionImpl::forward(
    torch::Tensor x, torch::Tensor spin, torch::Tensor isospin){

    // The input will be of the shape [N_walkers, n_particles, n_dim]
    // c10::InferenceMode guard(true);
    // mean subtract if needed:
    if (cfg.mean_subtract){
        // Average over particles
        auto mean = torch::mean(x, {1});

        x = x - mean.reshape({x.sizes()[0], 1, x.sizes()[2]});
    }

    auto correlation =  dsc(x);

    if (sampler_cfg.use_spin || sampler_cfg.use_isospin){

        auto slater = spatial_slater(x);

        // Compute the spin and isospin components of the matrix:
        if (sampler_cfg.use_spin){
            slater = slater * slater_matrix(spin, spin_spinor_2d);
        }
    
        if (sampler_cfg.use_isospin){
            slater = slater * slater_matrix(isospin, isospin_spinor_2d);
        }

        correlation = correlation * torch::linalg::det(slater);

    }


    return correlation;

}

torch::Tensor ManyBodyWavefunctionImpl::spatial_slater(torch::Tensor x){
    // We get individual response for every particle in the slater determinant
    // The axis of the particles is 1 (nwalkers, nparticles, ndim)

    // We also have to build a matrix of size [nparticles, nparticles] which we
    // then take a determinant of.

    // The axes of the determinant are state (changes vertically) and
    // particle (changes horizontally)

    // We need to compute the spatial components.
    // This computes each spatial net (nparticles of them) for each particle (nparticles)
    // You can see how this builds to an output shape of (Np, nP)
    //
    // Flatten the input for a neural network to compute
    // over all particles at once, for each network:

    // The matrix indexes in the end should be:
    // [walker, state, particle]
    // In other words, columns (y, 3rd index) share the particle
    // and rows (x, 2nd index) share the state

    // First, calculate all the individual rows:
    std::vector<torch::Tensor> rows; rows.resize(sampler_cfg.n_particles);

    for (int64_t i = 0; i < sampler_cfg.n_particles; i ++){
        rows[i] = spatial_nets[i](x).transpose(2,1);
    }

    // Stack all the rows together:
    auto spatial_slater = torch::cat(rows, 1);

    return spatial_slater;
}

torch::Tensor ManyBodyWavefunctionImpl::slater_matrix(torch::Tensor s_input, torch::Tensor s_matrix){

    // No grad needed in this part:
    // c10::InferenceMode guard(true);

    // Tile the spinor tensor (shape: n_walkers, n_particles)
    // We need to make it have a shape of (n_walkers, n_particles, n_particles)
    // Which means tiling the last dimension:

    auto repeated_spinor = torch::tile(s_input, {1, sampler_cfg.n_particles});
    repeated_spinor = repeated_spinor.reshape(
        {-1, sampler_cfg.n_particles, sampler_cfg.n_particles});
    auto s_slater = torch::pow(0.5*(repeated_spinor + s_matrix), 2);

    return s_slater;
}

