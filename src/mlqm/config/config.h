/**
 * @defgroup   CONFIG configuration
 *
 * @brief      This file implements configuration.
 *
 * @author     Corey.adams
 * @date       2022
 */

#pragma once

#include "SamplerConfig.h"
#include "ModelConfig.h"
#include "HamiltonianConfig.h"


struct Config{
    SamplerConfig sampler;
    ManyBodyWavefunctionConfig wavefunction;
    NuclearHamiltonianConfig hamiltonian;
    float delta;
    float epsilon;
    int n_iterations;
    std::string out_dir;
    size_t n_concurrent_jacobian;

    Config(){
        delta = 0.01;
        epsilon = 0.001;
        n_iterations = 50;
        n_concurrent_jacobian = 1;
        out_dir = "";
    }

    void validate_config(){
        // input to individual-particle correlator must == n_dim of each particle
        this -> wavefunction.correlator_config.individual_config.n_input = this -> sampler.n_dim;
        // Output of individual-particle must equal latent size
        this -> wavefunction.correlator_config.individual_config.n_output 
         = this -> wavefunction.correlator_config.latent_space;
        // input to individual-particle correlator must == latent_space
        this -> wavefunction.correlator_config.aggregate_config.n_input 
        = this -> wavefunction.correlator_config.latent_space;
        // Input to spatial configs in the Slater matrix must == n_dim
        this -> wavefunction.spatial_config.n_input = this -> sampler.n_dim;
        // Output to same config must equal 1
        this -> wavefunction.spatial_config.n_output = 1;
        return;
    }
};

void to_json(json& j, const Config& s);
void from_json(const json& j, Config& s);
