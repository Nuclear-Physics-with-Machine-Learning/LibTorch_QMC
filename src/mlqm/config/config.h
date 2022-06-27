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
};

void to_json(json& j, const Config& s);
void from_json(const json& j, Config& s);
