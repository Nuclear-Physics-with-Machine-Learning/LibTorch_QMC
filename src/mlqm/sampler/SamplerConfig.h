/**
 * @defgroup   SAMPLERCONIFG Sampler Conifg
 *
 * @brief      This file implements Sampler Conifg.
 *
 * @author     Corey.adams
 * @date       2022
 */

#pragma once

struct SamplerConfig{
    int n_walkers;
    int n_particles;
    int n_dim;
    int n_spin_up;
    int n_protons;
    bool use_spin;
    bool use_isospin;

    SamplerConfig() :  
        n_walkers(500),
        n_particles(2),
        n_dim(3),
        n_spin_up(0),
        n_protons(0),
        use_spin(false),
        use_isospin(false)
    {}
};
