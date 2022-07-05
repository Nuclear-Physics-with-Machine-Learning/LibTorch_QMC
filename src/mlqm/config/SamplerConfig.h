/**
 * @defgroup   SAMPLERCONIFG Sampler Conifg
 *
 * @brief      This file implements Sampler Conifg.
 *
 * @author     Corey.adams
 * @date       2022
 */

#pragma once

#include "nlohmann/json.hpp"
using json = nlohmann::json;

struct SamplerConfig{
    int n_walkers;
    int n_particles;
    int n_dim;
    int n_spin_up;
    int n_protons;
    bool use_spin;
    bool use_isospin;
    float kick_mean;
    float kick_std;
    int n_thermalize;
    int n_void_steps;
    int n_observable_measurements;
    int n_concurrent_obs_per_rank;

    SamplerConfig() :
        n_walkers(500),
        n_particles(1),
        n_dim(1),
        n_spin_up(1),
        n_protons(1),
        use_spin(true),
        use_isospin(true),
        kick_mean(0.0),
        kick_std(0.5),
        n_thermalize(1000),
        n_void_steps(200),
        n_observable_measurements(1),
        n_concurrent_obs_per_rank(1)
    {}
};

void to_json(json& j, const SamplerConfig& s);
void from_json(const json& j, SamplerConfig& s);
