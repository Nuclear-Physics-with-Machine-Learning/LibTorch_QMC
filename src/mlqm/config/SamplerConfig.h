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

    // SamplerConfig() :  
    //     n_walkers(10000),
    //     n_particles(10),
    //     n_dim(3),
    //     n_spin_up(0),
    //     n_protons(0),
    //     use_spin(false),
    //     use_isospin(false),
    //     kick_mean(0.0),
    //     kick_std(0.2)
    // {}
};

void to_json(json& j, const SamplerConfig& s);
void from_json(const json& j, SamplerConfig& s);