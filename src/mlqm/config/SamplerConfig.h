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

// We include the logger here because config.h gets included everywhere
#define PLOG_OMIT_LOG_DEFINES
#include <plog/Log.h> // Step1: include the headers
#include "plog/Initializers/RollingFileInitializer.h"
#include "plog/Appenders/ConsoleAppender.h"


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