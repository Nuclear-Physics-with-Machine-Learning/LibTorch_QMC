/**
 * @defgroup   MODELCONFIG Model Configuration
 *
 * @brief      This file implements Model Configuration.
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



struct MLPConfig{
    int n_input;
    int n_output;
    int n_layers;
    int n_filters_per_layer;
    bool bias;
    bool residual;

};

void to_json(json& j, const MLPConfig& s);
void from_json(const json& j, MLPConfig& s);

struct DeepSetsCorrelaterConfig{
    MLPConfig individual_config;
    MLPConfig aggregate_config;
    float confinement;
    int latent_space;

};

void to_json(json& j, const DeepSetsCorrelaterConfig& s);
void from_json(const json& j, DeepSetsCorrelaterConfig& s);

struct ManyBodyWavefunctionConfig{
    DeepSetsCorrelaterConfig correlator_config;
    MLPConfig spatial_config;
    bool mean_subtract;
};

void to_json(json& j, const ManyBodyWavefunctionConfig& s);
void from_json(const json& j, ManyBodyWavefunctionConfig& s);
