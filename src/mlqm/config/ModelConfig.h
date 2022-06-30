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




struct MLPConfig{
    int n_input;
    int n_output;
    int n_layers;
    int n_filters_per_layer;
    bool bias;
    bool residual;

    // Define default values:
    MLPConfig(){
        n_input = 1;
        n_output = 1;
        n_layers = 4;
        n_filters_per_layer = 32;
        bias = true;
        residual = false;
    }

};

void to_json(json& j, const MLPConfig& s);
void from_json(const json& j, MLPConfig& s);

struct DeepSetsCorrelaterConfig{
    MLPConfig individual_config;
    MLPConfig aggregate_config;
    float confinement;
    int latent_space;

    // Default constructor:
    DeepSetsCorrelaterConfig(){
        confinement = 0.1;
        latent_space = 32;
        individual_config.n_output = latent_space;
        aggregate_config.n_input = latent_space;
    }

};

void to_json(json& j, const DeepSetsCorrelaterConfig& s);
void from_json(const json& j, DeepSetsCorrelaterConfig& s);

struct ManyBodyWavefunctionConfig{
    DeepSetsCorrelaterConfig correlator_config;
    MLPConfig spatial_config;
    bool mean_subtract;

    ManyBodyWavefunctionConfig(){
        mean_subtract = true;
    }
};

void to_json(json& j, const ManyBodyWavefunctionConfig& s);
void from_json(const json& j, ManyBodyWavefunctionConfig& s);
