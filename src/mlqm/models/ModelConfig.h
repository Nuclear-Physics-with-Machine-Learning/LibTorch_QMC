/**
 * @defgroup   MODELCONFIG Model Configuration
 *
 * @brief      This file implements Model Configuration.
 *
 * @author     Corey.adams
 * @date       2022
 */

#pragma once

struct MLPConfig{
    int n_input;
    int n_output;
    int n_layers;
    int n_filters_per_layer;
    bool bias;
    bool residual;

    MLPConfig() :
        n_input(3),
        n_output(1),
        n_layers(4),
        n_filters_per_layer(32),
        bias(true),
        residual(false)
    {}
    MLPConfig(int in, int out) :
        n_input(in),
        n_output(out),
        n_layers(4),
        n_filters_per_layer(32),
        bias(true),
        residual(false)
    {}
};

struct DeepSetsCorrelaterConfig{
    MLPConfig individual_config;
    MLPConfig aggregate_config;
    float confinement;
    int latent_space;

    DeepSetsCorrelaterConfig() : 
        individual_config(3, 32),
        aggregate_config(32, 1),
        confinement(0.1),
        latent_space(32)
    {}

};
