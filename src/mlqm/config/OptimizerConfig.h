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


struct OptimizerConfig{


    std::array<float, 2> delta;
    std::array<float, 2> epsilon;
    size_t n_iterations;
    size_t n_opt_trials;
    enum {FLAT, ADAPTIVE_DELTA, ADAPTIVE_EPSILON} mode;


    // Define default values:
    OptimizerConfig(){
        delta = {0.01, 0.01};
        epsilon = {0.001, 0.001};
        n_iterations = 50;
        n_opt_trials = 10;
        mode = FLAT;
    }

};

void to_json(json& j, const OptimizerConfig& s);
void from_json(const json& j, OptimizerConfig& s);
