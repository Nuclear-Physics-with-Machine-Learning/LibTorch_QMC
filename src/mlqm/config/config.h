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


struct Config{
    SamplerConfig sampler;
    DeepSetsCorrelaterConfig wavefunction;
};

void to_json(json& j, const Config& s);
void from_json(const json& j, Config& s);
