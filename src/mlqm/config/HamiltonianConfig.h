/**
 * @defgroup   HAMILTONIANCONFIG Hamiltonian Configuration
 *
 * @brief      This file implements Hamiltonian Configuration.
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


struct NuclearHamiltonianConfig{
    float  M;
    float  HBAR;
    float  OMEGA;

    NuclearHamiltonianConfig(){
        M = 1.0;
        HBAR = 1.0;
        OMEGA = 1.0;
    }

};

void to_json(json& j, const NuclearHamiltonianConfig& s);
void from_json(const json& j, NuclearHamiltonianConfig& s);