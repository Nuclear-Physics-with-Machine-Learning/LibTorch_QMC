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
