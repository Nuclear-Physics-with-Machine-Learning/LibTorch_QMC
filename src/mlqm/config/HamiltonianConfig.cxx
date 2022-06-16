#include "HamiltonianConfig.h"


void to_json(json& j, const NuclearHamiltonianConfig& hc){
    j = json{
        {"M"   , hc.M},
        {"HBAR", hc.HBAR},
        {"OMEGA", hc.OMEGA},
    };
}    

void from_json(const json& j, NuclearHamiltonianConfig& hc ){

    j.at("M").get_to(hc.M);
    j.at("HBAR").get_to(hc.HBAR);
    j.at("OMEGA").get_to(hc.OMEGA);
}