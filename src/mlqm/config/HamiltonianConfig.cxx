#include "HamiltonianConfig.h"


void to_json(json& j, const NuclearHamiltonianConfig& hc){
    j = json{
        {"M"   , hc.M},
        {"HBAR", hc.HBAR},
        {"OMEGA", hc.OMEGA},
    };
}    

void from_json(const json& j, NuclearHamiltonianConfig& hc ){

    if (j.contains("M")) j.at("M").get_to(hc.M);
    if (j.contains("HBAR")) j.at("HBAR").get_to(hc.HBAR);
    if (j.contains("OMEGA")) j.at("OMEGA").get_to(hc.OMEGA);
}