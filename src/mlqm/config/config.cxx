#include "config.h"



void to_json(json& j, const Config& mlp){
    j = json{
        {"sampler"     , mlp.sampler},
        {"wavefunction", mlp.wavefunction},
    };
}    

void from_json(const json& j, Config& mlp ){

    j.at("sampler").get_to(mlp.sampler);
    j.at("wavefunction").get_to(mlp.wavefunction);
}