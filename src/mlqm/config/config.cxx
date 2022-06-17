#include "config.h"



void to_json(json& j, const Config& cfg){
    j = json{
        {"sampler"     , cfg.sampler},
        {"wavefunction", cfg.wavefunction},
        {"hamiltonian", cfg.hamiltonian},
        {"delta", cfg.delta},
        {"epsilon", cfg.epsilon},
        {"n_iterations", cfg.n_iterations},
    };
}    

void from_json(const json& j, Config& cfg ){

    j.at("sampler").get_to(cfg.sampler);
    j.at("wavefunction").get_to(cfg.wavefunction);
    j.at("hamiltonian").get_to(cfg.hamiltonian);
    j.at("delta").get_to(cfg.delta);
    j.at("epsilon").get_to(cfg.epsilon);
    j.at("n_iterations").get_to(cfg.n_iterations);
}