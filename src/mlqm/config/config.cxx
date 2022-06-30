#include "config.h"



void to_json(json& j, const Config& cfg){
    j = json{
        {"sampler"     , cfg.sampler},
        {"wavefunction", cfg.wavefunction},
        {"hamiltonian",  cfg.hamiltonian},
        {"optimizer",    cfg.optimizer},
        {"out_dir",      cfg.out_dir},
        {"n_concurrent_jacobian", cfg.n_concurrent_jacobian},
    };
}

void from_json(const json& j, Config& cfg ){

    if (j.contains("sampler")) j.at("sampler").get_to(cfg.sampler);
    if (j.contains("wavefunction")) j.at("wavefunction").get_to(cfg.wavefunction);
    if (j.contains("hamiltonian")) j.at("hamiltonian").get_to(cfg.hamiltonian);
    if (j.contains("optimizer")) j.at("optimizer").get_to(cfg.optimizer);
    if (j.contains("out_dir")) j.at("out_dir").get_to(cfg.out_dir);
    if (j.contains("n_concurrent_jacobian"))
        j.at("n_concurrent_jacobian").get_to(cfg.n_concurrent_jacobian);
}
