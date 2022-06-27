#include "SamplerConfig.h"

void to_json(json& j, const SamplerConfig& s){
    j = json{
        {"n_walkers"   , s.n_walkers},
        {"n_particles" , s.n_particles},
        {"n_dim"       , s.n_dim},
        {"n_spin_up"   , s.n_spin_up},
        {"n_protons"   , s.n_protons},
        {"use_spin"    , s.use_spin},
        {"use_isospin" , s.use_isospin},
        {"kick_mean"   , s.kick_mean},
        {"kick_std"    , s.kick_std},
        {"n_thermalize"  , s.n_thermalize},
        {"n_void_steps"  , s.n_void_steps},
        {"n_observable_measurements" , s.n_observable_measurements},
        {"n_concurrent_obs_per_rank" , s.n_concurrent_obs_per_rank}
    };
}    

void from_json(const json& j, SamplerConfig& s ){

    if (j.contains("n_walkers"))
        j.at("n_walkers").get_to(s.n_walkers);
    if (j.contains("n_particles"))
        j.at("n_particles").get_to(s.n_particles);
    if (j.contains("n_dim"))
        j.at("n_dim").get_to(s.n_dim);
    if (j.contains("n_spin_up"))
        j.at("n_spin_up").get_to(s.n_spin_up);
    if (j.contains("n_protons"))
        j.at("n_protons").get_to(s.n_protons);
    if (j.contains("use_spin"))
        j.at("use_spin").get_to(s.use_spin);
    if (j.contains("use_isospin"))
        j.at("use_isospin").get_to(s.use_isospin);
    if (j.contains("kick_mean"))
        j.at("kick_mean").get_to(s.kick_mean);
    if (j.contains("kick_std"))
        j.at("kick_std").get_to(s.kick_std);
    if (j.contains("n_thermalize"))
        j.at("n_thermalize").get_to( s.n_thermalize);
    if (j.contains("n_void_steps"))
        j.at("n_void_steps").get_to( s.n_void_steps);
    if (j.contains("n_observable_measurements"))
        j.at("n_observable_measurements").get_to(s.n_observable_measurements);
    if (j.contains("n_concurrent_obs_per_rank"))
        j.at("n_concurrent_obs_per_rank").get_to(s.n_concurrent_obs_per_rank);
}