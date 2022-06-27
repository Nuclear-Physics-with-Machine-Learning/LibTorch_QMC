#include "ModelConfig.h"



void to_json(json& j, const MLPConfig& mlp){
    j = json{
        {"n_input"             , mlp.n_input},
        {"n_output"            , mlp.n_output},
        {"n_layers"            , mlp.n_layers},
        {"n_filters_per_layer" , mlp.n_filters_per_layer},
        {"bias"                , mlp.bias},
        {"residual"            , mlp.residual}
    };
}

void from_json(const json& j, MLPConfig& mlp ){

    // Overrides from json are only enforced if the key is present!

    if (j.contains("n_input"))
        j.at("n_input").get_to(mlp.n_input);
    if (j.contains("n_output"))
        j.at("n_output").get_to(mlp.n_output);
    if (j.contains("n_layers"))
        j.at("n_layers").get_to(mlp.n_layers);
    if (j.contains("n_filters_per_layer"))
        j.at("n_filters_per_layer").get_to(mlp.n_filters_per_layer);
    if (j.contains("bias"))
        j.at("bias").get_to(mlp.bias);
    if (j.contains("residual"))
        j.at("residual").get_to(mlp.residual);
}

void to_json(json& j, const DeepSetsCorrelaterConfig& d){
    j = json{
        {"individual_config" , d.individual_config},
        {"aggregate_config"  , d.aggregate_config},
        {"confinement"       , d.confinement},
        {"latent_space"      , d.latent_space}
    };
}

void from_json(const json& j, DeepSetsCorrelaterConfig& s ){

    if (j.contains("individual_config"))
        j.at("individual_config").get_to(s.individual_config);
    if (j.contains("aggregate_config"))
        j.at("aggregate_config").get_to(s.aggregate_config);
    if (j.contains("confinement"))
        j.at("confinement").get_to(s.confinement);
    if (j.contains("latent_space"))
        j.at("latent_space").get_to(s.latent_space);
}


void to_json(json& j, const ManyBodyWavefunctionConfig& m){
    j = json{
        {"correlator_config" , m.correlator_config},
        {"spatial_config",     m.spatial_config},
        {"mean_subtract",      m.mean_subtract},
    };
}
void from_json(const json& j, ManyBodyWavefunctionConfig& m){
    if (j.contains("correlator_config"))
        j.at("correlator_config").get_to(m.correlator_config);
    if (j.contains("spatial_config"))
        j.at("spatial_config").get_to(m.spatial_config);
    if (j.contains("mean_subtract"))
        j.at("mean_subtract").get_to(m.mean_subtract);
}
