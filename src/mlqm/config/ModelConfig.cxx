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

    j.at("n_input").get_to(mlp.n_input);
    j.at("n_output").get_to(mlp.n_output);
    j.at("n_layers").get_to(mlp.n_layers);
    j.at("n_filters_per_layer").get_to(mlp.n_filters_per_layer);
    j.at("bias").get_to(mlp.bias);
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

    j.at("individual_config").get_to(s.individual_config);
    j.at("aggregate_config").get_to(s.aggregate_config);
    j.at("confinement").get_to(s.confinement);
    j.at("latent_space").get_to(s.latent_space);
}
