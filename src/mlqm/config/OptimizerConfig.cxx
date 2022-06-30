#include "OptimizerConfig.h"



void to_json(json& j, const OptimizerConfig& opt){
    j = json{
        {"delta"        , opt.delta},
        {"epsilon"      , opt.epsilon},
        {"n_iterations" , opt.n_iterations},
        {"mode"         , opt.mode},
    };
}

void from_json(const json& j, OptimizerConfig& opt ){

    // Overrides from json are only enforced if the key is present!

    if (j.contains("delta"))
        j.at("delta").get_to(opt.delta);
    if (j.contains("epsilon"))
        j.at("epsilon").get_to(opt.epsilon);
    if (j.contains("n_iterations"))
        j.at("n_iterations").get_to(opt.n_iterations);
    if (j.contains("mode"))
        j.at("mode").get_to(opt.mode);
}
