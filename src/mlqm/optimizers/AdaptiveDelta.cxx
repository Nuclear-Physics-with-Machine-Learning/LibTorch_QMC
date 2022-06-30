#include "AdaptiveDelta.h"

AdaptiveDelta::AdaptiveDelta(Config cfg, torch::TensorOptions opts)
    : BaseOptimizer(cfg, opts)
{

}

std::map<std::string, torch::Tensor> AdaptiveDelta::compute_updates_and_metrics(
    std::vector<torch::Tensor> current_psi,
    std::vector<torch::Tensor> & gradients,
    torch::Tensor & next_energy)
{

    auto opt_metrics = delta_optimizer(
            current_psi, gradients, next_energy);

    return opt_metrics;
}

std::map<std::string, torch::Tensor> AdaptiveDelta::delta_optimizer(
    std::vector<torch::Tensor> current_psi,
    std::vector<torch::Tensor> & gradients,
    torch::Tensor & next_energy)
{

    auto delta_min = torch::full({}, config.optimizer.delta[0], options);
    auto delta_max = torch::full({}, config.optimizer.delta[1], options);
    auto epsilon   = torch::full({}, config.optimizer.epsilon[0], options);

    torch::Tensor S_ij;
    auto dp_i =  compute_gradients(
        estimator["dpsi_i"],
        estimator["energy"],
        estimator["dpsi_i_EL"],
        estimator["dpsi_ij"],
        epsilon,
        S_ij);

    // Delta and epsilon are by default arrays (length 2) specifying trials
    // Take the first, by default.

    size_t n_iterations(config.optimizer.n_opt_trials);

    // These are metrics to track how big the update is.
    std::vector<torch::Tensor> energies;    energies.resize(n_iterations);
    std::vector<torch::Tensor> overlaps;    overlaps.resize(n_iterations);
    std::vector<torch::Tensor> acoses;      acoses.resize(n_iterations);
    std::vector<torch::Tensor> par_dists;   par_dists.resize(n_iterations);
    std::vector<torch::Tensor> ratios;      ratios.resize(n_iterations);


    auto delta_options = torch::tensor({
        0.0001, 0.0002, 0.0005,
        0.001, 0.002, 0.005,
        // 0.01, 0.02, 0.05,
        0.1, 0.5, 1.0, 2.0
    });


    // Now, loop and recompute each time:
    for (size_t i_iteration = 0; i_iteration < n_iterations; i_iteration ++){
        auto this_delta = delta_options[i_iteration];
        auto this_gradients = this_delta * dp_i;
        // Unpack the gradients
        gradients = unflatten_weights_or_gradients(this_gradients);
        auto adaptive_params = adaptive_wavefunction->parameters();

        // Set the adaptive wavefunction's gradients to these gradients:
        auto original_weights = wavefunction->parameters();
        {
            c10::InferenceMode guard(true);

            // Even though it's a flat optimization, we recompute the energy to get the overlap too:

            for (size_t i_layer = 0; i_layer < original_weights.size(); i_layer ++){

                // Bit of a crappy way to do this, may need to optimize later
                adaptive_wavefunction -> parameters()[i_layer] +=
                    original_weights[i_layer] - adaptive_params[i_layer] + gradients[i_layer];
                // PLOG_INFO << "  set to :" << adaptive_wavefunction->parameters()[i_layer];
            }
        }
        // Compute the new energy:
        energies[i_iteration] = recompute_energy(
            adaptive_wavefunction, current_psi,
            overlaps[i_iteration], acoses[i_iteration]);

        PLOG_INFO << "Energy " << i_iteration << ": " << energies[i_iteration];

        // Compute the parameter distance:
        par_dists[i_iteration] = grad_calc.par_dist(this_gradients, S_ij);
        ratios[i_iteration]    = torch::abs(
            par_dists[i_iteration] - acoses[i_iteration]) /
            torch::abs(par_dists[i_iteration] + acoses[i_iteration]+ 1e-8);


    }


    //
    // // Apply cuts to all metrics:
    // auto par_dist_cut   = par_dists < 0.1;
    // auto acoses_cut     = acoses < 0.1;
    // auto overlaps_cut   = overlaps > 0.9;
    // auto ratio_cut      = ratio < 0.4;
    //
    // // Union all cuts
    // auto final_cuts = torch::logical_and(par_dist_cut, acoses_cut);
    // final_cuts = torch::logical_and(final_cuts, overlaps_cut);
    // final_cuts = torch::logical_and(final_cuts, ratio_cut);

    // PLOG_INFO << final_cuts

    bool success = false;
    size_t best_index = 0;
    torch::Tensor best_energy = torch::full({}, 99999., options);;

    // Now, we loop over all found values and select the best:
    for (size_t i_iteration = 0; i_iteration < n_iterations; i_iteration ++){
        // Check if this energy passes
        if ((par_dists[i_iteration] < 0.1).to(torch::kBool).data_ptr<bool>()[0] &&
            (acoses[i_iteration] < 0.1).to(torch::kBool).data_ptr<bool>()[0] &&
            (overlaps[i_iteration] > 0.9).to(torch::kBool).data_ptr<bool>()[0] &&
            (ratios[i_iteration] < 0.4).to(torch::kBool).data_ptr<bool>()[0] )
        {
            if ((energies[i_iteration] < best_energy).to(torch::kBool).data_ptr<bool>()[0]){
                best_energy = energies[i_iteration];
                best_index  = i_iteration;
                success = true;
            }
        }
    }

    PLOG_INFO << "Success: " << success;
    PLOG_INFO << "Best energy: " << best_energy;
    PLOG_INFO << "Best index: " << best_index;

    std::map<std::string, torch::Tensor> opt_metrics;

    if (success){
        opt_metrics = {
           {"optimizer/delta"   , delta_options[best_index]},
           {"optimizer/eps"     , epsilon},
           {"optimizer/overlap" , overlaps[best_index]},
           {"optimizer/par_dist", par_dists[best_index]},
           {"optimizer/acos"    , acoses[best_index]},
           {"optimizer/ratio"   , ratios[best_index]},
       };
       gradients = unflatten_weights_or_gradients(delta_options[best_index] * dp_i);
       next_energy = best_energy;
    }
    else{
        unflatten_weights_or_gradients(torch::zeros_like(dp_i, options));
        next_energy = torch::full({}, 0.0, options);
        opt_metrics = {
           {"optimizer/delta"   , torch::full({}, 0.0, options)},
           {"optimizer/eps"     , torch::full({}, 1.0, options)},
           {"optimizer/overlap" , torch::full({}, 0.0, options)},
           {"optimizer/par_dist", torch::full({}, 0.0, options)},
           {"optimizer/acos"    , torch::full({}, 0.0, options)},
           {"optimizer/ratio"   , torch::full({}, 0.0, options)},
       };
    }

    // return opt_metrics
    return opt_metrics;

}
