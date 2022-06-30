#include "BaseOptimizer.h"

#include <chrono>
using namespace std::chrono;

#ifdef OPENMP_FOUND
#include <omp.h>
#endif

BaseOptimizer::BaseOptimizer(Config cfg, torch::TensorOptions opts)
    : config(cfg),
      options(opts),
      sampler(cfg.sampler, opts),
      hamiltonian(cfg.hamiltonian, opts),
      grad_calc(opts),
      jac_calc(opts)
{

    size = 1;

    // Turn off inference mode at the highest level
    c10::InferenceMode guard(false);

//
// #ifdef OPENMP_FOUND
//     omp_set_num_threads(32);
//     PLOG_INFO << "Using omp threads: " << omp_get_num_threads();
// #endif
    /*
    We need to know how many times to loop over the walkers and metropolis step.
    The total number of observations is set: config.sampler.n_observable_measurements
    There is an optimization to walk in parallel with n_concurrent_obs_per_rank
    Without MPI, the number of loops is then n_observable_measurements / n_concurrent_obs_per_rank
    WITH MPI, we have to reduce the number of loops by the total number of ranks.

    */

    n_loops_total = int(config.sampler.n_observable_measurements /
            config.sampler.n_concurrent_obs_per_rank);

    if (MPI_AVAILABLE){
        n_loops_total = int(n_loops_total / size);
    }

    predicted_energy = torch::full({}, 0.0, options);



    // We do a check that n_loops_total * n_concurrent_obs_per_rank matches expectations:
    if (n_loops_total * config.sampler.n_concurrent_obs_per_rank*size
            != config.sampler.n_observable_measurements){
        std::stringstream exception_str;
        exception_str << "Total number of observations to compute is unexpected!\n";
        exception_str << "  Expected to have " << config.sampler.n_observable_measurements << ", have:\n";
        exception_str << "  -- A loop of " << config.sampler.n_concurrent_obs_per_rank << " observations";
        exception_str << " for " << n_loops_total << " loops over " << this -> size << " ranks";
        exception_str << "  -- (" << config.sampler.n_concurrent_obs_per_rank << ")*(" << n_loops_total << "";
        exception_str << ")*(" << this -> size << ") != " << config.sampler.n_observable_measurements << "\n";
        PLOG_ERROR << exception_str.str();
        throw std::exception();
    }

    // Create a model:
    wavefunction = ManyBodyWavefunction(cfg.wavefunction, options);
    wavefunction -> to(options.device());
    wavefunction -> to(torch::kFloat64);

    adaptive_wavefunction = ManyBodyWavefunction(cfg.wavefunction, options);
    adaptive_wavefunction -> to(options.device());
    adaptive_wavefunction -> to(torch::kFloat64);



    auto params = wavefunction -> named_parameters();

    n_parameters = 0;
    size_t n_layers = wavefunction->parameters().size();
    for (auto & key_pair : params){
        int64_t local_params = 1;
        for (auto & s : key_pair.value().sizes()){
            local_params *= s;
        }
        // Store the shapes:
        flat_shapes.push_back(local_params);
        full_shapes.push_back(key_pair.value().sizes());
        // Add this layer to the total:
        n_parameters += local_params;

        // Set the adaptive wavefunction parameters to the original one:
        // adaptive_wavefunction->named_parameters()[key_pair.key()].set_(key_pair.value());
    }

    PLOG_INFO << "Total parameters: " << n_parameters;
    PLOG_INFO << "Total layers: " << n_layers;

    // We can't set concurrency if the concurrency is too high:
    if (config.n_concurrent_jacobian > n_parameters){
        PLOG_WARNING << "Jacobian Concurrency is larger than the number of parameters";
        PLOG_WARNING << "This may cause segmentation faults or incorrect calculations";
    }
    // Initialize random input:
    auto input = sampler.sample_x();

    if (config.n_concurrent_jacobian > input.sizes()[0]){
        PLOG_WARNING << "Jacobian Concurrency is larger than the number of concurrent walkers";
        PLOG_WARNING << "This may cause segmentation faults or incorrect calculations";
    }
    // Set up the concurrency in the jacobian calculator:
    jac_calc.set_parallelization(wavefunction, config.n_concurrent_jacobian);





    // Thermalize the walkers for some large number of initial thermalizations:
    PLOG_INFO << "Begin initial equilibration with " << input.sizes()[0] << " walkers";
    // float acceptance = equilibrate(cfg.sampler.n_thermalize).data_ptr<double>()[0];
    auto acceptance = equilibrate(cfg.sampler.n_thermalize);
    PLOG_INFO << "Done initial equilibration, acceptance = " << acceptance.cpu().data_ptr<float>()[0];


}

BaseOptimizer::~BaseOptimizer(){}

torch::Tensor BaseOptimizer::equilibrate(int64_t n_iterations){
    torch::Tensor acceptance = sampler.kick(n_iterations, wavefunction);
    return acceptance;
}


std::vector<torch::Tensor> BaseOptimizer::compute_O_observables(
    torch::Tensor flattened_jacobian,
    torch::Tensor energy,
    torch::Tensor w_of_x)
{
    std::vector<torch::Tensor> return_tensors;

    // dspi_i is the reduction of the jacobian over all walkers.
    // In other words, it's the mean gradient of the parameters with respect to inputs.
    // This is effectively the measurement of O^i in the paper.
    auto normed_fj  = flattened_jacobian;

    auto dpsi_i = torch::mean(normed_fj, 0);
    // dpsi_i = torch::reshape(dpsi_i, {-1,1})
    // To compute <O^m O^n>
    auto normed_fj_t = torch::transpose(normed_fj, 1, 0);


    ///TODO : divide by n_walkers_per_obs - DONE?
    auto dpsi_ij = torch::matmul(normed_fj_t, normed_fj) / config.sampler.n_walkers;


    ///TODO: Compute <O^m H>
    // # Computing <O^m H>:
    auto e_reshaped = torch::reshape(energy, {1,config.sampler.n_walkers});

    // PLOG_INFO << "e_reshaped.sizes(): " << e_reshaped.sizes();
    // PLOG_INFO << "normed_fj.sizes(): " << normed_fj.sizes();
    auto dpsi_i_EL = torch::matmul(e_reshaped, normed_fj).flatten();
    // # This makes this the same shape as the other tensors
    // dpsi_i_EL = tf.reshape(dpsi_i_EL, [-1, 1])


    return_tensors.push_back(dpsi_i);
    return_tensors.push_back(dpsi_ij);
    return_tensors.push_back(dpsi_i_EL);

    return return_tensors;
}



std::vector<torch::Tensor> BaseOptimizer::walk_and_accumulate_observables(){
    /*
    Internal function to take a wavefunction and set of walkers and compute observables
    */
    estimator.clear();

    std::vector<torch::Tensor> current_psi;

    // Loop over observations:
    for (int i_loop = 0; i_loop < n_loops_total; ++i_loop)
    {
        // First do a void walk to thermalize after a new configuration.
        // By default, this will use the previous walkers as a starting configurations.
        // #   This one does all the kicks in a compiled function.
        auto acceptance = equilibrate(config.sampler.n_void_steps);


        // Get the current walker locations:
        auto x_current = sampler.sample_x();
        auto spin      = sampler.sample_spin();
        auto isospin   = sampler.sample_isospin();

        // Compute the observables:
        // Here, perhaps we can compute the d_i of obs_energy:
        torch::Tensor energy_jf, ke_jf, ke_direct, pe, w_of_x;
        auto energy = \
            hamiltonian.energy(
                wavefunction, x_current, // spin, isospin,  // spin/isospin not yet
                energy_jf, // return by reference
                ke_jf,
                ke_direct,
                pe,
                w_of_x
            );

        // R is computed but it needs to be WRT the center of mass of all particles
        // So, mean subtract if needed:
        if (config.wavefunction.mean_subtract){
            // Average over particles
            auto mean = torch::mean(x_current, {1});
            x_current = x_current - mean.reshape({x_current.sizes()[0], 1, x_current.sizes()[2]});
        }

        auto r = torch::sum(torch::pow(x_current,2.), {1,2});
        r = torch::mean(torch::sqrt(r));
        // Here, we split the energy and other objects into
        // sizes of nwalkers_per_observation
        // if config.sampler.n_concurrent_obs_per_rank != 1:
        // Split walkers:
        auto x_current_v  = torch::chunk(x_current, config.sampler.n_concurrent_obs_per_rank, 0);
        if (config.sampler.use_spin) {
            auto spin_v       = torch::chunk(spin,      config.sampler.n_concurrent_obs_per_rank, 0);
        }
        if (config.sampler.use_isospin) {
            auto isospin_v    = torch::chunk(isospin,    config.sampler.n_concurrent_obs_per_rank, 0);
        }



        // Split observables:
        auto energy_v     = torch::chunk(energy,    config.sampler.n_concurrent_obs_per_rank, 0);
        auto energy_jf_v  = torch::chunk(energy_jf, config.sampler.n_concurrent_obs_per_rank, 0);
        auto ke_jf_v      = torch::chunk(ke_jf,     config.sampler.n_concurrent_obs_per_rank, 0);
        auto ke_direct_v  = torch::chunk(ke_direct, config.sampler.n_concurrent_obs_per_rank, 0);
        auto pe_v         = torch::chunk(pe,        config.sampler.n_concurrent_obs_per_rank, 0);
        auto w_of_x_v     = torch::chunk(w_of_x,    config.sampler.n_concurrent_obs_per_rank, 0);

        // for (auto w : w_of_x_v) current_psi.push_back(w);
        current_psi.push_back(w_of_x);



        // For each observation, we compute the jacobian.
        auto jac = jac_calc.batch_jacobian_reverse(x_current, wavefunction);

        // Jac shape is [n_walkers, n_parameters] and we need to normalize by the wavefunction values
        jac = jac / torch::reshape(w_of_x, {-1, 1});



        // split the jacobian into chunks as well, per configuration:
        auto split_jacobian = torch::chunk(jac, config.sampler.n_concurrent_obs_per_rank, 0);

        // Now, compute observables, store them in an estimator:

        for (int64_t i_obs = 0; i_obs < config.sampler.n_concurrent_obs_per_rank; i_obs ++){
            auto obs_energy      = energy_v[i_obs]     / config.sampler.n_walkers;
            auto obs_energy_jf   = energy_jf_v[i_obs]  / config.sampler.n_walkers;
            auto obs_ke_jf       = ke_jf_v[i_obs]      / config.sampler.n_walkers;
            auto obs_ke_direct   = ke_direct_v[i_obs]  / config.sampler.n_walkers;
            auto obs_pe          = pe_v[i_obs]         / config.sampler.n_walkers;

            auto observables = compute_O_observables(
                split_jacobian[i_obs], obs_energy, w_of_x_v[i_obs]);

            // this is a poor version of tuple unpacking:
            torch::Tensor dpsi_i = observables[0];
            torch::Tensor dpsi_ij = observables[1];
            torch::Tensor dpsi_i_EL = observables[2];
            // Accumulate variables:


            estimator.accumulate("energy",     torch::sum(obs_energy));
            estimator.accumulate("energy2",    torch::sum(torch::pow(obs_energy, 2)));
            estimator.accumulate("energy_jf",  torch::sum(obs_energy_jf));
            estimator.accumulate("energy2_jf", torch::sum(torch::pow(obs_energy_jf, 2)));
            estimator.accumulate("ke_jf",      torch::sum(obs_ke_jf));
            estimator.accumulate("ke_direct",  torch::sum(obs_ke_direct));
            estimator.accumulate("pe",         torch::sum(obs_pe));
            estimator.accumulate("acceptance", torch::mean(acceptance));
            estimator.accumulate("r",          torch::mean(r));
            estimator.accumulate("dpsi_i",     dpsi_i);
            estimator.accumulate("dpsi_i_EL",  dpsi_i_EL);
            estimator.accumulate("dpsi_ij",    dpsi_ij);
            estimator.accumulate("weight",     torch::ones(1));

        }
    }

    // INTERCEPT HERE with MPI to allreduce the estimator objects.
    if (MPI_AVAILABLE) {
        estimator.allreduce();
    }

    return current_psi;

}








std::map<std::string, torch::Tensor> BaseOptimizer::sr_step(){
    std::map<std::string, torch::Tensor> metrics = {};


    // We do a thermalization step again:
    equilibrate(config.sampler.n_thermalize);

    // Clear the walker history:
    sampler.reset_history();

    // Now, actually apply the loop and compute the observables:
    std::vector<torch::Tensor> current_psi = walk_and_accumulate_observables();

    // At this point, we need to average the observables that feed into the optimizer:
    estimator.finalize(torch::ones({}, options)*config.sampler.n_observable_measurements);


    auto error = torch::sqrt(
            torch::pow(estimator["energy2"] - estimator["energy"], 2) \
            / (config.sampler.n_observable_measurements-1));
    auto error_jf = torch::sqrt(
            torch::pow(estimator["energy2_jf"] - estimator["energy_jf"], 2) \
            / (config.sampler.n_observable_measurements-1));



    metrics["energy/energy"]     = estimator["energy"];
    metrics["energy/error"]      = error;
    metrics["energy/energy_jf"]  = estimator["energy_jf"];
    metrics["energy/error_jf"]   = error_jf;
    metrics["metropolis/acceptance"] = estimator["acceptance"];
    metrics["metropolis/r"]      = estimator["r"];
    metrics["energy/ke_jf"]      = estimator["ke_jf"];
    metrics["energy/ke_direct"]  = estimator["ke_direct"];
    metrics["energy/pe"]         = estimator["pe"];



    // Here, we call the function to optimize eps/delta and compute the gradients:

    torch::Tensor next_energy;
    std::vector<torch::Tensor> delta_p;
    auto opt_metrics = this -> compute_updates_and_metrics(current_psi, delta_p, next_energy);

    metrics.insert(opt_metrics.begin(), opt_metrics.end());
    // Compute the ratio of the previous energy and the current energy
    auto energy_diff = predicted_energy - estimator["energy"];
    metrics["energy/energy_diff"] = energy_diff;

    // And apply them to the wave function:
    apply_gradients(delta_p);

    // Before moving on, set the predicted_energy:
    predicted_energy = next_energy;

    return metrics;

}
void BaseOptimizer::apply_gradients(
    const std::vector<torch::Tensor> & gradients){

    c10::InferenceMode guard(true);

    // Even though it's a flat optimization, we recompute the energy to get the overlap too:
    for (size_t i_layer = 0; i_layer < full_shapes.size(); i_layer ++){

        // Bit of a crappy way to do this, may need to optimize later
        wavefunction -> parameters()[i_layer] += gradients[i_layer];
        // PLOG_INFO << "  set to :" << adaptive_wavefunction->parameters()[i_layer];
    }

}


torch::Tensor BaseOptimizer::compute_gradients(
    torch::Tensor dpsi_i,
    torch::Tensor energy,
    torch::Tensor dpsi_i_EL,
    torch::Tensor dpsi_ij,
    torch::Tensor eps,
    torch::Tensor & S_ij){


    // Get the natural gradients and S_ij
    auto f_i = grad_calc.f_i(dpsi_i, energy, dpsi_i_EL);

    S_ij = grad_calc.S_ij(dpsi_ij, dpsi_i);

    // Regularize S_ij with as small and eps as possible:

    auto dp_i = grad_calc.pd_solve(S_ij, eps, f_i);

    // for (int i_trial = 0; i_trial < 5; ++i_trial)
    // {

    // }
    //     try:

    //         break
    //     except tf.errors.InvalidArgumentError:
    //         logger.debug("Cholesky solve failed, continuing with higher regularization")
    //         eps *= 2.
    //     continue

    return dp_i;
}


torch::Tensor BaseOptimizer::recompute_energy(
    ManyBodyWavefunction trial_wf, std::vector<torch::Tensor> current_psi,
    torch::Tensor & overlap, torch::Tensor & acos){

    // Reset the aux estimator:
    re_estimator.clear();

    // For all the current_psi positions, recompute the energy in the new locations:
    auto x_history = sampler.x_history();
    auto spin_history = sampler.spin_history();
    auto isospin_history = sampler.isospin_history();

    int64_t start = 0;
    int64_t end = n_loops_total;

    // TODO - this can likely be optimized going through all obs in big vectors

    for (int64_t i_loop = start; i_loop < end; i_loop++){

        auto this_x = x_history[i_loop];
        auto this_spin = spin_history[i_loop];
        auto this_isospin = isospin_history[i_loop];
        auto this_psi = current_psi[i_loop];

        torch::Tensor energy_jf, ke_jf, ke_direct, pe, w_of_x;
        auto energy = \
            hamiltonian.energy(
                trial_wf, this_x, // this_spin, this_isospin,  // spin/isospin not yet
                energy_jf, // return by reference
                ke_jf,
                ke_direct,
                pe,
                w_of_x
            );

        // we need to compute the ratio of wavefunctions for each walker configuration.
        // We can do that before chunking and then split the ratio:
        auto wavefunction_ratio = w_of_x / this_psi;

        auto probability_ratio = torch::pow(wavefunction_ratio, 2.);

        // this should be a dot product.
        auto normalized_energy = energy * probability_ratio;

        // If necessary, chunk the observables cared about:

        auto energy_v = torch::chunk(normalized_energy,  config.sampler.n_concurrent_obs_per_rank, 0);
        auto w_ratio  = torch::chunk(wavefunction_ratio, config.sampler.n_concurrent_obs_per_rank, 0);
        auto p_ratio  = torch::chunk(probability_ratio,  config.sampler.n_concurrent_obs_per_rank, 0);


        // Accumulate all the pieces:
        for (int64_t i_obs = 0; i_obs < config.sampler.n_concurrent_obs_per_rank; i_obs ++){

            re_estimator.accumulate("energy", torch::sum(energy_v[i_obs]));
            re_estimator.accumulate("weight", torch::sum(p_ratio[i_obs]));
            re_estimator.accumulate("wavefunction_ratio", torch::sum(w_ratio[i_obs]));
            re_estimator.accumulate("N", torch::ones({},options)*config.sampler.n_walkers);

        }
    }



    if (MPI_AVAILABLE){
        re_estimator.allreduce();
    }


    // # What's the total weight?  Use that for the finalization:
    auto total_weight = re_estimator["weight"];

    //  Get the overlap
    auto wavefunction_ratio = re_estimator["wavefunction_ratio"];
    auto probability_ratio = re_estimator["weight"];

    auto N = re_estimator["N"];


    auto overlap2 = torch::pow(wavefunction_ratio / N, 2) / (probability_ratio / N);


    re_estimator.finalize(total_weight);

    // Set the return-by-ref values
    overlap  = torch::sqrt(overlap2);
    acos     = torch::pow(torch::acos(overlap),2);


    return re_estimator["energy"];


}

std::map<std::string, torch::Tensor> BaseOptimizer::compute_updates_and_metrics(
    std::vector<torch::Tensor> current_psi,
    std::vector<torch::Tensor> & gradients,
    torch::Tensor & next_energy){

    PLOG_INFO << "BASE update";
    // This particular instance is just a pass through, but intended
    // to be an overriden entry point to a better optimization algorithm
    auto opt_metrics = flat_optimizer(
        current_psi, gradients, next_energy);

    return opt_metrics;

}

std::vector<torch::Tensor> BaseOptimizer::unflatten_weights_or_gradients(
    torch::Tensor flat_gradients){

    // We take in a tensor of flat gradients; we return the gradients chopped up
    // and reshaped to proper values.

    std::vector<torch::Tensor> result;


    int64_t start = 0;
    int64_t end = flat_shapes[0];

    for (size_t i_layer = 0; i_layer < full_shapes.size(); i_layer ++){


        // First get the slice in a flat form:
        auto flat_layer = flat_gradients.index({Slice(start, end)});

        // update the bounds of the indexing for next time:
        if (i_layer != full_shapes.size() - 1){
             // We move the end to the start:
            start = end;
            // and add to the end the next value:
            end += flat_shapes[i_layer+1];
        }

        // reshape and append:
        result.push_back(flat_layer.reshape(full_shapes[i_layer]));

    }

    return result;
}

std::map<std::string, torch::Tensor> BaseOptimizer::flat_optimizer(
    std::vector<torch::Tensor> current_psi,
    std::vector<torch::Tensor> & gradients,
    torch::Tensor & next_energy){

    torch::Tensor S_ij;
    auto dp_i =  compute_gradients(
        estimator["dpsi_i"],
        estimator["energy"],
        estimator["dpsi_i_EL"],
        estimator["dpsi_ij"],
        torch::full({}, config.optimizer.epsilon[0], options),
        S_ij);

    // Delta and epsilon are by default arrays (length 2) specifying trials
    // Take the first, by default.

    auto delta_p = dp_i * torch::full({}, config.optimizer.delta[0], options);

    // Unpack the gradients
    // TODO
    gradients = unflatten_weights_or_gradients(delta_p);
    auto adaptive_params = adaptive_wavefunction->parameters();


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
    torch::Tensor acos, overlap;
    next_energy = recompute_energy(
        adaptive_wavefunction, current_psi,
        overlap, acos);

    // Compute the parameter distance:
    torch::Tensor par_dist = grad_calc.par_dist(config.optimizer.delta[0] * dp_i, S_ij);
    torch::Tensor ratio    = torch::abs(par_dist - acos) / torch::abs(par_dist + acos+ 1e-8);


    std::map<std::string, torch::Tensor> opt_metrics = {
        {"optimizer/delta"   , torch::full({}, config.optimizer.delta[0], options)},
        {"optimizer/eps"     , torch::full({}, config.optimizer.epsilon[0], options)},
        {"optimizer/overlap" , overlap},
        {"optimizer/par_dist", par_dist},
        {"optimizer/acos"    , acos},
        {"optimizer/ratio"   , ratio},
    };

    // return opt_metrics
    return opt_metrics;
}
