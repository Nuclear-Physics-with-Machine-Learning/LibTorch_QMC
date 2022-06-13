#include "BaseOptimizer.h"

#include <chrono>

#ifdef OPENMP_FOUND
#include <omp.h>
#endif

BaseOptimizer::BaseOptimizer(Config cfg, torch::TensorOptions opts)
    : config(cfg),
      options(opts),
      sampler(cfg.sampler, opts),
      hamiltonian(cfg.hamiltonian, opts),
      grad_calc(opts)
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

    PLOG_DEBUG << " -- Coordinating loop length";


    // We do a check that n_loops_total * n_concurrent_obs_per_rank matches expectations:
    if (n_loops_total * config.sampler.n_concurrent_obs_per_rank*size != config.sampler.n_observable_measurements){
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
    wavefunction = ManyBodyWavefunction(cfg.wavefunction, options, cfg.sampler.n_particles);
    wavefunction -> to(options.device());
    wavefunction -> to(torch::kFloat64);

    auto params = wavefunction -> named_parameters();

    n_parameters = 0;
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
    }

    PLOG_INFO << "Total parameters: " << n_parameters;

    // Initialize random input:
    auto input = sampler.sample_x();

    // Thermalize the walkers for some large number of initial thermalizations:
    PLOG_INFO << "Begin initial equilibration";
    // float acceptance = equilibrate(cfg.sampler.n_thermalize).data_ptr<double>()[0];
    auto acceptance = equilibrate(cfg.sampler.n_thermalize);
    PLOG_INFO << "Done initial equilibration, acceptance = " << acceptance;

    auto metrics = sr_step();

    for (int64_t i = 0; i < 10; i ++){
        auto start = std::chrono::high_resolution_clock::now();
        metrics = sr_step();
        auto stop = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli>  duration = stop - start;
        PLOG_INFO << "Duration: " << duration.count() << "[ms]";
    }
    PLOG_INFO << metrics;

    // PLOG_DEBUG << jac;

  // std::cout << "input.sizes(): "<< input.sizes() << std::endl;

  // auto w_of_x = wavefunction(input);

  // auto start = std::chrono::high_resolution_clock::now();
  // w_of_x = wavefunction(input);
  // auto stop = std::chrono::high_resolution_clock::now();
  // auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
  // std::cout << "Just WF time: " << (duration.count()) << " milliseconds" << std::endl;
  // std::cout << "w_of_x.sizes(): " << w_of_x.sizes() << std::endl;


}

torch::Tensor BaseOptimizer::equilibrate(int64_t n_iterations){
    torch::Tensor acceptance = sampler.kick(n_iterations, wavefunction);
    return acceptance;
}


torch::Tensor BaseOptimizer::jacobian(torch::Tensor x_current, ManyBodyWavefunction wavefunction){

    // This function goes through the jacobian like so:
    // -for every walker, compute the gradient of the wavefunction output
    // with respect to the parameters.
    // Then, compactify (real word?) the gradients into a flat tensor.
    // Return the tensor (shape of [n_walkers, n_parameters])

    // How many walkers?
    auto n_walkers = x_current.sizes()[0];


    // Compute the value of the wavefunction for all walkers:
    auto psi = wavefunction(x_current);

    // psi = psi.repeat(n_walkers);
    auto psi_v = torch::chunk(psi, n_walkers);


    // Construct a series of grad outputs:
    auto outputs = torch::eye(n_walkers);

    // Flatten and chunk it:
    auto output_list = torch::chunk(torch::flatten(outputs), n_walkers);



    auto jacobian_flat = torch::zeros({n_walkers, n_parameters}, options);

    #pragma omp parallel for
    for (int64_t i_walker = 0; i_walker < n_walkers; ++i_walker){

        // Compute the gradients:
        auto jacobian_list = torch::autograd::grad(
            {psi_v[i_walker]}, // outputs
            wavefunction->parameters(), //inputs
            {output_list[i_walker]}, // grad_outputs
            true,  // retain graph
            false, // create_graph
            false  // allow_unused
        );
        std::vector<torch::Tensor> flat_this_jac;
        for(auto & grad : jacobian_list) flat_this_jac.push_back(grad.flatten());
        // Flatten the jacobian and put it into the larger matrix.
        // Note the normalization by wavefunction happens here too.
        jacobian_flat.index_put_({i_walker, Slice()}, torch::cat(flat_this_jac) / psi_v[i_walker]);

    }


    // PLOG_DEBUG << jacobian_flat;

    return jacobian_flat;

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
    // auto dpsi_ij = tf.linalg.matmul(normed_fj_t, normed_fj, transpose_a = True) / self.n_walkers_per_observation


    ///TODO: Compute <O^m H>
    // # Computing <O^m H>:
    auto e_reshaped = torch::reshape(energy, {1,config.sampler.n_walkers});


    auto dpsi_i_EL = torch::matmul(e_reshaped, normed_fj);
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

        for (auto w : w_of_x_v) current_psi.push_back(w);



        // For each observation, we compute the jacobian.
        // flattened_jacobian, flat_shape = self.batched_jacobian(
            // config.sampler.n_concurrent_obs_per_rank, x_current,
            // spin, isospin, _wavefunction, self.jacobian)

        auto jac = jacobian(x_current, wavefunction);

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
            estimator.accumulate("acceptance", acceptance);
            estimator.accumulate("r",          r);
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

    // self.latest_gradients = None
    // self.latest_psi       = None



    // We do a thermalization step again:
    equilibrate(config.sampler.n_thermalize);

    // Clear the walker history:
    sampler.reset_history();


    // Now, actually apply the loop and compute the observables:
    std::vector<torch::Tensor> current_psi = walk_and_accumulate_observables();

    // At this point, we need to average the observables that feed into the optimizer:
    estimator.finalize(config.sampler.n_observable_measurements);


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

    PLOG_INFO << "Metrics set";


    // Here, we call the function to optimize eps and compute the gradients:

    ///TODO: HERE
    torch::Tensor next_energy;
    // delta_p, opt_metrics, next_energy = self.compute_updates_and_metrics(current_psi)

    ///TODO: HERE
    // metrics.update(opt_metrics)
    // # Compute the ratio of the previous energy and the current energy
    // auto energy_diff = predicted_energy - estimator["energy"];
    // metrics["energy/energy_diff"] = energy_diff;

    torch::Tensor gradients;
    auto opt_metrics = compute_updates_and_metrics(gradients);
        // # dp_i, opt_metrics = self.grad_calc.sr(
        // #     self.estimator["energy"],
        // #     self.estimator["dpsi_i"],
        // #     self.estimator["dpsi_i_EL"],
        // #     self.estimator["dpsi_ij"])



    // ///TODO HERE
    // // And apply them to the wave function:
    // apply_gradients(delta_p, self.wavefunction.trainable_variables);
    // self.latest_gradients = delta_p
    // self.latest_psi = current_psi

    // Before moving on, set the predicted_energy:
    predicted_energy = next_energy;

    return  metrics;

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

std::map<std::string, torch::Tensor> BaseOptimizer::compute_updates_and_metrics(torch::Tensor & gradients){

    std::map<std::string, torch::Tensor> opt_metrics;
    return opt_metrics;

}
