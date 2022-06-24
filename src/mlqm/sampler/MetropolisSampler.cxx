/**
 * @defgroup   METROPOLISSAMPLER Metropolis Sampler
 *
 * @brief      This file implements Metropolis Sampler.
 *
 * @author     Corey.adams
 * @date       2022
 */

#include "MetropolisSampler.h"


MetropolisSampler::MetropolisSampler(SamplerConfig _cfg, torch::TensorOptions options)
    : cfg(_cfg),
      opts(options), kick_opts(options)
{
    // Initialize the tensor

    // We do an optimization to parallize observations
    n_walkers_opt_shape = cfg.n_walkers * cfg.n_concurrent_obs_per_rank;


    this -> x = torch::randn({n_walkers_opt_shape, cfg.n_particles, cfg.n_dim}, opts);
    if(_cfg.use_spin){
        this -> spin    = -1. * torch::ones({n_walkers_opt_shape,cfg.n_particles}, opts); // spin either up/down
        // Set a fixed number to 1:
        for (int64_t i = 0; i < cfg.n_spin_up; i++){
            if (i >= cfg.n_particles) break;
            spin.index_put_({Slice{}, i},1.0);
        }
    }
    if(_cfg.use_isospin){
        this -> isospin = -1 * torch::ones({n_walkers_opt_shape,cfg.n_particles}, opts); // isospin either up/down
        // Set a fixed number to 1:
        for (int64_t i = 0; i < cfg.n_protons; i++){
            if (i >= cfg.n_particles) break;
            isospin.index_put_({Slice{}, i},1.0);
        }
    }

    kick_opts = kick_opts.dtype(torch::kFloat32);

}

torch::Tensor MetropolisSampler::sample_x(){
    stored_x_history.push_back(torch::clone(x));
    return torch::clone(x);
}

torch::Tensor MetropolisSampler::sample_spin(){
    stored_spin_history.push_back(torch::clone(spin));
    return torch::clone(spin);
}
torch::Tensor MetropolisSampler::sample_isospin(){
    stored_isospin_history.push_back(torch::clone(isospin));
    return torch::clone(isospin);
}

void MetropolisSampler::reset_history(){
    stored_x_history.clear();
    stored_spin_history.clear();
    stored_isospin_history.clear();
}


torch::Tensor MetropolisSampler::kick(int n_kicks, ManyBodyWavefunction wavefunction){

    // First, convert to lower precision for this part!
    x = x.to(torch::kFloat32);
    wavefunction -> to(torch::kFloat32);

    torch::Tensor acceptance = torch::zeros({1}, kick_opts);

    ///TODO Initialize spin with non-zero WF
    {
        // Apply inference mode in this function
        c10::InferenceMode guard(true);


        // We need to compute the wave function twice:
        // Once for the original coordiate, and again for the kicked coordinates
        // Calculate the current wavefunction value:
        torch::Tensor current_wavefunction = wavefunction(x);

        // Generate a long set of random number from which we will pull:
        torch::Tensor random_numbers = torch::rand({n_kicks, n_walkers_opt_shape}, kick_opts);

        // Generate a long list of kicks:
        torch::Tensor kicks = torch::randn({n_kicks, n_walkers_opt_shape,cfg.n_particles,cfg.n_dim}, kick_opts);

        kicks = cfg.kick_mean + cfg.kick_std * kicks;

        // Adding spin:
        //  A meaningful metropolis move is to pick a pair and exchange the spin
        //  ONly one pair gets swapped at a time
        //  Change the isospin of a pair as well.
        //  The spin coordinate is 2 dimensions per particle: spin and isospin (each up/down)
        //

        // Computing modulus squa f wavefunction in new vs old coordinates
        //  - this kicks randomly with a guassian, and has an acceptance probaility
        // However, what we can do instead is to add a drift term
        // Instead of kicking with a random gaussian, we compute the derivative
        // with respect to X.
        // Multiply it by sigma^2
        // Then, clip the drift so it is not too large.
        // New coordinates are the old + gaussian + drift
        // Acceptance is ratio of modulus sq d wavefunction IF the move is symmetric
        // So need to weight the modulus with a drift reweighting term.


        // Spin typically thermalizes first.
        // Fewer spin configurations allowed due to total spin conservation
        //

        // Loop over every kick and perform the kick:

        for (int i_kick = 0; i_kick < n_kicks; i_kick ++){
             // Create a kick:
            torch::Tensor kicked = x + kicks[i_kick];

            // Compute the values of the wave function, which should be of shape
            torch::Tensor kicked_wavefunction = wavefunction(kicked);

            // Probability is the ratio of kicked **2 to original
            torch::Tensor probability = (kicked_wavefunction / current_wavefunction).pow_(2);

            // Find the locations where the move is accepted:
            torch::Tensor condition = probability > random_numbers[i_kick];

            // Gather the current values of the wavefunction:
            current_wavefunction = at::where(condition, kicked_wavefunction, current_wavefunction);

            // std::cout << "condition: " << condition << std::endl;
            // std::cout << "kicked shape: " << kicked.sizes() << std::endl;
            // std::cout << "x shape: " << x.sizes() << std::endl;

            torch::Tensor reshaped_condition = torch::reshape(condition, {-1, 1, 1});

            x = at::where(reshaped_condition, kicked, x);

            acceptance += torch::mean(at::_cast_Float(condition));

        }

        acceptance /= n_kicks;
    } // inference mode
    // Finally, back to double precision
    x = x.to(torch::kFloat64);
    wavefunction -> to(torch::kFloat64);


    return acceptance;

}
