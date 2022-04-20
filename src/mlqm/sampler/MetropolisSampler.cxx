/**
 * @defgroup   METROPOLISSAMPLER Metropolis Sampler
 *
 * @brief      This file implements Metropolis Sampler.
 *
 * @author     Corey.adams
 * @date       2022
 */

#include "MetropolisSampler.h"


MetropolisSampler::MetropolisSampler(SamplerConfig _cfg)
    : cfg(_cfg)
{
    // Initialize the tensor

    this -> x = torch::randn({cfg.n_walkers,cfg.n_particles,cfg.n_dim});

}

torch::Tensor MetropolisSampler::kick(int n_kicks, DeepSetsCorrelator wavefunction){


        // We need to compute the wave function twice:
        // Once for the original coordiate, and again for the kicked coordinates
        torch::Tensor acceptance = torch::zeros({1});
        // Calculate the current wavefunction value:
        torch::Tensor current_wavefunction = wavefunction(x);

        // Generate a long set of random number from which we will pull:
        torch::Tensor random_numbers = torch::rand({n_kicks, cfg.n_walkers,1});

        // Generate a long list of kicks:
        torch::Tensor kicks = torch::randn({n_kicks, cfg.n_walkers,cfg.n_particles,cfg.n_dim});

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
            auto condition = probability > random_numbers[i_kick];

            // Gather the current values of the wavefunction:
            current_wavefunction = at::where(condition, kicked_wavefunction, current_wavefunction);

            // std::cout << "condition: " << condition << std::endl;
            // std::cout << "kicked shape: " << kicked.sizes() << std::endl;
            // std::cout << "x shape: " << x.sizes() << std::endl;

            auto reshaped_condition = torch::reshape(condition, {-1, 1, 1});

            x = at::where(reshaped_condition, kicked, x);

            acceptance += torch::mean(at::_cast_Float(condition));

        }
   
        acceptance /= n_kicks;

        std::cout << "Kicking done!" << std::endl;

 /*
  *



        for i_kick in tf.range(nkicks):

           

            # [nwalkers, 1]


            probability = tf.math.pow(kicked_wavefunction / current_wavefunction,2)
            # Acceptance is whether the probability for that walker is greater than
            # a random number between [0, 1).
            # Pull the random numbers and create a boolean array
            # accept      = probability >  tf.random.uniform(shape=[shape[0],1])
            accept      = probability >  tf.gather(random_numbers, i_kick)
            # accept      = probability >  tf.math.log(tf.random.uniform(shape=[shape[0],1]))

            # Grab the kicked wavefunction in the places it is new, to speed up metropolis:
            current_wavefunction = tf.where(accept, kicked_wavefunction, current_wavefunction)

            # # We need to broadcast accept to match the right shape
            # # Needs to come out to the shape [nwalkers, nparticles, ndim]
            # spatial_accept = tf.tile(accept, [1,tf.reduce_prod(shape[1:])])
            # spatial_accept = tf.reshape(spatial_accept, shape)
            # walkers = tf.where(spatial_accept, kicked, walkers)

            walkers = tf.where(tf.reshape(accept, (-1,1,1)), kicked, walkers)

            acceptance = tf.reduce_mean(tf.cast(accept, dtype=dtype))
        */
        return acceptance;

}