/**
 * @defgroup   BASEOPTIMIZER Base Optimizer
 *
 * @brief      This file implements Base Optimizer.
 *
 * @author     Corey.adams
 * @date       2022
 */

#pragma once


#include "mlqm/models/ManyBodyWavefunction.h"
#include "mlqm/sampler/MetropolisSampler.h"
#include "mlqm/hamiltonians/NuclearHamiltonian.h"
#include "mlqm/config/config.h"

#include "Accumulator.h"
#include "GradientCalculator.h"

class BaseOptimizer
{
public:
    BaseOptimizer(Config cfg, torch::TensorOptions opts);
    ~BaseOptimizer(){}
    
    /**
     * @brief      Compute jacobian of the wavefunction parameters for each configuration
     *
     * @param[in]  x_current     current configurations
     * @param[in]  wavefunction  The wavefunction
     *
     * @return     jacobian matrix of shape [n_walkers, n_parameters]
     */
    torch::Tensor jacobian(torch::Tensor x_current, ManyBodyWavefunction wavefunction);

    std::vector<torch::Tensor> compute_O_observables(
        torch::Tensor flattened_jacobian, 
        torch::Tensor energy, 
        torch::Tensor w_of_x);

    /**
     * @brief      Run the thermalization for some number of iterations
     *
     * @param[in]  n_iterations  The n iterations
     */
    torch::Tensor equilibrate(int64_t n_iterations);

    /**
     * @brief      Recompute the energy
     *
     * @param[in]  test_wf         The test wf
     * @param[in]  current_w_of_x  The current w of x
     *
     * @return     { description_of_the_return_value }
     */
    std::vector<torch::Tensor> recompute_energy(ManyBodyWavefunction test_wf, torch::Tensor current_w_of_x);

    /**
     * @brief      Calculates the gradients.
     *
     * @param[in]  dpsi_i     The dpsi i
     * @param[in]  energy     The energy
     * @param[in]  dpsi_i_EL  The dpsi i el
     * @param[in]  dpsi_ij    The dpsi ij
     * @param[in]  torch      The torch
     *
     * @return     The gradients.
     */
    torch::Tensor compute_gradients(
        torch::Tensor dpsi_i,
        torch::Tensor energy,
        torch::Tensor dpsi_i_EL,
        torch::Tensor dpsi_ij,
        torch::Tensor eps,
        torch::Tensor & S_ij);

    std::vector<torch::Tensor> walk_and_accumulate_observables();
    
    std::map<std::string, torch::Tensor> compute_updates_and_metrics(torch::Tensor & gradients);
    void unflatten_weights_or_gradients();

    std::map<std::string, torch::Tensor> sr_step();

private:

    // Overall configuration:
    Config config;
    
    // Tensor Creation Options:
    torch::TensorOptions options;

    // Used to initialize, draw, and thermalize configurations:
    MetropolisSampler sampler;

    // The main wavefunction:
    ManyBodyWavefunction wavefunction = nullptr;

    // The trial wavefunction for update optimization:
    ManyBodyWavefunction adaptive_wavefunction = nullptr;

    // The hamiltonian class
    NuclearHamiltonian hamiltonian;

    // Tool for gradient calculations
    GradientCalculator grad_calc;

    // Way to accumlate and all reduce objects:
    Accumulator estimator;
 
    // Global MPI size
    int size;

    const bool MPI_AVAILABLE = false;

    // estimation of predicted energy:
    torch::Tensor predicted_energy;

    // Total number of parameters in the wavefunction:
    int64_t n_parameters;

    // Store the flattened shape of each parameter layer:
    std::vector<int64_t> flat_shapes;

    // Store the actual shape of each parameter layer:
    std::vector<c10::IntArrayRef> full_shapes;

    // Objects that are useful to have stored:
    int n_loops_total;


};



