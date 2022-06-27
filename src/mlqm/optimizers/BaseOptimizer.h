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
#include "JacobianCalculator.h"

class BaseOptimizer
{
public:
    BaseOptimizer(Config cfg, torch::TensorOptions opts);
    ~BaseOptimizer(){}

    /**
     * @brief      Calculates the O observables. (dpsi_i, dpsi_ij, dpsi_i_EL)
     *
     * @param[in]  flattened_jacobian  The flattened jacobian
     * @param[in]  energy              The energy
     * @param[in]  w_of_x              The w of x
     *
     * @return     The o observables.
     */
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
    torch::Tensor recompute_energy(
        ManyBodyWavefunction test_wf, std::vector<torch::Tensor> current_w_of_x,
        torch::Tensor & overlap, torch::Tensor & acos);

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

    /**
     * @brief      Compute the next energy and optimization metrics
     *
     * @param[in]  current_psi  The current psi
     * @param      gradients    The gradients
     * @param      next_energy  The next energy
     *
     * @return     optimiziation metrics
     */
    std::map<std::string, torch::Tensor> flat_optimizer(
        std::vector<torch::Tensor> current_psi,
        std::vector<torch::Tensor> & gradients,
        torch::Tensor & next_energy);


    /**
     * @brief      Main loop implementation per iteration
     *
     * @return     Returns tensor of current_psi values for each observation step
     */
    std::vector<torch::Tensor> walk_and_accumulate_observables();

    /**
     * @brief      Calculates the updates and metrics.
     *
     * @param[in]  current_psi  The current psi
     * @param      gradients    The gradients
     * @param      next_energy  The next energy
     *
     * @return     The metrics.
     */
    std::map<std::string, torch::Tensor> compute_updates_and_metrics(
        std::vector<torch::Tensor> current_psi,
        std::vector<torch::Tensor> & gradients,
        torch::Tensor & next_energy);

    /**
     * @brief      Divide and reshape the weights/gradients to match the model.
     *
     * @param[in]  flat_gradients  The flat gradients
     *
     * @return     { description_of_the_return_value }
     */
    std::vector<torch::Tensor> unflatten_weights_or_gradients(torch::Tensor flat_gradients);

    /**
     * @brief      Run the single iteration of Stochastic Reconfiguration
     *
     * @return     returns the dictionary of metric values
     */
    std::map<std::string, torch::Tensor> sr_step();

    /**
     * @brief      Update the wavefunction with the supplied gradients
     *
     * @param[in]  gradients  The gradients
     */
    void apply_gradients(const std::vector<torch::Tensor> & gradients);

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
    Accumulator estimator, re_estimator;

    // Tool for computing jacobians:
    JacobianCalculator jac_calc;

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
