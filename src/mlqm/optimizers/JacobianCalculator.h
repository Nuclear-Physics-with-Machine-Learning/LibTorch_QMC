/**
 * @defgroup   JACOBIANCALCULATOR Jacobian Calculator
 *
 * @brief      This file implements Jacobian Calculator.
 *
 * @author     Corey.adams
 * @date       2022
 */

#pragma once

#include <torch/torch.h>

#include "mlqm/models/ManyBodyWavefunction.h"

class JacobianCalculator
{
public:
    // Constructor takes tensor options and a wavefunction
    JacobianCalculator(torch::TensorOptions opts) : options(opts) {}
    ~JacobianCalculator(){}

    /**
     * @brief       Initialize copies of the wavefunction that can be used to batch
     *              Jacobian calculations in either mode.  Only needs to be called once
     *              at initialization.
     *
     *
     * @param[in]  wf           Initialized target wavefunction
     * @param[in]  concurrency  How many copies to make for concurrency.
     */
    void set_parallelization(ManyBodyWavefunction wf, size_t concurrency);

    /**
     * @brief      Compute jacobian of the wavefunction parameters for each configuration
     *             Uses forward mode AD implemented as backwards twice per parameter.
     *
     * @param[in]  x_current     current configurations
     * @param[in]  wavefunction  The wavefunction
     *
     * @return     jacobian matrix of shape [n_walkers, n_parameters]
     */
    torch::Tensor jacobian_forward(torch::Tensor x_current, ManyBodyWavefunction wavefunction);

    /**
     * @brief      Batch calls to the forward mode jacobian calculation, using wf copies.
     *
     * @param[in]  x_current     The x current
     * @param[in]  wavefunction  The wavefunction
     *
     * @return     jacobian_matrix of shape [n_walkers, n_parameters]
     */
    torch::Tensor batch_jacobian_forward(torch::Tensor x_current, ManyBodyWavefunction wavefunction);

    /**
     * @brief      Compute jacobian of the wavefunction parameters for each configuration
     *             Uses reverse mode AD, once per walker.
     *
     * @param[in]  x_current     current configurations
     * @param[in]  wavefunction  The wavefunction
     *
     * @return     jacobian matrix of shape [n_walkers, n_parameters]
     */
    torch::Tensor jacobian_reverse(torch::Tensor x_current, ManyBodyWavefunction wavefunction);

    /**
     * @brief      Batch calls to the reverse mode jacobian calculation, using wf copies.
     *
     * @param[in]  x_current     The x current
     * @param[in]  wavefunction  The wavefunction
     *
     * @return     jacobian_matrix of shape [n_walkers, n_parameters]
     */
    torch::Tensor batch_jacobian_reverse(torch::Tensor x_current, ManyBodyWavefunction wavefunction);


    /**
     * @brief      Sets the weights in the WF based on the set weights
     *
     * @param      wf       The new value
     * @param[in]  weights  The weights
     */
    void set_weights(ManyBodyWavefunction & wf,
        const std::vector<torch::Tensor> & weights);

    /**
     * @brief Compute one column of the jacobian function (one parameter, all walkers)
     */
    torch::Tensor jacobian_forward_weight(torch::Tensor psi,
        torch::Tensor x_current, ManyBodyWavefunction wavefunction, int64_t weight_index);

    /**
     * @brief      Compute the jacobian numerically.  Meant for checking.
     *
     * @param[in]  x_current     The x current
     * @param[in]  wavefunction  The wavefunction
     *
     * @return     Jacobian of walkers, WF weights
     */
    torch::Tensor numerical_jacobian(torch::Tensor x_current, ManyBodyWavefunction wavefunction);

private:

    // Vector of wavefunctions used to parallelize the jacobian.
    std::vector<ManyBodyWavefunction> wf_copies;

    // Tensor Creation Options:
    torch::TensorOptions options;

    // How many flattened parameters?
    int64_t n_parameters;

    // What level of concurrency?
    int64_t concurrency;


    // Store the flattened shape of each parameter layer:
    std::vector<int64_t> flat_shapes;

    // Store the actual shape of each parameter layer:
    std::vector<c10::IntArrayRef> full_shapes;


};
