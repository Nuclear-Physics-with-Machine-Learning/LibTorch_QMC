/**
 * @defgroup   NuclearHamiltonian NuclearHamiltonian
 *
 * @brief      This file implements NuclearHamiltonian.
 *
 * @author     Corey.adams
 * @date       2022
 */

#pragma once

#include <torch/torch.h>

#include "mlqm/models/DeepSetsCorrelator.h"
#include "mlqm/config/HamiltonianConfig.h"

class NuclearHamiltonian
{
public:
    NuclearHamiltonian(torch::TensorOptions options){_opts = options;}
    ~NuclearHamiltonian(){}
    

    /**
     * @brief      Calculate the energy of the wavefunction at the specified inputs
     *
     * @param[in]  wavefunction  The wavefunction
     * @param[in]  inputs        The inputs
     *
     * @return     Tensor of per-configuration energy
     */
    torch::Tensor energy(DeepSetsCorrelator wavefunction, torch::Tensor inputs);

    /**
     * @brief      Compute the real kinetic energy
     *
     * @param[in]  w_of_x   The w of x
     * @param[in]  d2w_dx2  The d 2 w dx 2
     *
     * @return     Tensor of per-configuration kinetic energy
     */
    torch::Tensor kinetic_energy(torch::Tensor w_of_x, torch::Tensor d2w_dx2);


    /**
     * @brief      Compute the JF kinetic energy
     *
     * @param[in]  w_of_x   The w of x
     * @param[in]  d2w_dx2  The d 2 w dx 2
     *
     * @return     Tensor of per-configuration JF kinetic energy
     */
    torch::Tensor kinetic_energy_jf(torch::Tensor w_of_x, torch::Tensor dw_dx);

private:

    /**
     * @brief      Calculates the derivatives.
     *
     * @param[in]  wavefunction  The wavefunction
     * @param[in]  inputs        The inputs
     *
     * @return     Wavefunction value + derivatives (1st, 2nd order Hessian diagonal).
     */
    std::vector<torch::Tensor> compute_derivatives(
        DeepSetsCorrelator wavefunction, torch::Tensor inputs);

    // Copy of tensor creation ops, used in compute derivatives
    torch::TensorOptions _opts;

    // configuration object:
    NuclearHamiltonianConfig cfg;
};