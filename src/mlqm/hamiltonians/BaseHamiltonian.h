/**
 * @defgroup   BASEHAMILTONIAN Base Hamiltonian
 *
 * @brief      This file implements Base Hamiltonian.
 *
 * @author     Corey.adams
 * @date       2022
 */

#pragma once

#include <torch/torch.h>

#include "mlqm/models/DeepSetsCorrelator.h"

class BaseHamiltonian
{
public:
    BaseHamiltonian(){}
    ~BaseHamiltonian(){}
    

    torch::Tensor energy(DeepSetsCorrelator wavefunction, torch::Tensor inputs);

private:


    std::vector<torch::Tensor> compute_derivatives(
        DeepSetsCorrelator wavefunction, torch::Tensor inputs);
    
};