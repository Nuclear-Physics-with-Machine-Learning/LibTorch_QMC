/**
 * @brief      This class describes a gradient calculator.
 */

#pragma once

#include <torch/torch.h>

class GradientCalculator
{
public:
    GradientCalculator(torch::TensorOptions opt) : options(opt){}
    ~GradientCalculator(){}


    torch::Tensor f_i(torch::Tensor dpsi_i, torch::Tensor energy, torch::Tensor dpsi_i_EL);
    torch::Tensor S_ij(torch::Tensor dpsi_ij, torch::Tensor dpsi_i);
    torch::Tensor par_dist(torch::Tensor dp_i, torch::Tensor S_ij);
    torch::Tensor regularize_S_ij(torch::Tensor S_ij, torch::Tensor eps);
    torch::Tensor pd_solve(torch::Tensor S_ij, torch::Tensor eps, torch::Tensor f_i);
    

private:
    // Tensor Creation Options:
    torch::TensorOptions options;

};