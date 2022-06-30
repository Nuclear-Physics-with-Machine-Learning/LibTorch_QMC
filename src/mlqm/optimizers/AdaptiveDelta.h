/**
 * @defgroup   Adaptive Delta Optimizer
 *
 * @brief      This file implements Adaptive Delta (learning rate) Optimizer.
 *
 * @author     Corey.adams
 * @date       2022
 */

#pragma once

#include "BaseOptimizer.h"

class AdaptiveDelta : public BaseOptimizer
{

public:
    AdaptiveDelta(Config cfg, torch::TensorOptions opts);
    ~AdaptiveDelta(){}

    std::map<std::string, torch::Tensor> compute_updates_and_metrics(
        std::vector<torch::Tensor> current_psi,
        std::vector<torch::Tensor> & gradients,
        torch::Tensor & next_energy) override;

    std::map<std::string, torch::Tensor> delta_optimizer(
        std::vector<torch::Tensor> current_psi,
        std::vector<torch::Tensor> & gradients,
        torch::Tensor & next_energy);
};
