/**
 * @defgroup   MANYBODYWAVEFUNCTION Many Body Wavefunction
 *
 * @brief      This file implements Many Body Wavefunction.
 *
 * @author     Corey.adams
 * @date       2022
 */

#pragma once

#include "mlqm/config/config.h"
#include "DeepSetsCorrelator.h"
#include "MLP.h"

struct ManyBodyWavefunctionImpl : torch::nn::Module {
    ManyBodyWavefunctionImpl(
        ManyBodyWavefunctionConfig _cfg,
        SamplerConfig sampler_cfg,
        torch::TensorOptions options
    );

    /**
     * @brief      Forward pass of the network
     *
     * @param[in]  x        spatial coordinates
     * @param[in]  spin     The spin
     * @param[in]  isospin  The isospin
     *
     * @return     single-valued wavefunction
     */
    torch::Tensor forward(torch::Tensor x, torch::Tensor spin, torch::Tensor isospin);

    torch::Tensor slater_matrix(torch::Tensor s_input, torch::Tensor s_matrix);

    torch::Tensor spatial_slater(torch::Tensor x);

    // Configuration:
    ManyBodyWavefunctionConfig cfg;

    // Store the Sampler Config for copies:
    SamplerConfig sampler_cfg;

    // Correlator function:
    DeepSetsCorrelator dsc;
    // Individual components:
    // torch::nn::ModuleList spatial_nets;
    std::vector<MLP> spatial_nets;
    // Tensor creation options
    torch::TensorOptions opts;

    // Tensors to simplify the spin and isospin products
    torch::Tensor spin_spinor_2d;
    torch::Tensor isospin_spinor_2d;

};

TORCH_MODULE(ManyBodyWavefunction);
