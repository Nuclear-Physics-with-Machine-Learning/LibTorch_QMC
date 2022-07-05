#include "NuclearHamiltonian.h"


NuclearHamiltonian::NuclearHamiltonian(
    NuclearHamiltonianConfig config, torch::TensorOptions options) :
    cfg(config)
{_opts = options;}

torch::Tensor NuclearHamiltonian::energy(
    ManyBodyWavefunction wavefunction,
    torch::Tensor inputs, torch::Tensor spin, torch::Tensor isospin,
    torch::Tensor & energy_jf,
    torch::Tensor & ke_jf,
    torch::Tensor & ke_direct,
    torch::Tensor & pe,
    torch::Tensor & w_of_x)
{

    // Get the necessary derivatives:
    auto derivatives = compute_derivatives(wavefunction, inputs, spin, isospin);

    w_of_x                = derivatives[0];
    torch::Tensor dw_dx   = derivatives[1];
    torch::Tensor d2w_dx2 = derivatives[2];

    // PLOG_INFO << "w_of_x.sizes() " << w_of_x.sizes();
    // PLOG_INFO << "torch::mean(w_of_x) " << torch::mean(w_of_x);
    // PLOG_INFO << "dw_dx.sizes() " << dw_dx.sizes();
    // PLOG_INFO << "torch::mean(dw_dx) " << torch::mean(dw_dx);
    // PLOG_INFO << "d2w_dx2.sizes() " << d2w_dx2.sizes();
    // PLOG_INFO << "torch::mean(d2w_dx2) " << torch::mean(d2w_dx2);

    // This function takes the inputs
    // And computes the expectation value of the energy at each input point
    pe        = potential_energy(inputs);
    // PLOG_INFO << "pe.sizes() " << pe.sizes();
    // PLOG_INFO << "torch::mean(pe) " << torch::mean(pe);
    ke_jf     = kinetic_energy_jf(w_of_x, dw_dx);
    // PLOG_INFO << "ke_jf.sizes() " << ke_jf.sizes();
    // PLOG_INFO << "torch::mean(ke_jf) " << torch::mean(ke_jf);
    ke_direct = kinetic_energy(w_of_x, d2w_dx2);
    // PLOG_INFO << "ke_direct.sizes() " << ke_direct.sizes();
    // PLOG_INFO << "torch::mean(ke_direct) " << torch::mean(ke_direct);


    // Total energy computations:
    torch::Tensor energy = pe + ke_direct;
    energy_jf            = pe + ke_jf;

    return energy;
}


std::vector<torch::Tensor> NuclearHamiltonian::compute_derivatives(
    ManyBodyWavefunction wavefunction,
    torch::Tensor inputs, torch::Tensor spin, torch::Tensor isospin)
{

    // First, we set the inputs to tensors that require grad:
    inputs.requires_grad_(true);

    // Now, compute the value of the wavefunction:
    torch::Tensor w_of_x = wavefunction(inputs, spin, isospin);


    torch::Tensor v = torch::ones(inputs.sizes());

    // Compute the gradients dw_dx:
    auto grad_output = torch::ones_like(w_of_x);
    auto dw_dx = torch::autograd::grad(
        {w_of_x},
        {inputs},
        /*grad_outputs=*/{grad_output},
        /*retain_graph=*/true,
        /*create_graph=*/true
        )[0];

    // Now to compute the second derivatives.

    // int64_t n_walkers = inputs.sizes()[0];
    int64_t n_particles = inputs.sizes()[1];
    int64_t n_dim = inputs.sizes()[2];


    torch::Tensor d2w_dx2 = torch::zeros_like(dw_dx, _opts);

    for(int64_t i_particle = 0; i_particle < n_particles; i_particle++){
        for(int64_t j_dim = 0; j_dim < n_dim; j_dim++){
            // Take a slice of the derivatives:
            auto dw_dx_ij = dw_dx.index({Slice(), i_particle, j_dim});
            // std::cout << "dw_dx_ij.sizes(): " << dw_dx_ij.sizes() << std::endl;
            auto d2w_dx2_ij = torch::autograd::grad(
                /*outputs=*/ {dw_dx_ij},
                /*inputs=*/  {inputs},
                /*grad_outputs=*/ {grad_output},
                /*retain_graph=*/ true,
                /*create_graph=*/ true
                )[0];
            auto this_grad_slice = d2w_dx2_ij.index({Slice(), i_particle, j_dim});
            d2w_dx2.index_put_({Slice(), i_particle, j_dim}, this_grad_slice);
        }
    }

    std::vector<torch::Tensor> val_and_grads;
    val_and_grads.push_back(w_of_x);
    val_and_grads.push_back(dw_dx);
    val_and_grads.push_back(d2w_dx2);

    return val_and_grads;

}

torch::Tensor NuclearHamiltonian::kinetic_energy(torch::Tensor w_of_x, torch::Tensor d2w_dx2){

    // We need to weigh with 1/w_of_x
    auto inverse_w = torch::reshape(1./(w_of_x + 1e-8), {-1, 1});

    // Reduce the hessian but only over spatial dimensions:
    auto summed_d2 = torch::sum(d2w_dx2, {2,});

    // Compute the kinetic energy:
    auto ke =  torch::sum(inverse_w * summed_d2, {1});

    return -(cfg.HBAR * cfg.HBAR / (2*cfg.M)) * ke;
}

torch::Tensor NuclearHamiltonian::kinetic_energy_jf(torch::Tensor w_of_x, torch::Tensor dw_dx){
    /*
    < x | KE | psi > / < x | psi > =  1 / 2m [ < x | p | psi > / < x | psi >  = 1/2 w * x**2
    */

    auto reshaped = torch::reshape(w_of_x, {-1,1,1});
    // PLOG_INFO << "reshaped: " << reshaped;

    auto internal_arg = dw_dx / (reshaped + 1e-8);
    // PLOG_INFO << "dw_dx: " << dw_dx;
    // PLOG_INFO << "internal_arg: " << internal_arg;
    // PLOG_INFO << "torch::pow(internal_arg, 2): " << torch::pow(internal_arg, 2);
    // Contract d2_w_dx over spatial dimensions and particles:
    auto ke_jf = torch::sum(torch::pow(internal_arg,2), {1,2});
    // PLOG_INFO << "ke_jf: " << ke_jf;

    return (cfg.HBAR * cfg.HBAR / (2*cfg.M)) * ke_jf;
}

torch::Tensor NuclearHamiltonian::potential_energy(torch::Tensor inputs){
    auto x_squared = 0.5 * cfg.M * cfg.OMEGA*cfg.OMEGA * torch::sum(torch::pow(inputs, 2.), {1, 2});
    return x_squared;

}

/*


*/
