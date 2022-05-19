#include "NuclearHamiltonian.h"


torch::Tensor NuclearHamiltonian::energy(
    ManyBodyWavefunction wavefunction,
    torch::Tensor inputs)
{
    auto junk = compute_derivatives(wavefunction, inputs);
    return torch::Tensor();
}


std::vector<torch::Tensor> NuclearHamiltonian::compute_derivatives(
    ManyBodyWavefunction wavefunction,
    torch::Tensor inputs)
{


    // First, we set the inputs to tensors that require grad:
    inputs.requires_grad_(true);

    // Now, compute the value of the wavefunction:
    torch::Tensor w_of_x = wavefunction(inputs);

    torch::Tensor v = torch::ones(inputs.sizes());
    
    // Compute the gradients dw_dx:
    auto grad_output = torch::ones_like(w_of_x);
    std::cout << "w_of_x.sizes(): " << w_of_x.sizes() << std::endl;
    auto dw_dx = torch::autograd::grad(
        {w_of_x}, 
        {inputs}, 
        /*grad_outputs=*/{grad_output}, 
        /*retain_graph=*/true, 
        /*create_graph=*/true
        )[0];

    // Now to compute the second derivatives.

    int64_t n_walkers = inputs.sizes()[0];
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
    auto inverse_w = 1./w_of_x;

    // Reduce the hessian but only over spatial dimensions:
    auto summed_d2 = torch::sum(d2w_dx2, {2,});

        // inverse_w = tf.reshape(1/(w_of_x), (-1,1) )
        // # Only reduce over the spatial dimension here:
        // summed_d2 = tf.reduce_sum(d2w_dx2, axis=(2))


    // Compute the kinetic energy:
    auto ke = -(cfg.HBAR * cfg.HBAR / (2*cfg.M)) * torch::sum(inverse_w * summed_d2, {1});
        // ke = -(self.HBAR**2 / (2 * M)) * tf.reduce_sum(inverse_w * summed_d2, axis=1)

    return ke;
}

torch::Tensor NuclearHamiltonian::kinetic_energy_jf(torch::Tensor w_of_x, torch::Tensor dw_dx){
    /*
    < x | KE | psi > / < x | psi > =  1 / 2m [ < x | p | psi > / < x | psi >  = 1/2 w * x**2
    */


    auto internal_arg = dw_dx / torch::reshape(w_of_x, {-1,1,1});

    // Contract d2_w_dx over spatial dimensions and particles:
    auto ke_jf = (cfg.HBAR * cfg.HBAR / (2*cfg.M)) * torch::sum(torch::pow(internal_arg,2), {1,2});

    return ke_jf;
}

/*

    @tf.function
    def kinetic_energy_jf(self, *, w_of_x: tf.Tensor, dw_dx: tf.Tensor, M):
        """Return Kinetic energy

        Calculate and return the KE directly

        Otherwise, exception

        Arguments:
            dw_of_x/dx {tf.Tensor} -- Computed derivative of the wavefunction

        Returns:
            tf.Tensor - kinetic energy (JF) of shape [1]
        """
        # < x | KE | psi > / < x | psi > =  1 / 2m [ < x | p | psi > / < x | psi >  = 1/2 w * x**2



        internal_arg = dw_dx / tf.reshape(w_of_x, (-1,1,1))

        # Contract d2_w_dx over spatial dimensions and particles:
        ke_jf = (self.HBAR**2 / (2 * M)) * tf.reduce_sum(internal_arg**2, axis=(1,2))


        return ke_jf

    @tf.function
    def kinetic_energy(self, *, w_of_x : tf.Tensor, d2w_dx2 : tf.Tensor, M):
        """Return Kinetic energy


        If all arguments are supplied, calculate and return the KE.

        Arguments:
            d2w_dx2 {tf.Tensor} -- Computed second derivative of the wavefunction
            KE_JF {tf.Tensor} -- JF computation of the kinetic energy

        Returns:
            tf.Tensor - potential energy of shape [1]
        """


        inverse_w = tf.reshape(1/(w_of_x), (-1,1) )
        # Only reduce over the spatial dimension here:
        summed_d2 = tf.reduce_sum(d2w_dx2, axis=(2))

        ke = -(self.HBAR**2 / (2 * M)) * tf.reduce_sum(inverse_w * summed_d2, axis=1)

        return ke

*/