#include "BaseHamiltonian.h"


torch::Tensor BaseHamiltonian::energy(
    DeepSetsCorrelator wavefunction,
    torch::Tensor inputs)
{
    auto junk = compute_derivatives(wavefunction, inputs);
    return torch::Tensor();
}


std::vector<torch::Tensor> BaseHamiltonian::compute_derivatives(
    DeepSetsCorrelator wavefunction,
    torch::Tensor inputs)
{

    std::cout << "inputs.requires_grad() " << inputs.requires_grad() << std::endl;


    // First, we set the inputs to tensors that require grad:
    inputs.requires_grad_(true);

    std::cout << "inputs.requires_grad() " << inputs.requires_grad() << std::endl;

    // Now, compute the value of the wavefunction:
    torch::Tensor w_of_x = wavefunction(inputs);

    std::cout << "w_of_x.grad_fn()->name() " << w_of_x.grad_fn()->name() << std::endl;

    torch::Tensor v = torch::ones(inputs.sizes());
    w_of_x.backward(v);

    // auto grad_output = torch::ones_like(w_of_x);
    //
    // auto gradient = torch::autograd::grad({w_of_x}, {inputs});
    //
    // std::cout << "grad_output.sizes() " << grad_output.sizes() << std::endl;

/*
        n_walkers = inputs.shape[0]
        n_particles = inputs.shape[1]
        n_dim = inputs.shape[2]

        # Turning off all tape watching except for the inputs:
        # Using the outer-most tape to watch the computation of the first derivative:
        with tf.GradientTape() as tape:
            # Use the inner tape to watch the computation of the wavefunction:
            tape.watch(inputs)
            with tf.GradientTape() as second_tape:
                second_tape.watch(inputs)
                w_of_x = wavefunction(inputs, spin, isospin)
            # Get the derivative of w_of_x with respect to inputs
            dw_dx = second_tape.gradient(w_of_x, inputs)

        # Get the derivative of dw_dx with respect to inputs (aka second derivative)

        # We have to extract the diagonal of the jacobian, which comes out with shape
        # [nwalkers, nparticles, dimension, nwalkers, nparticles, dimension]

        # The indexes represent partial derivative indexes, so,
        # d2w_dx2[i_w, n1,d1, n2, d2] represents the second derivative of the
        # wavefunction at dimension d1

        # This is the full hessian computation:
        d2w_dx2 = tape.batch_jacobian(dw_dx, inputs)

        # And this contracts:
        d2w_dx2 = tf.einsum("wpdpd->wpd",d2w_dx2)

        return w_of_x, dw_dx, d2w_dx2
*/

    return std::vector<torch::Tensor>();
}
