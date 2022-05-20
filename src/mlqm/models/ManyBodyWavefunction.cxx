#include "ManyBodyWavefunction.h"

torch::Tensor ManyBodyWavefunctionImpl::forward(torch::Tensor x){

    // The input will be of the shape [N_walkers, n_particles, n_dim]
    // c10::InferenceMode guard(true);

    return dsc(x);


}
