#include "JacobianCalculator.h"




void JacobianCalculator::set_weights(ManyBodyWavefunction & wf,
    const std::vector<torch::Tensor> & weights){

    c10::InferenceMode guard(true);

    auto current_params = wf->parameters();

    // Even though it's a flat optimization, we recompute the energy to get the overlap too:
    for (size_t i_layer = 0; i_layer < current_params.size(); i_layer ++){

        // Bit of a crappy way to do this, may need to optimize later
        wf -> parameters()[i_layer] += weights[i_layer] - current_params[i_layer];
        // PLOG_INFO << "  set to :" << adaptive_wavefunction->parameters()[i_layer];
    }

}

void JacobianCalculator::set_parallelization(ManyBodyWavefunction wf, size_t concurrency){
    // Initialize the copies of the wavefunction used in the batch_jacobian:

    // We capture the number of parameters here, too.
    n_parameters = 0;
    for (auto & key_pair : wf->named_parameters()){
        int64_t local_params = 1;
        for (auto & s : key_pair.value().sizes()){
            local_params *= s;
        }
        // Store the shapes:
        flat_shapes.push_back(local_params);
        full_shapes.push_back(key_pair.value().sizes());
        // Add this layer to the total:
        n_parameters += local_params;

        // Set the adaptive wavefunction parameters to the original one:
        // adaptive_wavefunction->named_parameters()[key_pair.key()].set_(key_pair.value());
    }


    wf_copies.clear();
    // don't worry about setting parameters yet.
    for (size_t i = 0; i < concurrency; i ++){
        wf_copies.push_back(ManyBodyWavefunction(wf->cfg, options));
        wf_copies.back() -> to(options.device());
        wf_copies.back() -> to(torch::kFloat64);
    }

    this->concurrency = concurrency;

}

torch::Tensor JacobianCalculator::jacobian_reverse(
    torch::Tensor x_current, ManyBodyWavefunction wavefunction){
    // This function goes through the jacobian like so:
    // Duplicate the input by n_parameters
    // How many walkers?
    auto n_walkers = x_current.sizes()[0];

    // Compute the value of the wavefunction for all walkers:
    auto psi = wavefunction(x_current);

    // psi = psi.repeat(n_walkers);
    auto psi_v = torch::chunk(psi, n_walkers);

    //
    // // Construct a series of grad outputs:
    // auto outputs = torch::eye(n_walkers, options);
    //
    // // Flatten and chunk it:
    // auto output_list = torch::chunk(torch::flatten(outputs), n_walkers);
    //


    auto jacobian_flat = torch::zeros({n_walkers, n_parameters}, options);

    int64_t i_walker = 0;

    for (i_walker = 0; i_walker < n_walkers; ++i_walker){

        // Compute the gradients:
        auto jacobian_list = torch::autograd::grad(
            {psi_v[i_walker]}, // outputs
            wavefunction->parameters(), //inputs
            {}, // grad_outputs
            // {output_list[i_walker]}, // grad_outputs
            true,  // retain graph
            false, // create_graph
            false  // allow_unused
        );
        std::vector<torch::Tensor> flat_this_jac;
        for(auto & grad : jacobian_list) flat_this_jac.push_back(grad.flatten());

        // Flatten the jacobian and put it into the larger matrix.
        // Note the normalization by wavefunction happens here too.
        jacobian_flat.index_put_({i_walker, Slice()}, torch::cat(flat_this_jac));
    }

    return jacobian_flat;

}


torch::Tensor JacobianCalculator::batch_jacobian_reverse(
    torch::Tensor x_current, ManyBodyWavefunction wavefunction){

    // First, make n copies of the wavefunction:

    size_t n_wavefunctions = wf_copies.size();
    #pragma omp parallel for
    for (size_t i = 0; i < n_wavefunctions; i ++){
        // Set the parameters:
        set_weights(wf_copies[i], wavefunction->parameters());

    }

    // Next, chunk the inputs:
    auto input_chunks = torch::chunk(x_current, concurrency);


    // Prepare a list of jacobian chunks for output:
    std::vector<torch::Tensor> jac_chunks;
    jac_chunks.resize(concurrency);

    int64_t i_chunk;

    #pragma omp parallel for
    for (i_chunk = 0; i_chunk < concurrency; i_chunk ++){
        jac_chunks[i_chunk] = jacobian_reverse(input_chunks[i_chunk], wf_copies[i_chunk]);
    }


    // Stack up the jacobian chunks:
    auto jacobian_full =  torch::cat(jac_chunks, 0);

    return jacobian_full;

}


torch::Tensor JacobianCalculator::jacobian_forward(
    torch::Tensor x_current, ManyBodyWavefunction wavefunction){

    // PLOG_INFO << "Enter forward";

    // How many walkers?
    auto n_walkers = x_current.sizes()[0];

    // Compute the value of the wavefunction for all walkers:
    auto psi = wavefunction(x_current);

    auto jacobian_flat = torch::zeros({n_walkers, n_parameters}, options);

    int64_t i_column = 0;

    // Loop over the parameters:
    size_t i_layer;
    #pragma omp parallel for
    for (i_layer = 0; i_layer < wavefunction->parameters().size(); i_layer++){

        auto this_layer = wavefunction->parameters()[i_layer];


        // How many parameters in this layer?
        auto local_n_params = flat_shapes[i_layer];

        // This is the first grad outputs:
        auto v = torch::ones_like(psi, options);
        v.requires_grad_(true);

        // Loop over the individual weights in these parameters:
        for (int64_t i_weight = 0; i_weight < local_n_params; i_weight ++){
            // Construction of the second grad outputs:
            auto u = torch::zeros({local_n_params}, options);
            // Set one parameter non-zero:
            u.index_put_({i_weight}, 1.0);

            u = u.reshape(full_shapes[i_layer]);

            // First pass backwards:
            auto vjp = torch::autograd::grad(
                {psi},          // outputs
                {this_layer},   //inputs
                {v},            // grad_outputs
                true,           // retain graph
                true,           // create_graph
                false           // allow_unused
            );

            // Second backwards pass:
            auto output = torch::autograd::grad(
                {vjp},      // outputs
                {v},        //inputs
                {u},        // grad_outputs
                true,       // retain graph
                true,       // create_graph
                false       // allow_unused
            ).front();


            // PLOG_INFO << "Update " << output.sizes();
            // Put it into the jacobian:
            jacobian_flat.index_put_({Slice(), i_column}, output);
            i_column ++;
        }
    }

    return jacobian_flat;
}

torch::Tensor JacobianCalculator::jacobian_forward_weight(torch::Tensor psi,
    torch::Tensor x_current, ManyBodyWavefunction wavefunction, int64_t weight_index){


    // Need to interpolate to the correct layer of the wavefunction:
    size_t loop_layer = 0;
    int64_t layer_index = 0;

    int64_t running_index = 0;
    // Loop over the global index until it's found:
    for (size_t i_layer = 0; i_layer < flat_shapes.size(); i_layer ++){
        if (weight_index < running_index + flat_shapes[i_layer]){
            // This is the right layer
            loop_layer = i_layer;
            // And this will be the position within this layer:
            layer_index = weight_index - running_index;
            break;
        }
        else{
            running_index += flat_shapes[i_layer];
        }
    }



    // This is the first grad outputs:
    auto v = torch::ones_like(psi, options);
    v.requires_grad_(true);

    // Construction of the second grad outputs:
    // auto u = torch::ones({}, options);
    auto u = torch::zeros({flat_shapes[loop_layer]}, options);
    // Set one parameter non-zero:
    u.index_put_({layer_index}, 1.0);

    u = u.reshape(full_shapes[loop_layer]);

    // Get just this parameter:
    auto this_param = wavefunction->parameters()[loop_layer];

    // First pass backwards:
    auto vjp = torch::autograd::grad(
        {psi},          // outputs
        {this_param},   // inputs
        {v},            // grad_outputs
        true,           // retain graph
        true,           // create_graph
        false           // allow_unused
    );

    // Second backwards pass:
    auto output = torch::autograd::grad(
        {vjp},      // outputs
        {v},        // inputs
        {u},        // grad_outputs
        true,       // retain graph
        true,       // create_graph
        false       // allow_unused
    ).front();

    return output;

}


torch::Tensor JacobianCalculator::batch_jacobian_forward(
    torch::Tensor x_current, ManyBodyWavefunction wavefunction){

    auto n_walkers = x_current.sizes()[0];

    // First, make n copies of the wavefunction:

    size_t n_wavefunctions = wf_copies.size();
    #pragma omp parallel for
    for (size_t i = 0; i < n_wavefunctions; i ++){
        // Set the parameters:
        set_weights(wf_copies[i], wavefunction->parameters());

    }
    // Forward pass
    auto psi = wavefunction(x_current);


    // Compute just the first column:

    auto first_column = jacobian_forward_weight(psi, x_current, wavefunction, 0);

    PLOG_INFO << first_column[0];
    PLOG_INFO << first_column[1];

    auto jacobian_flat = torch::zeros({n_walkers, n_parameters}, options);

    // Loop over every parameter:
    std::vector<torch::Tensor> jacobian_columns;
    #pragma omp parallel for num_threads(concurrency)
    for (int64_t i_param = 0; i_param < n_parameters; i_param++){
        size_t wf_index = i_param % n_wavefunctions;
        // Forward pass
        auto psi = wf_copies[wf_index](x_current);

        jacobian_flat.index_put_(
            {Slice(), i_param},
            jacobian_forward_weight(psi, x_current, wf_copies[wf_index], i_param) );
    }


    //
    // // Next, chunk the inputs:
    // auto input_chunks = torch::chunk(x_current, concurrency);
    //
    //
    // // Prepare a list of jacobian chunks for output:
    // std::vector<torch::Tensor> jac_chunks;
    // jac_chunks.resize(concurrency);
    //
    // int64_t i_chunk;
    //
    // #pragma omp parallel for
    // for (i_chunk = 0; i_chunk < concurrency; i_chunk ++){
    //     jac_chunks[i_chunk] = jacobian_forward(input_chunks[i_chunk], wf_copies[i_chunk]);
    // }
    //
    //
    return jacobian_flat;

    // return torch::Tensor();
}

torch::Tensor JacobianCalculator::numerical_jacobian(
    torch::Tensor x_current, ManyBodyWavefunction wavefunction, float kick_size){

    // No need for gradients here
    c10::InferenceMode guard(true);


    // PLOG_INFO << "Enter forward";

    // How many walkers?
    auto n_walkers = x_current.sizes()[0];

    // // Compute the value of the wavefunction for all walkers:
    auto psi = wavefunction(x_current);

    auto jacobian_flat = torch::zeros({n_walkers, n_parameters}, options);


    int64_t i_column = 0;

    // Loop over the parameters:
    size_t i_layer;
    for (i_layer = 0; i_layer < wavefunction->parameters().size(); i_layer++){

        auto this_layer = wavefunction->parameters()[i_layer];


        // How many parameters in this layer?
        auto local_n_params = flat_shapes[i_layer];

        // Loop over the individual weights in these parameters:
        for (int64_t i_weight = 0; i_weight < local_n_params; i_weight ++){
            
            // Update the weight of this particular layer
            // Kick it up:
            wavefunction -> parameters()[i_layer].flatten()[i_weight] += kick_size;

            auto psi_up = wavefunction(x_current);

            wavefunction -> parameters()[i_layer].flatten()[i_weight] += kick_size;
            auto psi_up_up = wavefunction(x_current);

            // Now, kick it down (undo the first kick and add negative kick):
            wavefunction -> parameters()[i_layer].flatten()[i_weight] += -3.*kick_size;

            auto psi_down = wavefunction(x_current);
            wavefunction -> parameters()[i_layer].flatten()[i_weight] -= kick_size;
            auto psi_down_down = wavefunction(x_current);

            // Undo all kicks before the next iteration:
            wavefunction -> parameters()[i_layer].flatten()[i_weight] += 2*kick_size;


            auto this_jacobian_column = (
                    ( 1./12) * psi_down_down + 
                    (-2./3.) * psi_down +
                    ( 2./3.) * psi_up + 
                    (-1./12) * psi_up_up
                ) / (kick_size);

            // PLOG_INFO << "Update " << output.sizes();
            // Put it into the jacobian:
            jacobian_flat.index_put_({Slice(), i_column}, this_jacobian_column);
            i_column ++;
        }
    }

    return jacobian_flat;

}
