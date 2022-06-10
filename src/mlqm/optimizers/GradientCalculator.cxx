#include "GradientCalculator.h"

torch::Tensor GradientCalculator::f_i(
    torch::Tensor dpsi_i, torch::Tensor energy, torch::Tensor dpsi_i_EL){
    // return (dpsi_i * energy - dpsi_i_EL)
    return dpsi_i * energy - dpsi_i_EL;
}
torch::Tensor GradientCalculator::S_ij(
    torch::Tensor dpsi_ij, torch::Tensor dpsi_i){
    // return dpsi_ij - dpsi_i * tf.transpose(dpsi_i)    
    return dpsi_ij -  dpsi_i * torch::transpose(dpsi_i, 0, 1); 
}
torch::Tensor GradientCalculator::par_dist(
    torch::Tensor dp_i, torch::Tensor S_ij){
    // dist = tf.reduce_sum(dp_i*tf.linalg.matmul(S_ij, dp_i))
    // return dist
    return torch::sum(dp_i * torch::matmul(S_ij, dp_i));
}
torch::Tensor GradientCalculator::regularize_S_ij(
    torch::Tensor S_ij, torch::Tensor eps){
    // dtype = S_ij.dtype
    // npt   = S_ij.shape[0]
    // S_ij_d = S_ij + eps * tf.eye(npt, dtype=dtype)
    // return S_ij_d
    auto npt = S_ij.sizes()[0];
    return S_ij + eps * torch::eye(npt, options);
}
torch::Tensor GradientCalculator::pd_solve(
    torch::Tensor S_ij, torch::Tensor eps, torch::Tensor f_i){
    // # Regularize along the diagonal:
    // S_ij_d = self.regularize_S_ij(S_ij, eps)

    // # Next, we need S_ij to be positive definite.
    // U_ij = tf.linalg.cholesky(S_ij_d)

    // dp_i = tf.linalg.cholesky_solve(U_ij, f_i)

    // return dp_i

    // Regularize along the diagonal:
    auto S_ij_d = regularize_S_ij(S_ij, eps);

    // Next, we need S_ij to be positive definite.
    auto U_ij = torch::linalg::cholesky(S_ij_d);

    // Solve the equation
    auto dp_i = torch::linalg::solve_triangular(
        U_ij, // input
        f_i, // other
        false, // upper
        true, // left
        true // unitriangular
    );
    return dp_i;
}



        
