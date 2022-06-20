#include "GradientCalculator.h"

#include <torch/linalg.h>

torch::Tensor GradientCalculator::f_i(
    torch::Tensor dpsi_i, torch::Tensor energy, torch::Tensor dpsi_i_EL){
    // return (dpsi_i * energy - dpsi_i_EL)
    // PLOG_INFO << "dpsi_i.sizes(): " << dpsi_i.sizes();
    // PLOG_INFO << "energy.sizes(): " << energy.sizes();
    // PLOG_INFO << "dpsi_i_EL.sizes(): " << dpsi_i_EL.sizes();
    return dpsi_i * energy - dpsi_i_EL;
}
torch::Tensor GradientCalculator::S_ij(
    torch::Tensor dpsi_ij, torch::Tensor dpsi_i){
    // return dpsi_ij - dpsi_i * tf.transpose(dpsi_i)
    // PLOG_INFO << "dpsi_ij.sizes(): " << dpsi_ij.sizes();
    // PLOG_INFO << "dpsi_i.sizes(): " << dpsi_i.sizes();
    auto dpsi_i_extended = torch::reshape(dpsi_i, {-1, 1});
    // PLOG_INFO << "dpsi_i_extended.sizes(): " << dpsi_i_extended.sizes();
    // PLOG_INFO << "torch::transpose(dpsi_i_extended, 0, 1).sizes(): " << torch::transpose(dpsi_i_extended, 0, 1).sizes();
    return dpsi_ij -  dpsi_i_extended * torch::transpose(dpsi_i_extended, 0, 1);
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
    // PLOG_INFO << "f_i.sizes(): " << f_i.sizes();
    // PLOG_INFO << "S_ij.sizes(): " << S_ij.sizes();

    // Regularize along the diagonal:
    auto S_ij_d = regularize_S_ij(S_ij, eps);
    // PLOG_INFO << "S_ij_d.sizes(): " << S_ij_d.sizes();

    // Next, we need S_ij to be positive definite.
    auto U_ij = torch::linalg::cholesky(S_ij_d);
    // PLOG_INFO << "U_ij.sizes(): " << U_ij.sizes();


    // The solve_triangular function expects to be batching
    // matrix solves, so have to feed in batch size 1:

    auto n_params = U_ij.sizes()[0];



    /*
     * U_ij needs to be shape nxn, and triangular
     * f_i needs to be shape nxk, here we take k == 1
     * that produces and output tensor of shape nxk, again k = 1
     */

    f_i  = torch::reshape(f_i,  {n_params, 1});
    // U_ij = torch::reshape(U_ij, {n_params, n_params});
    // PLOG_INFO << "f_i.sizes(): " << f_i.sizes();
    // PLOG_INFO << "U_ij.sizes(): " << U_ij.sizes();

    // Solve the equation
    auto dp_i = torch::linalg::solve_triangular(
        U_ij, // input
        f_i, // other
        false, // upper
        true, // left
        false // unitriangular
    );

    /*
     *If upper= True (resp. False) just the upper (resp. lower)
     *triangular half of A will be accessed. The elements below
     *the main diagonal will be considered to be zero and
     *will not be accessed.
     *
     *If unitriangular= True, the diagonal of A is assumed to be
     *ones and will not be accessed.
     *
     *The result may contain NaN s if the diagonal of A contains
     *zeros or elements that are very close to zero and
     *unitriangular= False (default) or if the input matrix has
     *very small eigenvalues.
     *
     *Supports inputs of float, double, cfloat and cdouble dtypes.
     *Also supports batches of matrices, and if the inputs are
     *batches of matrices then the output has the same batch
     *dimensions.
    */

    return dp_i;
}
