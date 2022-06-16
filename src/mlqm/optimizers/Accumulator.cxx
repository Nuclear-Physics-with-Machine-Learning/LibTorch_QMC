#include "Accumulator.h"

void Accumulator::accumulate(const std::string& key, torch::Tensor value){
    
    // No gradients of accumulated variables, set inference mode
    c10::InferenceMode guard(true);
    if(this -> find(key) == this -> end()){
        this ->operator[](key) = value;
    }
    else{
        this->operator[](key) += value;
    }
}

void Accumulator::finalize(torch::Tensor weight){
    
    // No gradients of accumulated variables, set inference mode
    c10::InferenceMode guard(true);

    map<std::string, torch::Tensor>::iterator it;
    for (it = this->begin(); it != this -> end(); it ++){
        if (it->first == "weight"){continue;}
        else{
            it -> second /= weight;
        }
    }

    return;

}

void Accumulator::allreduce(){
    ///TODO this
    return;
}

    // # @tf.function
    // def allreduce(self):

    //     for key in self.keys():
    //         self[key] = hvd.allreduce(self[key], op=hvd.Sum, device_dense="GPU")
    //     return

    // # def accumulate(self, weight=1, ** kwargs):
    // #     # energy, energy_jf, ke_jf, ke_direct, pe, acceptance,weight,r,dpsi_i,dpsi_i_EL,dpsi_ij,estim_wgt) :
    // #     for key in kwargs:
    // #         self.tensor_dict[key]      += kwargs[key] * weight
    // #         if key == "energy" or key == "energy_jf":
    // #             self.tensor_dict[key+"2"]  += (kwargs[key]* weight)**2

    // #     self.tensor_dict['weight'] += weight


    // def finalize(self, weight):

    //     for key in self.keys():
    //         if key == 'weight': continue
    //         self[key] /= weight

    //     return

    //         # error= tf.sqrt((self.tensor_dict["energy2"] - self.tensor_dict["energy"]**2) / (nav-1))
    //         # error_jf = tf.sqrt((self.tensor_dict["energy_jf2"] - self.tensor_dict["energy_jf"]**2) / (nav-1))
    //         # return error, error_jf
