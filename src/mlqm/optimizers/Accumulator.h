/**
 * @defgroup   ACCUMULATOR Accumulator
 *
 * @brief      This file implements Accumulator.
 *
 * @author     Corey.adams
 * @date       2022
 */

#pragma once

#include <string>
#include <map>
#include <torch/torch.h>


class Accumulator : public std::map<std::string, torch::Tensor>
{
public:
    Accumulator() {}
    ~Accumulator(){}
    
    void accumulate(const std::string& key, torch::Tensor value);

    void finalize(torch::Tensor weight);

    void allreduce();

private: 


};
