/*
 * File: networdqi.hpp
 * Project: impl
 * File Created: Sunday, 2nd January 2022 10:06:07 am
 * Author: Xu En (xuen@mokar.com)
 * -----
 * Last Modified: Sunday, 2nd January 2022 10:06:10 am
 * Modified By: Xu En (xuen@mokahr.com)
 * -----
 * Copyright 2021 - 2022 Your Company, Moka
 */
#ifndef SRC_IMPL_NETWORDQI_HPP_
#define SRC_IMPL_NETWORDQI_HPP_
#include <onnxruntime_cxx_api.h>
#include <map>
#include <string>
#include <vector>
#include "../core/darts.hpp"
#include "../core/segment.hpp"

namespace darts {

/**
 * @brief 利用ONNX神经网络对数据进行表示与
 *
 */
class OnnxPersenter : public CellPersenter {
   private:
    static const char* PMODEL_PATH_KEY;
    static const char* QMODEL_PATH_KEY;

   public:
    /**
     * @brief init this
     *
     * @param param
     * @return int
     */
    int initalize(const std::map<std::string, std::string>& param) {
        Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "darts");
        Ort::SessionOptions session_options;
        session_options.DisableMemPattern();
        session_options.DisableCpuMemArena();
        session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

        Ort::Session session(env, model_path, session_options);
        // print model input layer (node names, types, shape etc.)
        Ort::AllocatorWithDefaultOptions allocator;

        // print number of model input nodes
        size_t num_input_nodes                     = session.GetInputCount();
        std::vector<const char*> input_node_names  = {"input", "input_mask"};
        std::vector<const char*> output_node_names = {"output", "output_mask"};

        std::vector<int64_t> input_node_dims = {10, 20};
        size_t input_tensor_size             = 10 * 20;
        std::vector<float> input_tensor_values(input_tensor_size);
        for (unsigned int i = 0; i < input_tensor_size; i++)
            input_tensor_values[i] = (float)i / (input_tensor_size + 1);
        // create input tensor object from data values
        auto memory_info        = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info, input_tensor_values.data(),
                                                                  input_tensor_size, input_node_dims.data(), 2);
        assert(input_tensor.IsTensor());

        std::vector<int64_t> input_mask_node_dims = {1, 20, 4};
        size_t input_mask_tensor_size             = 1 * 20 * 4;
        std::vector<float> input_mask_tensor_values(input_mask_tensor_size);
        for (unsigned int i = 0; i < input_mask_tensor_size; i++)
            input_mask_tensor_values[i] = (float)i / (input_mask_tensor_size + 1);
        // create input tensor object from data values
        auto mask_memory_info        = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        Ort::Value input_mask_tensor = Ort::Value::CreateTensor<float>(
            mask_memory_info, input_mask_tensor_values.data(), input_mask_tensor_size, input_mask_node_dims.data(), 3);
        assert(input_mask_tensor.IsTensor());

        std::vector<Ort::Value> ort_inputs;
        ort_inputs.push_back(std::move(input_tensor));
        ort_inputs.push_back(std::move(input_mask_tensor));
        // score model & input tensor, get back output tensor
        auto output_tensors = session.Run(Ort::RunOptions{nullptr}, input_node_names.data(), ort_inputs.data(),
                                          ort_inputs.size(), output_node_names.data(), 2);

        // Get pointer to output tensor float values
        float* floatarr      = output_tensors[0].GetTensorMutableData<float>();
        float* floatarr_mask = output_tensors[1].GetTensorMutableData<float>();

        printf("Done!\n");
        return 0;

        return EXIT_SUCCESS;
    }
    /**
     * @brief set all word embeding
     *
     * @param dstSrc
     * @param cmap
     */
    void embed(AtomList* dstSrc, SegPath* cmap) const {}
    /**
     * @brief
     *
     * @param pre
     * @param next
     * @return double must >=0
     */
    double ranging(const Word* pre, const Word* next) const { return 0.0; }
    ~OnnxPersenter() {}
};

const char* OnnxPersenter::PMODEL_PATH_KEY = "pmodel.path";
const char* OnnxPersenter::QMODEL_PATH_KEY = "qmodel.path";
REGISTER_Persenter(OnnxPersenter);

}  // namespace darts
#endif  // SRC_IMPL_NETWORDQI_HPP_
