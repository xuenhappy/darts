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
#include <cstdlib>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include "../core/darts.hpp"
#include "../core/segment.hpp"
#include "./encoder.hpp"

namespace darts {

/**
 * @brief 利用ONNX神经网络对数据进行表示与
 *
 */
class OnnxPersenter : public CellPersenter {
   private:
    static const char* PMODEL_PATH_KEY;
    static const char* QMODEL_PATH_KEY;
    static const char* WORDPIECE_PARAM;
    static const char* TYPEENCODER_PARAM;

    Ort::Session* encode_model;
    Ort::Session* dist_model;
    std::shared_ptr<WordPice> wordpiece;
    std::shared_ptr<TypeEncoder> lencoder;

    Ort::Session* loadmodel(const char* model_path) {
        Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "darts");
        Ort::SessionOptions session_options;
        session_options.DisableMemPattern();
        session_options.DisableCpuMemArena();
        session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
        return new Ort::Session(env, model_path, session_options);
    }

    float* runEmbModel() {
        // print model input layer (node names, types, shape etc.)
        Ort::AllocatorWithDefaultOptions allocator;

        // print number of model input nodes
        size_t num_input_nodes                     = encode_model->GetInputCount();
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
        auto output_tensors = encode_model->Run(Ort::RunOptions{nullptr}, input_node_names.data(), ort_inputs.data(),
                                                ort_inputs.size(), output_node_names.data(), 2);

        // Get pointer to output tensor float values
        float* floatarr      = output_tensors[0].GetTensorMutableData<float>();
        float* floatarr_mask = output_tensors[1].GetTensorMutableData<float>();

        printf("Done!\n");
        return 0;
    }

   public:
    /**
     * @brief init this
     *
     * @param param
     * @return int
     */
    int initalize(const std::map<std::string, std::string>& params,
                  std::map<std::string, std::shared_ptr<SegmentPlugin>>& plugins) {
        auto it = plugins.find(WORDPIECE_PARAM);
        if (it == plugins.end()) {
            std::cerr << "no key find" << WORDPIECE_PARAM << std::endl;
            return EXIT_FAILURE;
        }
        this->wordpiece = std::dynamic_pointer_cast<WordPice>(it->second);
        if (this->wordpiece == nullptr) {
            std::cerr << "plugin init failed " << WORDPIECE_PARAM << std::endl;
            return EXIT_FAILURE;
        }

        it = plugins.find(TYPEENCODER_PARAM);
        if (it == plugins.end()) {
            std::cerr << "no key find" << TYPEENCODER_PARAM << std::endl;
            return EXIT_FAILURE;
        }
        this->lencoder = std::dynamic_pointer_cast<TypeEncoder>(it->second);
        if (this->lencoder == nullptr) {
            std::cerr << "plugin init failed " << TYPEENCODER_PARAM << std::endl;
            return EXIT_FAILURE;
        }
        // load model
        auto pit = params.find(PMODEL_PATH_KEY);
        if (pit == params.end()) {
            std::cerr << "no param key find" << PMODEL_PATH_KEY << std::endl;
            return EXIT_FAILURE;
        }
        this->encode_model = loadmodel(pit->second.c_str());
        if (!this->encode_model) {
            std::cerr << "load model " << PMODEL_PATH_KEY << " from path " << it->second << " failed!" << std::endl;
            return EXIT_FAILURE;
        }

        pit = params.find(QMODEL_PATH_KEY);
        if (pit == params.end()) {
            std::cerr << "no param key find" << QMODEL_PATH_KEY << std::endl;
            return EXIT_FAILURE;
        }
        this->dist_model = loadmodel(pit->second.c_str());
        if (!this->dist_model) {
            std::cerr << "load model " << QMODEL_PATH_KEY << " from path " << it->second << " failed!" << std::endl;
            return EXIT_FAILURE;
        }
        return EXIT_SUCCESS;
    }
    /**
     * @brief set all word embeding
     *
     * @param dstSrc
     * @param cmap
     */
    void embed(const AtomList& dstSrc, SegPath& cmap) const {}
    /**
     * @brief
     *
     * @param pre
     * @param next
     * @return double must >=0
     */
    double ranging(const Word* pre, const Word* next) const { return 0.0; }

    ~OnnxPersenter() {
        if (this->encode_model) {
            delete this->encode_model;
            this->encode_model = NULL;
        }
        if (this->dist_model) {
            delete this->dist_model;
            this->dist_model = NULL;
        }
        wordpiece = nullptr;
        lencoder  = nullptr;
    }
};

const char* OnnxPersenter::PMODEL_PATH_KEY   = "encoder.model";
const char* OnnxPersenter::QMODEL_PATH_KEY   = "quantizer.model";
const char* OnnxPersenter::WORDPIECE_PARAM   = "dict.wordpiece";
const char* OnnxPersenter::TYPEENCODER_PARAM = "dict.type.encoder";

REGISTER_Persenter(OnnxPersenter);

}  // namespace darts
#endif  // SRC_IMPL_NETWORDQI_HPP_
