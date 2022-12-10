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
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include "../core/darts.hpp"
#include "../core/segment.hpp"
#include "../utils/str_utils.hpp"
#include "./encoder.hpp"

namespace darts {

inline Ort::Session* loadmodel(const char* model_path) {
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "darts");
    Ort::SessionOptions session_options;
    session_options.DisableMemPattern();
    session_options.DisableCpuMemArena();
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
    return new Ort::Session(env, model_path, session_options);
}
/**
 * @brief a network onnx
 *
 */
class OnnxIndicator : public Indicator {
   private:
    static const char* MODEL_PATH_KEY;
    static const char* WORDPIECE_PARAM;
    static const char* TYPEENCODER_PARAM;

    size_t emdim;
    std::vector<const char*> input_node_names;
    std::vector<const char*> output_node_names;

    Ort::Session* session;
    std::shared_ptr<WordPice> wordpiece;
    std::shared_ptr<TypeEncoder> lencoder;

    int set_encode_dim(Ort::Session* session) {
        // print model input layer (node names, types, shape etc.)
        Ort::AllocatorWithDefaultOptions allocator;
        size_t num_input_nodes = session->GetInputCount();
        if (num_input_nodes != 2) {
            std::cerr << "this model input is not 1" << std::endl;
            return EXIT_FAILURE;
        }

        return EXIT_SUCCESS;
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
        if (it != plugins.end()) {
            this->lencoder = std::dynamic_pointer_cast<TypeEncoder>(it->second);
            if (this->lencoder == nullptr) {
                std::cerr << "plugin init failed " << TYPEENCODER_PARAM << std::endl;
                return EXIT_FAILURE;
            }
        }
        // load model
        auto pit = params.find(MODEL_PATH_KEY);
        if (pit == params.end()) {
            std::cerr << "no param key find" << MODEL_PATH_KEY << std::endl;
            return EXIT_FAILURE;
        }
        this->session = loadmodel(pit->second.c_str());
        if (!this->session) {
            std::cerr << "load model " << MODEL_PATH_KEY << " from path " << it->second << " failed!" << std::endl;
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
    void embed(const AtomList& dstSrc, SegPath& cmap) const {
        std::vector<const char*> input_node_names  = {"input"};
        std::vector<const char*> output_node_names = {"output"};

        std::vector<int64_t> input_node_dims = {1, 10, 2};
        size_t input_tensor_size             = 1 * 10 * 2;

        std::vector<float> input_tensor_values(input_tensor_size);
        for (unsigned int i = 0; i < input_tensor_size; i++)
            input_tensor_values[i] = (float)i / (input_tensor_size + 1);
        // create input tensor object from data values
        auto memory_info        = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info, input_tensor_values.data(),
                                                                  input_tensor_size, input_node_dims.data(), 3);
        assert(input_tensor.IsTensor());

        std::vector<Ort::Value> ort_inputs;
        ort_inputs.push_back(std::move(input_tensor));
        // score model & input tensor, get back output tensor
        std::vector<Ort::Value> output_tensors;
        session->Run(Ort::RunOptions{nullptr}, input_node_names.data(), ort_inputs.data(), ort_inputs.size(),
                     output_node_names.data(), output_tensors.data(), 2);

        // Get pointer to output tensor float values
        float* floatarr = output_tensors[0].GetTensorMutableData<float>();
        // set cmap att
    }

    ~OnnxIndicator() {
        if (this->session) {
            delete this->session;
            this->session = NULL;
        }
        input_node_names.clear();
        output_node_names.clear();
        wordpiece = nullptr;
        lencoder  = nullptr;
    }
};

class OnnxQuantizer : public Quantizer {
   private:
    static const char* MODEL_PATH_KEY;
    size_t emdim;
    std::vector<const char*> input_node_names;
    std::vector<const char*> output_node_names;
    Ort::Session* seesion;

    int set_encode_dim(Ort::Session* session) {
        // print model input layer (node names, types, shape etc.)
        Ort::AllocatorWithDefaultOptions allocator;
        size_t num_input_nodes = session->GetInputCount();
        if (num_input_nodes != 2) {
            std::cerr << "this model input is not 1" << std::endl;
            return EXIT_FAILURE;
        }
        return EXIT_SUCCESS;
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
        auto pit = params.find(MODEL_PATH_KEY);
        if (pit == params.end()) {
            std::cerr << "no param key find" << MODEL_PATH_KEY << std::endl;
            return EXIT_FAILURE;
        }
        this->seesion = loadmodel(pit->second.c_str());
        if (!this->seesion) {
            std::cerr << "load model " << MODEL_PATH_KEY << " from path " << pit->second << " failed!" << std::endl;
            return EXIT_FAILURE;
        }
        return EXIT_SUCCESS;
    }

    /**
     * @brief
     *
     * @param pre
     * @param next
     * @return double must >=0
     */
    double ranging(const std::shared_ptr<Word> pre, const std::shared_ptr<Word> next) const {
        if (pre == nullptr || next == nullptr) return 0.0;
        if (pre->getAtt() == nullptr || next->getAtt() == nullptr) {
            return 0.0;
        }
        // TODO

        return 0.0;
    }

    ~OnnxQuantizer() {
        if (this->seesion) {
            delete this->seesion;
            this->seesion = NULL;
        }
    }
};

/**
 * @brief
 *
 */
class OnnxDecider : public Decider {
   private:
    static const char* PMODEL_PARAM;
    static const char* QMODEL_PARAM;

    std::shared_ptr<Quantizer> quantizer;
    std::shared_ptr<Indicator> indicator;

   public:
    /**
     * @brief init this
     *
     * @param param
     * @return int
     */
    int initalize(const std::map<std::string, std::string>& params,
                  std::map<std::string, std::shared_ptr<SegmentPlugin>>& plugins) {
        // load model
        auto it = plugins.find(PMODEL_PARAM);
        if (it == plugins.end()) {
            std::cerr << "no key find" << PMODEL_PARAM << std::endl;
            return EXIT_FAILURE;
        }
        this->quantizer = std::dynamic_pointer_cast<Quantizer>(it->second);
        if (this->quantizer == nullptr) {
            std::cerr << "plugin init failed " << PMODEL_PARAM << std::endl;
            return EXIT_FAILURE;
        }
        it = plugins.find(QMODEL_PARAM);
        if (it == plugins.end()) {
            std::cerr << "no key find" << QMODEL_PARAM << std::endl;
            return EXIT_FAILURE;
        }
        this->indicator = std::dynamic_pointer_cast<Indicator>(it->second);
        if (this->indicator == nullptr) {
            std::cerr << "plugin init failed " << QMODEL_PARAM << std::endl;
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
    void embed(const AtomList& dstSrc, SegPath& cmap) const { indicator->embed(dstSrc, cmap); }
    /**
     * @brief
     *
     * @param pre
     * @param next
     * @return double must >=0
     */
    double ranging(const std::shared_ptr<Word> pre, const std::shared_ptr<Word> next) const {
        return quantizer->ranging(pre, next);
    }

    ~OnnxDecider() {
        quantizer = nullptr;
        indicator = nullptr;
    }
};

const char* OnnxDecider::PMODEL_PARAM = "encoder.model";
const char* OnnxDecider::QMODEL_PARAM = "quantizer.model";

REGISTER_Persenter(OnnxDecider);

/**
 * @brief a cell recongnizer that use hmm
 *
 */
class OnnxRecongnizer : public CellRecognizer {
   private:
    static const char* DATA_PATH_KEY;
    std::vector<std::string> labels;
    Ort::Session* seesion;

    int set_encode_dim(Ort::Session* session) {
        // print model input layer (node names, types, shape etc.)
        Ort::AllocatorWithDefaultOptions allocator;
        size_t num_input_nodes = session->GetInputCount();
        if (num_input_nodes != 2) {
            std::cerr << "this model input is not 1" << std::endl;
            return EXIT_FAILURE;
        }
        return EXIT_SUCCESS;
    }

   private:
    /**
     * @brief Get the feateure embeding object
     *
     * @param dstSrc
     * @param props
     */
    void decode(const AtomList& dstSrc, std::vector<size_t>& seq) const {}

   public:
    int initalize(const std::map<std::string, std::string>& params,
                  std::map<std::string, std::shared_ptr<SegmentPlugin>>& plugins) {
        auto iter = params.find(DATA_PATH_KEY);
        if (iter == params.end()) {
            std::cerr << DATA_PATH_KEY << " key not found in dictionary!" << std::endl;
            return EXIT_FAILURE;
        }
        split(iter->second, ",", labels);
        // check labels_size and dim size
        return EXIT_SUCCESS;
    }

    void addSomeCells(const AtomList& dstSrc, SegPath& cmap) const {
        std::vector<size_t> label_idx;
        decode(dstSrc, label_idx);
        Cursor cur = cmap.Head();
        size_t pos = 1;
        for (size_t i = 1; i < label_idx.size() - 1; ++i) {
            const std::string& nlabel  = labels[label_idx[i]];
            const std::string& nxlabel = labels[label_idx[i + 1]];
            if (nlabel == nxlabel) continue;
            if (nlabel[0] == 'I') {
                auto w = std::make_shared<Word>(dstSrc, pos, i + 1);
                if (nlabel.size() > 2) w->addLabel(nlabel.substr(2));
                cur = cmap.addNext(cur, w);
                pos = i + 1;
                continue;
            }
            if (nlabel.size() > 2 && nlabel.substr(1) == nxlabel.substr(1)) continue;
            auto w = std::make_shared<Word>(dstSrc, pos, i + 1);
            if (nlabel.size() > 2) w->addLabel(nlabel.substr(2));
            cur = cmap.addNext(cur, w);
            pos = i + 1;
        }
        if (pos < label_idx.size() - 1) {
            auto w = std::make_shared<Word>(dstSrc, pos, label_idx.size());

            const std::string& nlabel = labels[label_idx[pos]];
            if (nlabel.size() > 2) w->addLabel(nlabel.substr(2));
            cmap.addNext(cur, w);
        }
    }
};

const char* OnnxRecongnizer::DATA_PATH_KEY = "data.path";
REGISTER_Recognizer(OnnxRecongnizer);

}  // namespace darts
#endif  // SRC_IMPL_NETWORDQI_HPP_
