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
#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>
#include "../core/segment.hpp"
#include "../utils/filetool.hpp"
#include "../utils/strtool.hpp"
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
class OnnxIndicator {
   private:
    static const char* MODEL_PATH_KEY;
    static const char* WORDPIECE_PARAM;
    static const char* TYPEENCODER_PARAM;

    size_t emdim;
    Ort::Session* session;
    std::vector<const char*> input_name_;
    std::vector<const char*> output_name_;

    std::shared_ptr<WordPice> wordpiece;
    std::shared_ptr<TypeEncoder> lencoder;

    int validator(Ort::Session* session) {
        // check input tensor nums
        size_t input_count = session->GetInputCount();
        if (input_count != 2) {
            std::cerr << "This model is not supported by onnx indicator.  input nums must 2" << std::endl;
            return EXIT_FAILURE;
        }
        // check atomlist input tensor
        auto alist_tensor_info = session->GetInputTypeInfo(0).GetTensorTypeAndShapeInfo();
        size_t alist_dim_count = alist_tensor_info.GetDimensionsCount();
        if (alist_dim_count != 1) {  // timestep
            std::cerr << "This model is not supported by onnx indicator. alist dim is not 1!" << std::endl;
            return EXIT_FAILURE;
        }
        // check word input tensor
        auto word_tensor_info = session->GetInputTypeInfo(1).GetTensorTypeAndShapeInfo();
        size_t word_dim_count = word_tensor_info.GetDimensionsCount();
        if (word_dim_count != 2) {  // nums*idx
            std::cerr << "This model is not supported by onnx indicator. word dim is not 2!" << std::endl;
            return EXIT_FAILURE;
        }
        std::vector<int64_t> word_dims = word_tensor_info.GetShape();
        if (word_dims[1] != 3) {
            std::cerr << "This model is not supported by onnx indicator. word need be in NC format" << std::endl;
            return EXIT_FAILURE;
        }
        // check output tensor nums
        size_t out_count = session->GetOutputCount();
        if (out_count != 1) {
            std::cerr << "This model is not supported by onnx indicator. output nums must 1" << std::endl;
            return EXIT_FAILURE;
        }
        // check output tensor
        auto emb_tensor_info = session->GetOutputTypeInfo(0).GetTensorTypeAndShapeInfo();
        size_t emb_dim_count = emb_tensor_info.GetDimensionsCount();
        if (emb_dim_count != 2) {  // nums*dim
            std::cerr << "This model is not supported by onnx indicator. output dim is not 2!" << std::endl;
            return EXIT_FAILURE;
        }
        auto emb_dims = emb_tensor_info.GetShape();
        this->emdim   = emb_dims[1];

        Ort::AllocatorWithDefaultOptions ort_alloc;
        input_name_.emplace_back(session->GetInputNameAllocated(0, ort_alloc).get());
        input_name_.emplace_back(session->GetInputNameAllocated(1, ort_alloc).get());
        output_name_.emplace_back(session->GetOutputNameAllocated(0, ort_alloc).get());

        return EXIT_SUCCESS;
    }

   public:
    size_t getEmbSize() { return emdim; }
    /**
     * @brief init this
     *
     * @param param
     * @return int
     */
    int load(const std::map<std::string, std::string>& params,
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
        std::string modelpath = getResource(pit->second);
        this->session         = loadmodel(modelpath.c_str());
        if (!this->session) {
            std::cerr << "load model " << MODEL_PATH_KEY << " from path " << modelpath << " failed!" << std::endl;
            return EXIT_FAILURE;
        }
        if (validator(this->session)) {
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
        // create inputs
        std::vector<Ort::Value> ort_inputs;
        auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        // create alist tensor
        std::vector<int64_t> alist_tensor_values;
        alist_tensor_values.reserve(dstSrc.size() * 2 + 2);
        std::vector<size_t> start_(dstSrc.size(), 1);
        std::vector<size_t> ends_(dstSrc.size(), 1);
        // set alist data
        wordpiece->encode(dstSrc, [&](int code, int atom_postion) {
            if (atom_postion >= 0) {
                start_[atom_postion] = std::min(start_[atom_postion], alist_tensor_values.size());
                ends_[atom_postion]  = std::max(ends_[atom_postion], alist_tensor_values.size());
            }
            alist_tensor_values.push_back(code);
        });
        int64_t adim = alist_tensor_values.size();
        Ort::Value alist_input_tensor =
            Ort::Value::CreateTensor<int64_t>(memory_info, alist_tensor_values.data(), adim, &adim, 1);
        assert(alist_input_tensor.IsTensor());
        ort_inputs.push_back(std::move(alist_input_tensor));
        // create words tensor
        int64_t words_size              = cmap.Size() + 2;
        size_t words_tensor_size        = words_size * 3;
        std::vector<int64_t> words_dims = {words_size, 3};
        std::vector<int64_t> words_tensor_values(words_tensor_size);
        // set head
        words_tensor_values[0] = 0;
        words_tensor_values[1] = 0;
        words_tensor_values[2] = codemap::sep_code;
        // set tail
        words_tensor_values[words_tensor_size - 3] = adim - 1;
        words_tensor_values[words_tensor_size - 2] = adim - 1;
        words_tensor_values[words_tensor_size - 1] = codemap::cls_code;
        // set common
        cmap.iterRow(nullptr, -1, [this, &words_tensor_values, &start_, &ends_](Cursor cur) {
            auto w     = cur->val;
            size_t idx = (cur->idx + 1) * 3;

            words_tensor_values[idx]     = start_[w->st];
            words_tensor_values[idx + 1] = ends_[w->et - 1];
            words_tensor_values[idx + 2] = this->lencoder == nullptr ? codemap::unk_code : this->lencoder->encode(w);
        });
        Ort::Value words_input_tensor = Ort::Value::CreateTensor<int64_t>(
            memory_info, words_tensor_values.data(), words_tensor_size, words_dims.data(), words_dims.size());
        assert(words_input_tensor.IsTensor());
        ort_inputs.push_back(std::move(words_input_tensor));
        // run model
        Ort::Value output_tensor{nullptr};
        session->Run(Ort::RunOptions{nullptr}, input_name_.data(), ort_inputs.data(), ort_inputs.size(),
                     output_name_.data(), &output_tensor, output_name_.size());

        // set cmap data
        const float* floatarr = output_tensor.GetTensorMutableData<float>();
        cmap.SrcNode()->setAtt(std::shared_ptr<std::vector<float>>(new std::vector<float>(floatarr, floatarr + emdim)));
        cmap.iterRow(nullptr, -1, [this, floatarr](Cursor cur) {
            const float* arr = floatarr + (cur->idx + 1) * emdim;
            cur->val->setAtt(std::shared_ptr<std::vector<float>>(new std::vector<float>(arr, arr + emdim)));
        });
        const float* endarr = floatarr + (words_size - 1) * emdim;
        cmap.EndNode()->setAtt(std::shared_ptr<std::vector<float>>(new std::vector<float>(endarr, endarr + emdim)));
    }

    ~OnnxIndicator() {
        if (this->session) {
            delete this->session;
            this->session = nullptr;
        }
        input_name_.clear();
        output_name_.clear();
        wordpiece = nullptr;
        lencoder  = nullptr;
    }
};
const char* OnnxIndicator::MODEL_PATH_KEY    = "model.path";
const char* OnnxIndicator::WORDPIECE_PARAM   = "wordpiece.name";
const char* OnnxIndicator::TYPEENCODER_PARAM = "tencode.name";

class OnnxQuantizer {
   private:
    static const char* MODEL_PATH_KEY;

    size_t emdim;
    Ort::Session* session;
    std::vector<const char*> input_name_;
    std::vector<const char*> output_name_;

    int validator(Ort::Session* session) {
        // check input tensor nums
        size_t input_count = session->GetInputCount();
        if (input_count != 2) {
            std::cerr << "This model is not supported by onnx quantizer.  input nums must 2" << std::endl;
            return EXIT_FAILURE;
        }
        // check a1 input tensor
        auto a1_tensor_info = session->GetInputTypeInfo(0).GetTensorTypeAndShapeInfo();
        size_t a1_dim_count = a1_tensor_info.GetDimensionsCount();
        if (a1_dim_count != 1) {  // channel
            std::cerr << "This model is not supported by onnx quantizer. a1 dim is not 1!" << std::endl;
            return EXIT_FAILURE;
        }
        std::vector<int64_t> alist_dims = a1_tensor_info.GetShape();
        if (alist_dims[0] != emdim) {
            std::cerr << "This model is not supported by onnx quantizer. a1 need be in C format" << std::endl;
            return EXIT_FAILURE;
        }
        // check a2 input tensor
        auto a2_tensor_info = session->GetInputTypeInfo(1).GetTensorTypeAndShapeInfo();
        size_t a2_dim_count = a2_tensor_info.GetDimensionsCount();
        if (a2_dim_count != 1) {  // channel
            std::cerr << "This model is not supported by onnx quantizer. a2 dim is not 1!" << std::endl;
            return EXIT_FAILURE;
        }
        std::vector<int64_t> a2_dims = a2_tensor_info.GetShape();
        if (a2_dims[0] != emdim) {
            std::cerr << "This model is not supported by onnx quantizer. a2 need be in C format" << std::endl;
            return EXIT_FAILURE;
        }
        // check output tensor nums
        size_t out_count = session->GetOutputCount();
        if (out_count != 1) {
            std::cerr << "This model is not supported by onnx quantizer. output nums must 1" << std::endl;
            return EXIT_FAILURE;
        }
        // check output tensor
        auto out_tensor_info = session->GetOutputTypeInfo(0).GetTensorTypeAndShapeInfo();
        size_t out_dim_count = out_tensor_info.GetDimensionsCount();
        if (out_dim_count != 1) {  // nums*dim
            std::cerr << "This model is not supported by onnx quantizer. output dim is not 2!" << std::endl;
            return EXIT_FAILURE;
        }
        auto out_dims = out_tensor_info.GetShape();
        if (out_dims[0] != 1) {
            std::cerr << "This model is not supported by onnx quantizer. output dim is not 2!" << std::endl;
            return EXIT_FAILURE;
        }

        Ort::AllocatorWithDefaultOptions ort_alloc;
        input_name_.emplace_back(session->GetInputNameAllocated(0, ort_alloc).get());
        input_name_.emplace_back(session->GetInputNameAllocated(1, ort_alloc).get());
        output_name_.emplace_back(session->GetOutputNameAllocated(0, ort_alloc).get());

        return EXIT_SUCCESS;
    }

   public:
    /**
     * @brief init this
     *
     * @param param
     * @return int
     */
    int load(const std::map<std::string, std::string>& params, size_t edim) {
        this->emdim = edim;
        auto pit    = params.find(MODEL_PATH_KEY);
        if (pit == params.end()) {
            std::cerr << "no param key find" << MODEL_PATH_KEY << std::endl;
            return EXIT_FAILURE;
        }
        std::string modelpath = getResource(pit->second);
        this->session         = loadmodel(modelpath.c_str());
        if (!this->session) {
            std::cerr << "load model " << MODEL_PATH_KEY << " from path " << modelpath << " failed!" << std::endl;
            return EXIT_FAILURE;
        }
        if (validator(this->session)) {
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
        // create inputs
        std::vector<Ort::Value> ort_inputs;
        auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        int64_t adim     = static_cast<int64_t>(this->emdim);

        // set data
        auto x1              = pre->getAtt();
        Ort::Value a1_tensor = Ort::Value::CreateTensor<float>(memory_info, x1->data(), this->emdim, &adim, 1);
        assert(a1_tensor.IsTensor());
        ort_inputs.push_back(std::move(a1_tensor));

        auto x2              = next->getAtt();
        Ort::Value a2_tensor = Ort::Value::CreateTensor<float>(memory_info, x2->data(), this->emdim, &adim, 1);
        assert(a2_tensor.IsTensor());
        ort_inputs.push_back(std::move(a2_tensor));
        // run model
        Ort::Value output_tensor{nullptr};
        session->Run(Ort::RunOptions{nullptr}, input_name_.data(), ort_inputs.data(), ort_inputs.size(),
                     output_name_.data(), &output_tensor, output_name_.size());
        // Get pointer to output tensor float values
        return output_tensor.GetTensorMutableData<float>()[0];
    }

    ~OnnxQuantizer() {
        if (this->session) {
            delete this->session;
            this->session = nullptr;
        }
        input_name_.clear();
        output_name_.clear();
    }
};
const char* OnnxQuantizer::MODEL_PATH_KEY = "model.path";

/**
 * @brief
 *
 */
class OnnxDecider : public Decider {
   private:
    OnnxQuantizer quantizer;
    OnnxIndicator indicator;

   public:
    /**
     * @brief init this
     *
     * @param param
     * @return int
     */
    int initalize(const std::map<std::string, std::string>& params,
                  std::map<std::string, std::shared_ptr<SegmentPlugin>>& plugins) {
        if (indicator.load(params, plugins)) {
            return EXIT_FAILURE;
        }
        if (quantizer.load(params, indicator.getEmbSize())) {
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
    void embed(const AtomList& dstSrc, SegPath& cmap) const { indicator.embed(dstSrc, cmap); }
    /**
     * @brief
     *
     * @param pre
     * @param next
     * @return double must >=0
     */
    double ranging(const std::shared_ptr<Word> pre, const std::shared_ptr<Word> next) const {
        return quantizer.ranging(pre, next);
    }

    ~OnnxDecider() {}
};

REGISTER_Persenter(OnnxDecider);

/**
 * @brief a cell recongnizer that use hmm
 *
 */
class OnnxRecongnizer : public CellRecognizer {
   private:
    static const char* MODEL_PATH_KEY;
    static const char* WORDPIECE_PARAM;
    static const char* LABELS_KEY;

    std::vector<std::string> labels;
    std::shared_ptr<WordPice> wordpiece;

    Ort::Session* session;
    std::vector<const char*> input_name_;
    std::vector<const char*> output_name_;

    int validator(Ort::Session* session) {
        // check input tensor nums
        size_t input_count = session->GetInputCount();
        if (input_count != 2) {
            std::cerr << "This model is not supported by onnx recongnizer.  input nums must 2" << std::endl;
            return EXIT_FAILURE;
        }
        // check a1 input tensor
        auto a1_tensor_info = session->GetInputTypeInfo(0).GetTensorTypeAndShapeInfo();
        size_t a1_dim_count = a1_tensor_info.GetDimensionsCount();
        if (a1_dim_count != 1) {  // channel
            std::cerr << "This model is not supported by onnx recongnizer. a1 dim is not 1!" << std::endl;
            return EXIT_FAILURE;
        }
        auto word_tensor_info = session->GetInputTypeInfo(1).GetTensorTypeAndShapeInfo();
        size_t word_dim_count = word_tensor_info.GetDimensionsCount();
        if (word_dim_count != 2) {  // nums*idx
            std::cerr << "This model is not supported by onnx recongnizer. word dim is not 2!" << std::endl;
            return EXIT_FAILURE;
        }
        std::vector<int64_t> word_dims = word_tensor_info.GetShape();
        if (word_dims[1] != 2) {
            std::cerr << "This model is not supported by onnx recongnizer. word need be in NC format" << std::endl;
            return EXIT_FAILURE;
        }

        // check output tensor nums
        size_t out_count = session->GetOutputCount();
        if (out_count != 1) {
            std::cerr << "This model is not supported by onnx recongnizer. output nums must 1" << std::endl;
            return EXIT_FAILURE;
        }
        // check output tensor
        auto out_tensor_info = session->GetOutputTypeInfo(0).GetTensorTypeAndShapeInfo();
        size_t out_dim_count = out_tensor_info.GetDimensionsCount();
        if (out_dim_count != 1) {  // timestep
            std::cerr << "This model is not supported by onnx recongnizer. output dim is not 1!" << std::endl;
            return EXIT_FAILURE;
        }

        Ort::AllocatorWithDefaultOptions ort_alloc;
        input_name_.emplace_back(session->GetInputNameAllocated(0, ort_alloc).get());
        input_name_.emplace_back(session->GetInputNameAllocated(1, ort_alloc).get());
        output_name_.emplace_back(session->GetOutputNameAllocated(0, ort_alloc).get());

        return EXIT_SUCCESS;
    }

   private:
    /**
     * @brief Get the feateure embeding object
     *
     * @param dstSrc
     * @param props
     */
    void decode(const AtomList& dstSrc, std::vector<size_t>& seq) const {
        // create inputs
        auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        std::vector<Ort::Value> ort_inputs;
        // create tensor data
        std::vector<int64_t> alist_tensor_values;
        alist_tensor_values.reserve(dstSrc.size() * 2 + 2);
        int64_t words_size              = dstSrc.size() + 2;
        size_t words_tensor_size        = words_size * 2;
        std::vector<int64_t> words_dims = {words_size, 2};
        std::vector<int64_t> words_tensor_values(words_tensor_size);
        // set alist data
        wordpiece->encode(dstSrc, [&words_tensor_values, &alist_tensor_values](int code, int atom_postion) {
            if (atom_postion >= 0) {
                size_t idx   = atom_postion * 2 + 2;
                int64_t aidx = alist_tensor_values.size();

                words_tensor_values[idx]     = std::min(words_tensor_values[idx], aidx);
                words_tensor_values[idx + 1] = std::max(words_tensor_values[idx + 1], aidx);
            }
            alist_tensor_values.push_back(code);
        });
        int64_t adim = alist_tensor_values.size();
        // set head and tail
        words_tensor_values[0] = words_tensor_values[1] = 0;
        words_tensor_values[words_tensor_size - 2] = words_tensor_values[words_tensor_size - 1] = adim - 1;
        // push data
        Ort::Value alist_input_tensor =
            Ort::Value::CreateTensor<int64_t>(memory_info, alist_tensor_values.data(), adim, &adim, 1);
        assert(alist_input_tensor.IsTensor());
        ort_inputs.push_back(std::move(alist_input_tensor));

        Ort::Value words_input_tensor = Ort::Value::CreateTensor<int64_t>(
            memory_info, words_tensor_values.data(), words_tensor_size, words_dims.data(), words_dims.size());
        assert(words_input_tensor.IsTensor());
        ort_inputs.push_back(std::move(words_input_tensor));
        // score model & input tensor, get back output tensor
        Ort::Value output_tensor{nullptr};
        session->Run(Ort::RunOptions{nullptr}, input_name_.data(), ort_inputs.data(), ort_inputs.size(),
                     output_name_.data(), &output_tensor, output_name_.size());

        // Get pointer to output tensor float values
        int64_t* arr = output_tensor.GetTensorMutableData<int64_t>();
        seq.insert(seq.end(), arr, arr + dstSrc.size());
    }

   public:
    int initalize(const std::map<std::string, std::string>& params,
                  std::map<std::string, std::shared_ptr<SegmentPlugin>>& plugins) {
        auto iter = params.find(LABELS_KEY);
        if (iter == params.end()) {
            std::cerr << LABELS_KEY << " key not found in dictionary!" << std::endl;
            return EXIT_FAILURE;
        }
        split(iter->second, ",", labels);
        // load wordpice
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

        // load model
        auto pit = params.find(MODEL_PATH_KEY);
        if (pit == params.end()) {
            std::cerr << "no param key find" << MODEL_PATH_KEY << std::endl;
            return EXIT_FAILURE;
        }
        std::string modelpath = getResource(pit->second);
        this->session         = loadmodel(modelpath.c_str());
        if (!this->session) {
            std::cerr << "load model " << MODEL_PATH_KEY << " from path " << modelpath << " failed!" << std::endl;
            return EXIT_FAILURE;
        }
        if (validator(this->session)) {
            return EXIT_FAILURE;
        }

        return EXIT_SUCCESS;
    }

    void addWords(const AtomList& dstSrc, SegPath& cmap) const {
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
    ~OnnxRecongnizer() {
        if (session) {
            delete session;
            session = nullptr;
        }
        input_name_.clear();
        output_name_.clear();
        labels.clear();
    }
};

const char* OnnxRecongnizer::LABELS_KEY      = "label.list";
const char* OnnxRecongnizer::MODEL_PATH_KEY  = "model.path";
const char* OnnxRecongnizer::WORDPIECE_PARAM = "wordpice.name";

REGISTER_Recognizer(OnnxRecongnizer);

}  // namespace darts
#endif  // SRC_IMPL_NETWORDQI_HPP_
