/*
 * File: networdqi.hpp
 * Project: impl
 * File Created: Sunday, 2nd January 2022 10:06:07 am
 * Author: Xu En (nanhangxuen@163.com)
 * -----
 * Last Modified: Sunday, 2nd January 2022 10:06:10 am
 * Modified By: Xu En (nanhangxuen@163.com)
 * -----
 * Copyright 2021 - 2022 XuEn
 */
#ifndef SRC_IMPL_NETWORDQI_HPP_
#define SRC_IMPL_NETWORDQI_HPP_

#include <onnxruntime_cxx_api.h>
#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <iostream>
#include <limits>
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
    static Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "darts");
    Ort::SessionOptions session_options;
    session_options.EnableMemPattern();
    session_options.EnableCpuMemArena();
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    try {
        return new Ort::Session(env, model_path, session_options);
    } catch (const Ort::Exception& error) {
        std::cerr << "ERROR: load ONNX model " << model_path << ": " << error.what() << std::endl;
        return nullptr;
    }
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
    Ort::Session* session = nullptr;
    std::vector<char*> input_name_;
    std::vector<char*> output_name_;

    std::shared_ptr<WordPice> wordpiece;
    std::shared_ptr<TypeEncoder> lencoder;

    int validator(Ort::Session* session) {
        // check input tensor nums
        size_t input_count = session->GetInputCount();
        if (input_count != 2) {
            std::cerr << "This model is not supported by onnx indicator.  input nums must 2 not " << input_count
                      << std::endl;
            return EXIT_FAILURE;
        }
        // check atomlist input tensor
        auto alist_tensor      = session->GetInputTypeInfo(0);
        auto alist_tensor_info = alist_tensor.GetTensorTypeAndShapeInfo();
        size_t alist_dim_count = alist_tensor_info.GetDimensionsCount();
        if (alist_dim_count != 1) {  // timestep
            std::cerr << "This model is not supported by onnx indicator. alist dim must 1 not " << alist_dim_count
                      << std::endl;
            return EXIT_FAILURE;
        }
        // check word input tensor
        auto word_tensor      = session->GetInputTypeInfo(1);
        auto word_tensor_info = word_tensor.GetTensorTypeAndShapeInfo();
        size_t word_dim_count = word_tensor_info.GetDimensionsCount();
        if (word_dim_count != 2) {  // nums*idx
            std::cerr << "This model is not supported by onnx indicator. word dim must 2 not " << word_dim_count
                      << std::endl;
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
        auto emb_tensor      = session->GetOutputTypeInfo(0);
        auto emb_tensor_info = emb_tensor.GetTensorTypeAndShapeInfo();
        size_t emb_dim_count = emb_tensor_info.GetDimensionsCount();
        if (emb_dim_count != 2) {  // nums*dim
            std::cerr << "This model is not supported by onnx indicator. output dim must 2 not " << emb_dim_count
                      << std::endl;
            return EXIT_FAILURE;
        }
        auto emb_dims = emb_tensor_info.GetShape();
        this->emdim   = emb_dims[1];

        Ort::AllocatorWithDefaultOptions ort_alloc;
        for (int i = 0; i < input_count; i++) {
            auto name_ptr = session->GetInputNameAllocated(i, ort_alloc);
            input_name_.push_back(strdup(name_ptr.get()));
        }
        for (int i = 0; i < out_count; i++) {
            auto name_ptr = session->GetOutputNameAllocated(i, ort_alloc);
            output_name_.push_back(strdup(name_ptr.get()));
        }

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
        const size_t unset = std::numeric_limits<size_t>::max();
        std::vector<size_t> start_(dstSrc.size(), unset);
        std::vector<size_t> ends_(dstSrc.size(), unset);
        // set alist data
        wordpiece->encode(dstSrc, [&](int code, int atom_postion) {
            if (atom_postion >= 0) {
                auto sz = alist_tensor_values.size();
                if (start_[atom_postion] == unset) start_[atom_postion] = sz;
                ends_[atom_postion] = sz;
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
        cmap.iterRow(nullptr, -1, [this, &words_tensor_values, &start_, &ends_, unset](Cursor cur) {
            auto w     = cur->val;
            size_t idx = (cur->idx + 1) * 3;

            if (start_[w->st] == unset || ends_[w->et - 1] == unset) {
                throw std::runtime_error("candidate has no aligned WordPiece span");
            }

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
        const float* floatarr = output_tensor.GetTensorData<float>();
        auto embeddings = std::make_shared<std::vector<float>>(floatarr, floatarr + words_size * emdim);
        cmap.SrcNode()->setAttView(embeddings, 0, emdim);
        cmap.iterRow(nullptr, -1, [this, &embeddings](Cursor cur) {
            cur->val->setAttView(embeddings, (cur->idx + 1) * emdim, emdim);
        });
        cmap.EndNode()->setAttView(embeddings, (words_size - 1) * emdim, emdim);
    }

    ~OnnxIndicator() {
        if (this->session) {
            delete this->session;
            this->session = nullptr;
        }
        for (auto ptr : input_name_) free(ptr);
        for (auto ptr : output_name_) free(ptr);
        input_name_.clear();
        output_name_.clear();
        wordpiece = nullptr;
        lencoder  = nullptr;
    }
};
const char* OnnxIndicator::MODEL_PATH_KEY    = "pmodel.path";
const char* OnnxIndicator::WORDPIECE_PARAM   = "wordpiece.name";
const char* OnnxIndicator::TYPEENCODER_PARAM = "tencode.name";

class OnnxQuantizer {
    // ONNX contract: two [edges, embedding] float tensors produce one
    // [edges] association NLL tensor.  Values are path costs, so NaN, infinity,
    // and negative values are converted to a large finite rejection cost.
   private:
    static const char* MODEL_PATH_KEY;
    size_t emdim;
    Ort::Session* session;
    std::vector<char*> input_name_;
    std::vector<char*> output_name_;
    bool batched_input = false;

    int validator(Ort::Session* session) {
        // check input tensor nums
        size_t input_count = session->GetInputCount();
        if (input_count != 2) {
            std::cerr << "This model is not supported by onnx quantizer.  input nums must 2 not " << input_count
                      << std::endl;
            return EXIT_FAILURE;
        }
        // check a1 input tensor
        auto a1_tensor      = session->GetInputTypeInfo(0);
        auto a1_tensor_info = a1_tensor.GetTensorTypeAndShapeInfo();
        size_t a1_dim_count = a1_tensor_info.GetDimensionsCount();
        if (a1_dim_count != 1 && a1_dim_count != 2) {
            std::cerr << "This model is not supported by onnx quantizer. a1 dim must 1 or 2 not " << a1_dim_count
                      << std::endl;
            return EXIT_FAILURE;
        }
        std::vector<int64_t> alist_dims = a1_tensor_info.GetShape();
        if (alist_dims.back() != static_cast<int64_t>(emdim)) {
            std::cerr << "This model is not supported by onnx quantizer. a1 need be in C format" << std::endl;
            return EXIT_FAILURE;
        }
        // check a2 input tensor
        auto a2_tensor      = session->GetInputTypeInfo(1);
        auto a2_tensor_info = a2_tensor.GetTensorTypeAndShapeInfo();
        size_t a2_dim_count = a2_tensor_info.GetDimensionsCount();
        if (a2_dim_count != a1_dim_count) {
            std::cerr << "This model is not supported by onnx quantizer. input ranks differ: " << a1_dim_count << " and " << a2_dim_count
                      << std::endl;
            return EXIT_FAILURE;
        }
        std::vector<int64_t> a2_dims = a2_tensor_info.GetShape();
        if (a2_dims.back() != static_cast<int64_t>(emdim)) {
            std::cerr << "This model is not supported by onnx quantizer. a2 need be in C format" << std::endl;
            return EXIT_FAILURE;
        }
        // check output tensor nums
        size_t out_count = session->GetOutputCount();
        if (out_count != 1) {
            std::cerr << "This model is not supported by onnx quantizer. output nums must 1 not " << out_count
                      << std::endl;
            return EXIT_FAILURE;
        }
        // check output tensor
        auto out_tensor      = session->GetOutputTypeInfo(0);
        auto out_tensor_info = out_tensor.GetTensorTypeAndShapeInfo();
        size_t out_dim_count = out_tensor_info.GetDimensionsCount();
        if (out_dim_count != 1) {  // nums*dim
            std::cerr << "This model is not supported by onnx quantizer. output rank must 1 not " << out_dim_count
                      << std::endl;
            return EXIT_FAILURE;
        }
        auto out_dims = out_tensor_info.GetShape();
        if (a1_dim_count == 1 && out_dims[0] != 1) {
            std::cerr << "This model is not supported by onnx quantizer. scalar edge output must have size 1"
                      << std::endl;
            return EXIT_FAILURE;
        }
        batched_input = a1_dim_count == 2;
        Ort::AllocatorWithDefaultOptions ort_alloc;
        for (int i = 0; i < input_count; i++) {
            auto name_ptr = session->GetInputNameAllocated(i, ort_alloc);
            input_name_.push_back(strdup(name_ptr.get()));
        }
        for (int i = 0; i < out_count; i++) {
            auto name_ptr = session->GetOutputNameAllocated(i, ort_alloc);
            output_name_.push_back(strdup(name_ptr.get()));
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
     * @return negative log-probability, must be >= 0
     */
    double ranging(const std::shared_ptr<Word> pre, const std::shared_ptr<Word> next) const {
        if (pre == nullptr || next == nullptr) return 1e6;
        if (pre->getAttData() == nullptr || next->getAttData() == nullptr) {
            return 1e6;
        }
        // create inputs
        std::vector<Ort::Value> ort_inputs;
        auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        std::vector<int64_t> dims = {static_cast<int64_t>(this->emdim)};
        if (batched_input) dims.insert(dims.begin(), 1);

        // set data
        const float* x1 = pre->getAttData();
        Ort::Value a1_tensor = Ort::Value::CreateTensor<float>(
            memory_info, const_cast<float*>(x1), this->emdim, dims.data(), dims.size());
        assert(a1_tensor.IsTensor());
        ort_inputs.push_back(std::move(a1_tensor));

        const float* x2 = next->getAttData();
        Ort::Value a2_tensor = Ort::Value::CreateTensor<float>(
            memory_info, const_cast<float*>(x2), this->emdim, dims.data(), dims.size());
        assert(a2_tensor.IsTensor());
        ort_inputs.push_back(std::move(a2_tensor));
        // run model
        Ort::Value output_tensor{nullptr};
        session->Run(Ort::RunOptions{nullptr}, input_name_.data(), ort_inputs.data(), ort_inputs.size(),
                     output_name_.data(), &output_tensor, output_name_.size());
        // Get pointer to output tensor float values
        const double association_nll = output_tensor.GetTensorMutableData<float>()[0];
        return std::isfinite(association_nll) && association_nll >= 0.0 ? association_nll : 1e6;
    }

    void rangingBatch(
        const std::vector<std::pair<std::shared_ptr<Word>, std::shared_ptr<Word>>>& pairs,
        std::vector<double>& weights) const {
        if (!batched_input) {
            weights.reserve(weights.size() + pairs.size());
            for (const auto& pair : pairs) weights.push_back(ranging(pair.first, pair.second));
            return;
        }

        const size_t edge_count = pairs.size();
        if (edge_count == 0) return;
        thread_local std::vector<float> first;
        thread_local std::vector<float> second;
        thread_local std::vector<bool> valid;
        first.assign(edge_count * emdim, 0.0f);
        second.assign(edge_count * emdim, 0.0f);
        valid.assign(edge_count, false);
        for (size_t i = 0; i < edge_count; ++i) {
            const float* a = pairs[i].first ? pairs[i].first->getAttData() : nullptr;
            const float* b = pairs[i].second ? pairs[i].second->getAttData() : nullptr;
            valid[i] = a && b;
            if (a) std::copy_n(a, emdim, first.data() + i * emdim);
            if (b) std::copy_n(b, emdim, second.data() + i * emdim);
        }

        auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        std::vector<int64_t> dims = {static_cast<int64_t>(edge_count), static_cast<int64_t>(emdim)};
        std::vector<Ort::Value> inputs;
        inputs.reserve(2);
        inputs.push_back(Ort::Value::CreateTensor<float>(memory_info, first.data(), first.size(), dims.data(), dims.size()));
        inputs.push_back(Ort::Value::CreateTensor<float>(memory_info, second.data(), second.size(), dims.data(), dims.size()));
        Ort::Value output{nullptr};
        session->Run(Ort::RunOptions{nullptr}, input_name_.data(), inputs.data(), inputs.size(),
                     output_name_.data(), &output, output_name_.size());
        const float* association_nll = output.GetTensorData<float>();
        weights.reserve(weights.size() + edge_count);
        for (size_t i = 0; i < edge_count; ++i)
            weights.push_back(valid[i] && std::isfinite(association_nll[i]) && association_nll[i] >= 0.0f
                                  ? association_nll[i]
                                  : 1e6);
    }

    ~OnnxQuantizer() {
        if (this->session) {
            delete this->session;
            this->session = nullptr;
        }
        for (auto ptr : input_name_) free(ptr);
        for (auto ptr : output_name_) free(ptr);
        input_name_.clear();
        output_name_.clear();
    }
};
const char* OnnxQuantizer::MODEL_PATH_KEY = "qmodel.path";

/**
 * @brief
 *
 */
class OnnxDecider : public Decider {
   private:
    OnnxQuantizer quantizer;
    OnnxIndicator indicator;

    static bool isDeterministicRule(const std::shared_ptr<Word>& word) {
        if (!word) return false;
        const auto& labels = word->getLabels();
        return labels.find("DATE") != labels.end() || labels.find("DIGIT") != labels.end();
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
        if (indicator.load(params, plugins)) return EXIT_FAILURE;
        if (quantizer.load(params, indicator.getEmbSize())) return EXIT_FAILURE;
        return EXIT_SUCCESS;
    }
    /**
     * @brief set all word embeding
     *
     * @param dstSrc
     * @param cmap
     */
    void embed(const AtomList& dstSrc, SegPath& cmap) const {
        try {
            indicator.embed(dstSrc, cmap);
        } catch (const Ort::Exception& error) {
            std::cerr << "ONNX indicator inference failed: " << error.what() << std::endl;
        } catch (const std::exception& error) {
            std::cerr << "neural indicator failed: " << error.what() << std::endl;
        }
    }
    /**
     * @brief
     *
     * @param pre
     * @param next
     * @return association negative log-probability, must be >= 0
     */
    double ranging(const std::shared_ptr<Word> pre, const std::shared_ptr<Word> next) const {
        return quantizer.ranging(pre, next);
    }

    void rangingBatch(
        const std::vector<std::pair<std::shared_ptr<Word>, std::shared_ptr<Word>>>& pairs,
        std::vector<double>& weights) const override {
        const size_t offset = weights.size();
        quantizer.rangingBatch(pairs, weights);
        // Neural probabilities may be poorly calibrated for deterministic
        // date/time/unit rules. Preserve neural ordering while strongly
        // preferring paths through those high-confidence rule candidates.
        for (size_t index = 0; index < pairs.size(); ++index) {
            if (isDeterministicRule(pairs[index].first) ||
                isDeterministicRule(pairs[index].second))
                weights[offset + index] *= 0.05;
        }
    }

    ~OnnxDecider() {}
};

REGISTER_Decider(OnnxDecider);

/**
 * @brief a cell recongnizer that use hmm
 *
 */
class OnnxRecongnizer : public CellRecognizer {
    // ONNX contract: token ids [pieces], inclusive spans [candidates, 2], and
    // independent word probabilities [candidates].  Unlike BIO decoding this
    // permits arbitrary overlap; the downstream DAG decider chooses a path.
   private:
    static const char* MODEL_PATH_KEY;
    static const char* WORDPIECE_PARAM;
    static const char* MAX_SPAN_KEY;
    static const char* THRESHOLD_KEY;

    size_t max_span = 5;
    float threshold = 0.5f;
    std::vector<float> length_thresholds;
    std::shared_ptr<WordPice> wordpiece;

    Ort::Session* session = nullptr;
    std::vector<char*> input_name_;
    std::vector<char*> output_name_;

    int validator(Ort::Session* session) {
        // check input tensor nums
        size_t input_count = session->GetInputCount();
        if (input_count != 2) {
            std::cerr << "This model is not supported by onnx recongnizer.  input nums must 2 not " << input_count
                      << std::endl;
            return EXIT_FAILURE;
        }
        // check a1 input tensor
        auto a1_tensor      = session->GetInputTypeInfo(0);
        auto a1_tensor_info = a1_tensor.GetTensorTypeAndShapeInfo();
        auto a1_dim_count   = a1_tensor_info.GetDimensionsCount();
        if (a1_dim_count != 1) {  // channel
            std::cerr << "This model is not supported by onnx recongnizer. first dim must 1 not " << a1_dim_count
                      << std::endl;
            return EXIT_FAILURE;
        }

        // check a2 input tensor
        auto word_tensor      = session->GetInputTypeInfo(1);
        auto word_tensor_info = word_tensor.GetTensorTypeAndShapeInfo();
        size_t word_dim_count = word_tensor_info.GetDimensionsCount();
        if (word_dim_count != 2) {  // nums*idx
            std::cerr << "This model is not supported by onnx recongnizer. word dim must 2 not " << word_dim_count
                      << std::endl;
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
            std::cerr << "This model is not supported by onnx recongnizer. output nums must 1 not " << out_count
                      << std::endl;
            return EXIT_FAILURE;
        }
        // check output tensor
        auto out_tensor      = session->GetOutputTypeInfo(0);
        auto out_tensor_info = out_tensor.GetTensorTypeAndShapeInfo();
        if (out_tensor_info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
            std::cerr << "ONNX recognizer output must contain word probabilities" << std::endl;
            return EXIT_FAILURE;
        }
        size_t out_dim_count = out_tensor_info.GetDimensionsCount();
        if (out_dim_count != 1) {  // timestep
            std::cerr << "This model is not supported by onnx recongnizer. output dim must 1 not " << out_dim_count
                      << std::endl;
            return EXIT_FAILURE;
        }
        Ort::AllocatorWithDefaultOptions ort_alloc;
        for (int i = 0; i < input_count; i++) {
            auto name_ptr = session->GetInputNameAllocated(i, ort_alloc);
            input_name_.push_back(strdup(name_ptr.get()));
        }
        for (int i = 0; i < out_count; i++) {
            auto name_ptr = session->GetOutputNameAllocated(i, ort_alloc);
            output_name_.push_back(strdup(name_ptr.get()));
        }

        return EXIT_SUCCESS;
    }

   private:
    void predictSpans(const AtomList& dstSrc, std::vector<std::pair<size_t, size_t>>& spans,
                      std::vector<float>& probabilities) const {
        if (dstSrc.size() < 2) return;
        std::vector<int64_t> token_ids;
        std::vector<int64_t> starts(dstSrc.size(), -1);
        std::vector<int64_t> ends(dstSrc.size(), -1);
        wordpiece->encode(dstSrc, [&token_ids, &starts, &ends](int code, int atom_position) {
            const int64_t index = static_cast<int64_t>(token_ids.size());
            token_ids.push_back(code);
            if (atom_position < 0) return;
            if (starts[atom_position] < 0) starts[atom_position] = index;
            ends[atom_position] = index;
        });

        std::vector<int64_t> span_values;
        // Atom spans are half-open [start, end).  WordEncoder consumes inclusive
        // WordPiece bounds, hence ends[end - 1] in the ONNX input.
        for (size_t start = 0; start < dstSrc.size(); ++start) {
            const size_t limit = std::min(dstSrc.size(), start + max_span);
            for (size_t end = start + 2; end <= limit; ++end) {
                if (starts[start] < 0 || ends[end - 1] < 0) continue;
                spans.emplace_back(start, end);
                span_values.push_back(starts[start]);
                span_values.push_back(ends[end - 1]);
            }
        }
        if (spans.empty()) return;

        auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        const int64_t token_count = static_cast<int64_t>(token_ids.size());
        std::vector<int64_t> span_dims = {static_cast<int64_t>(spans.size()), 2};
        std::vector<Ort::Value> inputs;
        inputs.push_back(Ort::Value::CreateTensor<int64_t>(memory_info, token_ids.data(), token_ids.size(),
                                                           &token_count, 1));
        inputs.push_back(Ort::Value::CreateTensor<int64_t>(memory_info, span_values.data(), span_values.size(),
                                                           span_dims.data(), span_dims.size()));
        Ort::Value output{nullptr};
        session->Run(Ort::RunOptions{nullptr}, input_name_.data(), inputs.data(), inputs.size(),
                     output_name_.data(), &output, output_name_.size());
        const auto info = output.GetTensorTypeAndShapeInfo();
        if (info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
            info.GetElementCount() != spans.size()) {
            throw std::runtime_error("span recognizer returned an invalid probability tensor");
        }
        const float* values = output.GetTensorData<float>();
        probabilities.assign(values, values + spans.size());
    }

   public:
    int initalize(const std::map<std::string, std::string>& params,
                  std::map<std::string, std::shared_ptr<SegmentPlugin>>& plugins) {
        auto iter = params.find(MAX_SPAN_KEY);
        try {
            if (iter != params.end()) max_span = std::stoul(iter->second);
            iter = params.find(THRESHOLD_KEY);
            if (iter != params.end()) threshold = std::stof(iter->second);
        } catch (const std::exception& error) {
            std::cerr << "invalid neural recognizer numeric option: " << error.what() << std::endl;
            return EXIT_FAILURE;
        }
        if (max_span < 2 || max_span > 32) {
            std::cerr << MAX_SPAN_KEY << " must be in [2, 32]" << std::endl;
            return EXIT_FAILURE;
        }
        if (!(threshold >= 0.0f && threshold <= 1.0f)) {
            std::cerr << THRESHOLD_KEY << " must be in [0, 1]" << std::endl;
            return EXIT_FAILURE;
        }
        length_thresholds.assign(max_span + 1, threshold);
        for (size_t length = 2; length <= max_span; ++length) {
            const std::string key = std::string(THRESHOLD_KEY) + "." + std::to_string(length);
            iter = params.find(key);
            if (iter == params.end()) continue;
            try {
                length_thresholds[length] = std::stof(iter->second);
            } catch (const std::exception& error) {
                std::cerr << "invalid " << key << ": " << error.what() << std::endl;
                return EXIT_FAILURE;
            }
            if (!(length_thresholds[length] >= 0.0f && length_thresholds[length] <= 1.0f)) {
                std::cerr << key << " must be in [0, 1]" << std::endl;
                return EXIT_FAILURE;
            }
        }
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
        std::vector<std::pair<size_t, size_t>> spans;
        std::vector<float> probabilities;
        try {
            predictSpans(dstSrc, spans, probabilities);
        } catch (const std::exception& error) {
            std::cerr << "ONNX span recognizer inference failed: " << error.what() << std::endl;
            return;
        }
        Cursor cur = cmap.Head();
        for (size_t index = 0; index < spans.size(); ++index) {
            const size_t length = spans[index].second - spans[index].first;
            if (!std::isfinite(probabilities[index]) || probabilities[index] < length_thresholds[length]) continue;
            auto word = std::make_shared<Word>(dstSrc, spans[index].first, spans[index].second);
            word->addLabel("_NWORD");
            cur = cmap.addCell(word, cur);
        }
    }
    ~OnnxRecongnizer() {
        if (session) {
            delete session;
            session = nullptr;
        }
        for (auto ptr : input_name_) free(ptr);
        for (auto ptr : output_name_) free(ptr);
        input_name_.clear();
        output_name_.clear();
    }
};

const char* OnnxRecongnizer::MODEL_PATH_KEY  = "model.path";
const char* OnnxRecongnizer::WORDPIECE_PARAM = "wordpiece.name";
const char* OnnxRecongnizer::MAX_SPAN_KEY     = "max.span";
const char* OnnxRecongnizer::THRESHOLD_KEY    = "threshold";

REGISTER_Recognizer(OnnxRecongnizer);

class OnnxSyntaxRecongnizer : public CellRecognizer {
   private:
    static const char* MODEL_PATH_KEY;
    static const char* LABEL_PATH_KEY;
    static const char* WORDPIECE_PARAM;
    static const char* MAX_SPAN_KEY;
    static const char* THRESHOLD_KEY;

    size_t max_span = 5;
    float threshold = 0.5f;
    std::shared_ptr<WordPice> wordpiece;
    std::vector<std::string> labels;
    Ort::Session* session = nullptr;
    std::vector<char*> input_name_;
    std::vector<char*> output_name_;

    int loadLabels(const std::string& path) {
        std::ifstream input(getResource(path));
        if (!input.is_open()) return EXIT_FAILURE;
        std::string line;
        while (std::getline(input, line)) {
            darts::trim(line);
            if (!line.empty()) labels.push_back(line);
        }
        if (labels.size() < 2 || labels.front() != "NOT_WORD") {
            std::cerr << "syntax labels must start with NOT_WORD and contain POS classes" << std::endl;
            return EXIT_FAILURE;
        }
        return EXIT_SUCCESS;
    }

    int validator() {
        if (session->GetInputCount() != 2 || session->GetOutputCount() != 1) return EXIT_FAILURE;
        // TensorTypeAndShapeInfo is an unowned view. Keep its parent TypeInfo
        // alive until all shape and element-type checks have completed.
        auto span_type = session->GetInputTypeInfo(1);
        auto output_type = session->GetOutputTypeInfo(0);
        auto span_info = span_type.GetTensorTypeAndShapeInfo();
        auto output_info = output_type.GetTensorTypeAndShapeInfo();
        const auto span_shape = span_info.GetShape();
        const auto output_shape = output_info.GetShape();
        if (span_shape.size() != 2 || span_shape[1] != 2 ||
            output_shape.size() != 2 ||
            output_info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
            std::cerr << "syntax recognizer requires spans[N,2] and probabilities[N,C]" << std::endl;
            return EXIT_FAILURE;
        }
        if (output_shape[1] > 0 && static_cast<size_t>(output_shape[1]) != labels.size()) {
            std::cerr << "syntax model class count does not match label file" << std::endl;
            return EXIT_FAILURE;
        }
        Ort::AllocatorWithDefaultOptions allocator;
        for (size_t i = 0; i < session->GetInputCount(); ++i) {
            auto name = session->GetInputNameAllocated(i, allocator);
            input_name_.push_back(strdup(name.get()));
        }
        auto name = session->GetOutputNameAllocated(0, allocator);
        output_name_.push_back(strdup(name.get()));
        return EXIT_SUCCESS;
    }

   public:
    int initalize(const std::map<std::string, std::string>& params,
                  std::map<std::string, std::shared_ptr<SegmentPlugin>>& plugins) override {
        try {
            auto option = params.find(MAX_SPAN_KEY);
            if (option != params.end()) max_span = std::stoul(option->second);
            option = params.find(THRESHOLD_KEY);
            if (option != params.end()) threshold = std::stof(option->second);
        } catch (const std::exception& error) {
            std::cerr << "invalid syntax recognizer option: " << error.what() << std::endl;
            return EXIT_FAILURE;
        }
        if (max_span < 1 || max_span > 32 || threshold < 0.0f || threshold > 1.0f)
            return EXIT_FAILURE;

        auto dependency = plugins.find(WORDPIECE_PARAM);
        if (dependency == plugins.end()) return EXIT_FAILURE;
        wordpiece = std::dynamic_pointer_cast<WordPice>(dependency->second);
        if (!wordpiece) return EXIT_FAILURE;

        auto label_path = params.find(LABEL_PATH_KEY);
        auto model_path = params.find(MODEL_PATH_KEY);
        if (label_path == params.end() || model_path == params.end() ||
            loadLabels(label_path->second)) return EXIT_FAILURE;
        const std::string resolved = getResource(model_path->second);
        session = loadmodel(resolved.c_str());
        return session && !validator() ? EXIT_SUCCESS : EXIT_FAILURE;
    }

    void addWords(const AtomList& atoms, SegPath& path) const override {
        if (atoms.size() == 0) return;
        std::vector<int64_t> token_ids;
        std::vector<int64_t> starts(atoms.size(), -1), ends(atoms.size(), -1);
        wordpiece->encode(atoms, [&](int code, int atom_position) {
            const int64_t index = token_ids.size();
            token_ids.push_back(code);
            if (atom_position >= 0) {
                if (starts[atom_position] < 0) starts[atom_position] = index;
                ends[atom_position] = index;
            }
        });

        std::vector<std::pair<size_t, size_t>> spans;
        std::vector<int64_t> span_values;
        for (size_t start = 0; start < atoms.size(); ++start) {
            for (size_t end = start + 1; end <= std::min(atoms.size(), start + max_span); ++end) {
                if (starts[start] < 0 || ends[end - 1] < 0) continue;
                spans.emplace_back(start, end);
                span_values.push_back(starts[start]);
                span_values.push_back(ends[end - 1]);
            }
        }
        if (spans.empty()) return;

        try {
            auto memory = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
            const int64_t token_count = token_ids.size();
            std::vector<int64_t> span_dims = {static_cast<int64_t>(spans.size()), 2};
            std::vector<Ort::Value> inputs;
            inputs.push_back(Ort::Value::CreateTensor<int64_t>(
                memory, token_ids.data(), token_ids.size(), &token_count, 1));
            inputs.push_back(Ort::Value::CreateTensor<int64_t>(
                memory, span_values.data(), span_values.size(), span_dims.data(), 2));
            Ort::Value output{nullptr};
            session->Run(Ort::RunOptions{nullptr}, input_name_.data(), inputs.data(), inputs.size(),
                         output_name_.data(), &output, 1);
            const auto info = output.GetTensorTypeAndShapeInfo();
            const auto shape = info.GetShape();
            if (shape.size() != 2 || shape[0] != static_cast<int64_t>(spans.size()) ||
                shape[1] != static_cast<int64_t>(labels.size()))
                throw std::runtime_error("invalid syntax probability tensor");
            const float* probabilities = output.GetTensorData<float>();
            Cursor cursor = path.Head();
            for (size_t span = 0; span < spans.size(); ++span) {
                const float* row = probabilities + span * labels.size();
                const size_t type = std::max_element(row, row + labels.size()) - row;
                if (type == 0 || !std::isfinite(row[type]) || row[type] < threshold) continue;
                auto word = std::make_shared<Word>(atoms, spans[span].first, spans[span].second);
                word->addLabel(labels[type]);
                cursor = path.addCell(word, cursor);
            }
        } catch (const std::exception& error) {
            std::cerr << "ONNX syntax recognizer inference failed: " << error.what() << std::endl;
        }
    }

    ~OnnxSyntaxRecongnizer() {
        delete session;
        for (auto name : input_name_) free(name);
        for (auto name : output_name_) free(name);
    }
};

const char* OnnxSyntaxRecongnizer::MODEL_PATH_KEY = "model.path";
const char* OnnxSyntaxRecongnizer::LABEL_PATH_KEY = "label.path";
const char* OnnxSyntaxRecongnizer::WORDPIECE_PARAM = "wordpiece.name";
const char* OnnxSyntaxRecongnizer::MAX_SPAN_KEY = "max.span";
const char* OnnxSyntaxRecongnizer::THRESHOLD_KEY = "threshold";
REGISTER_Recognizer(OnnxSyntaxRecongnizer);

}  // namespace darts
#endif  // SRC_IMPL_NETWORDQI_HPP_
