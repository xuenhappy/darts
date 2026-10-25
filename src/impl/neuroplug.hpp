#ifndef SRC_IMPL_NETWORDQI_HPP_
#define SRC_IMPL_NETWORDQI_HPP_
// 神经插件：MNN 推理后端（2026-10-25 自 onnxruntime 迁移）
// 导出链：torch --dynamo--> onnx --MNNConvert--> .mnn（见 train_joint export）
// 数值契约：MNN Precision_High 强制 FP32；动态 shape 经 resizeTensor/resizeSession
#include <cstdlib>
#include <cstring>
#include <memory>
#include <ostream>
#include <string>
#include <unordered_map>
#include <vector>
#include "../core/segment.hpp"
#include "../utils/filetool.hpp"
#include "../utils/strtool.hpp"
#include "./encoder.hpp"
#include <MNN/Interpreter.hpp>
#include <MNN/Tensor.hpp>
namespace darts {

// MNN 会话包装：加载、按名喂取、动态 shape resize（带 shape 缓存去重）
struct MnnModel {
    mutable std::shared_ptr<MNN::Interpreter> net;
    mutable MNN::Session* session = nullptr;
    std::vector<std::string> input_names;
    std::vector<std::string> output_names;
    mutable std::vector<std::vector<int>> last_shapes;

    static int thread_count() {
        int threads = 1;
        if (const char* value = std::getenv("DARTS_MNN_THREADS"))
            threads = std::atoi(value);
        else if (const char* value = std::getenv("DARTS_ORT_THREADS"))
            threads = std::atoi(value);  // 兼容历史变量名
        return threads > 0 ? threads : 1;  // MNN 小模型并行同步开销大于收益，单线程最优
    }

    int load(const std::string& path) {
        this->net.reset(MNN::Interpreter::createFromFile(path.c_str()));
        if (!this->net) return EXIT_FAILURE;
        MNN::BackendConfig backend;
        // Precision_High 强制 FP32（Low/Normal 档位有 ~5e-3 漂移）；
        // DARTS_MNN_PRECISION=low 可换 fp16 算术换速度（E2E 影响 ~0.1pp 级）
        const char* prec = std::getenv("DARTS_MNN_PRECISION");
        backend.precision = (prec && std::string(prec) == "low")
            ? MNN::BackendConfig::Precision_Low
            : MNN::BackendConfig::Precision_High;
        MNN::ScheduleConfig config;
        config.type          = MNN_FORWARD_CPU;
        config.numThread     = thread_count();
        config.backendConfig = &backend;
        this->session = this->net->createSession(config);
        if (this->session == nullptr) return EXIT_FAILURE;
        for (const auto& item : this->net->getSessionInputAll(this->session))
            this->input_names.push_back(item.first);
        for (const auto& item : this->net->getSessionOutputAll(this->session))
            this->output_names.push_back(item.first);
        this->last_shapes.assign(this->input_names.size(), {});
        return EXIT_SUCCESS;
    }

    int input_index(const std::string& name) const {
        for (size_t i = 0; i < this->input_names.size(); ++i)
            if (this->input_names[i] == name) return static_cast<int>(i);
        return -1;
    }

    MNN::Tensor* input(const std::string& name) const {
        MNN::Tensor* tensor = this->net->getSessionInput(this->session, name.c_str());
        if (tensor == nullptr) {
            std::string names;
            for (const auto& item : this->input_names) names += " " + item;
            throw std::runtime_error("MNN model has no input " + name + "; available:" + names);
        }
        return tensor;
    }

    MNN::Tensor* output(size_t index = 0) const {
        const char* name = index < this->output_names.size() ? this->output_names[index].c_str() : nullptr;
        return this->net->getSessionOutput(this->session, name);
    }

    void resize(const std::vector<std::vector<int>>& shapes) const {
        bool changed = shapes.size() != this->last_shapes.size();
        if (!changed) {
            for (size_t i = 0; i < shapes.size(); ++i)
                if (shapes[i] != this->last_shapes[i]) {
                    changed = true;
                    break;
                }
        }
        if (!changed) return;
        for (size_t i = 0; i < shapes.size() && i < this->input_names.size(); ++i)
            this->net->resizeTensor(this->input(this->input_names[i]), shapes[i]);
        this->net->resizeSession(this->session);
        this->last_shapes = shapes;
    }

    // 宿主张量以 device（resize 后）镜像构造，shape/dtype 天然一致
    void feed_int64(const std::string& name, const int64_t* values) const {
        MNN::Tensor* device = this->input(name);
        MNN::Tensor host(device, MNN::Tensor::CAFFE);
        const size_t count = static_cast<size_t>(host.elementSize());
        if (host.getType().code == halide_type_int && host.getType().bits == 64)
            std::copy_n(values, count, host.host<int64_t>());
        else
            for (size_t i = 0; i < count; ++i) host.host<int32_t>()[i] = static_cast<int32_t>(values[i]);
        if (!device->copyFromHostTensor(&host)) throw std::runtime_error("MNN copyFromHostTensor failed");
    }

    void feed_float(const std::string& name, const float* values) const {
        MNN::Tensor* device = this->input(name);
        MNN::Tensor host(device, MNN::Tensor::CAFFE);
        std::copy_n(values, static_cast<size_t>(host.elementSize()), host.host<float>());
        if (!device->copyFromHostTensor(&host)) throw std::runtime_error("MNN copyFromHostTensor failed");
    }

    void run() const {
        if (this->net->runSession(this->session) != 0)
            throw std::runtime_error("MNN runSession failed");
    }

    bool has_output(const std::string& name) const {
        return std::find(output_names.begin(), output_names.end(), name) != output_names.end();
    }

    // 按名取输出（双头模型的多输出契约）
    std::vector<float> fetch_float_named(const std::string& name,
                                         std::vector<int>* shape = nullptr) const {
        MNN::Tensor* device = this->net->getSessionOutput(this->session, name.c_str());
        if (device == nullptr) throw std::runtime_error("MNN model has no output " + name);
        MNN::Tensor host(device, MNN::Tensor::CAFFE);
        if (!device->copyToHostTensor(&host)) throw std::runtime_error("MNN copyToHostTensor failed");
        if (shape != nullptr) *shape = host.shape();
        const float* data = host.host<float>();
        return std::vector<float>(data, data + host.elementSize());
    }

    // 输出拉回 host；shape 随产出回填
    std::vector<float> fetch_float(std::vector<int>* shape = nullptr, size_t index = 0) const {
        MNN::Tensor* device = this->output(index);
        if (device == nullptr) throw std::runtime_error("MNN model has no output");
        MNN::Tensor host(device, MNN::Tensor::CAFFE);
        if (!device->copyToHostTensor(&host)) throw std::runtime_error("MNN copyToHostTensor failed");
        if (shape != nullptr) *shape = host.shape();
        const float* data = host.host<float>();
        return std::vector<float>(data, data + host.elementSize());
    }
};

/**
 * @brief a network mnn
 *
 */
class MnnIndicator {
   private:
    static const char* MODEL_PATH_KEY;
    static const char* WORDPIECE_PARAM;
    static const char* TYPEENCODER_PARAM;

    size_t emdim = 0;
    MnnModel model;

    std::shared_ptr<WordPice> wordpiece;
    std::shared_ptr<TypeEncoder> lencoder;

    // 首次 load 后用 1 词 warmup：确定 emdim（MNN 输出 shape 运行期才定）
    int warmup() {
        std::vector<int64_t> sents{codemap::sep_code, codemap::cls_code};
        std::vector<int64_t> words{0, 0, codemap::sep_code};
        this->model.resize({{2}, {1, 3}});
        this->model.feed_int64("sents", sents.data());
        this->model.feed_int64("wordinfo", words.data());
        this->model.run();
        std::vector<int> shape;
        this->model.fetch_float(&shape);
        if (shape.size() != 2 || shape[0] != 1 || shape[1] <= 0) {
            std::cerr << "MNN indicator output must be [words, dim], got rank " << shape.size() << std::endl;
            return EXIT_FAILURE;
        }
        this->emdim = static_cast<size_t>(shape[1]);
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
        if (this->model.load(modelpath) != EXIT_SUCCESS) {
            std::cerr << "load model " << MODEL_PATH_KEY << " from path " << modelpath << " failed!" << std::endl;
            return EXIT_FAILURE;
        }
        try {
            if (this->warmup()) return EXIT_FAILURE;
        } catch (const std::exception& error) {
            std::cerr << "MNN indicator warmup failed: " << error.what() << std::endl;
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
        // create words tensor
        int64_t words_size       = cmap.Size() + 2;
        size_t words_tensor_size = words_size * 3;
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
        // 2026-10-25 一致性修复：神经提名词（MnnSyntaxRecongnizer.addWords）
        // 词典词多标签（{POS_DET,POS_r}）经 H 序 encode 得 POS_r=29——训练里
        // 该 type 的词典副本几乎不在金标路径 → 量化器惩罚（NLL 48）。
        // 因此：单 POS_* 标签（神经提名）直接用；其余走词典 H 序。
        cmap.iterRow(nullptr, -1, [this, &words_tensor_values, &start_, &ends_, unset](Cursor cur) {
            auto w     = cur->val;
            size_t idx = (cur->idx + 1) * 3;

            if (start_[w->st] == unset || ends_[w->et - 1] == unset) {
                throw std::runtime_error("candidate has no aligned WordPiece span");
            }

            words_tensor_values[idx]     = start_[w->st];
            words_tensor_values[idx + 1] = ends_[w->et - 1];
            // 一致性修复（见上方注释）：神经提名单 POS_* 标签直接编码；
            // 词典多标签词回退 H 序（与训练 reader 的词典副本节点同型）
            words_tensor_values[idx + 2] = this->lencoder == nullptr
                ? codemap::unk_code
                : this->lencoder->encodeSyntaxPreferred(w);
        });
        // run model
            this->model.resize({{static_cast<int>(adim)}, {static_cast<int>(words_size), 3}});
        this->model.feed_int64("sents", alist_tensor_values.data());
        this->model.feed_int64("wordinfo", words_tensor_values.data());
        this->model.run();
        std::vector<int> shape;
        std::vector<float> flat = this->model.fetch_float(&shape);
        if (flat.size() != static_cast<size_t>(words_size) * this->emdim)
            throw std::runtime_error("MNN indicator returned unexpected embedding count");

        if (std::getenv("DARTS_MNN_DEBUG")) {
            std::cerr << "[mnn-ind] T=" << adim << " W=" << words_size << " emdim=" << this->emdim
                      << " sents=[";
            for (int64_t v : alist_tensor_values) std::cerr << v << ",";
            std::cerr << "] wi=[";
            for (size_t k = 0; k < 24 && k < words_tensor_values.size(); ++k) std::cerr << words_tensor_values[k] << ",";
            std::cerr << "] emb0=[";
            for (size_t k = 0; k < 6; ++k) std::cerr << flat[k] << ",";
            std::cerr << "] emb1=[";
            for (size_t k = 128; k < 134; ++k) std::cerr << flat[k] << ",";
            std::cerr << "]" << std::endl;
        }
        // set cmap data
        auto embeddings = std::make_shared<std::vector<float>>(std::move(flat));
        cmap.SrcNode()->setAttView(embeddings, 0, emdim);
        cmap.iterRow(nullptr, -1, [this, &embeddings](Cursor cur) {
            cur->val->setAttView(embeddings, (cur->idx + 1) * emdim, emdim);
        });
        cmap.EndNode()->setAttView(embeddings, (words_size - 1) * emdim, emdim);
    }

    ~MnnIndicator() {}
};
const char* MnnIndicator::MODEL_PATH_KEY    = "pmodel.path";
const char* MnnIndicator::WORDPIECE_PARAM   = "wordpiece.name";
const char* MnnIndicator::TYPEENCODER_PARAM = "tencode.name";

class MnnQuantizer {
    // MNN contract: two [edges, embedding] float tensors produce one
    // [edges] association NLL tensor.  Values are path costs, so NaN, infinity,
    // and negative values are converted to a large finite rejection cost.
   private:
    static const char* MODEL_PATH_KEY;
    size_t emdim;
    MnnModel model;
    bool batched_input = false;

    // warmup 判定输入形态：[E,dim]（批量）或 [dim]（单边），
    // 并把 {1,dim}/{dim} 常驻 shape 缓存起来
    int warmup() {
        const std::vector<int> row = {static_cast<int>(this->emdim)};
        try {
            this->model.resize({{1, row[0]}, {1, row[0]}});
            MNN::Tensor* device = this->model.input("a");
            this->batched_input = device->shape().size() == 2;
            if (!this->batched_input) this->model.resize({row, row});
        } catch (const std::exception& error) {
            std::cerr << "MNN quantizer warmup failed: " << error.what() << std::endl;
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
    int load(const std::map<std::string, std::string>& params, size_t edim) {
        this->emdim = edim;
        auto pit    = params.find(MODEL_PATH_KEY);
        if (pit == params.end()) {
            std::cerr << "no param key find" << MODEL_PATH_KEY << std::endl;
            return EXIT_FAILURE;
        }
        std::string modelpath = getResource(pit->second);
        if (this->model.load(modelpath) != EXIT_SUCCESS) {
            std::cerr << "load model " << MODEL_PATH_KEY << " from path " << modelpath << " failed!" << std::endl;
            return EXIT_FAILURE;
        }
        if (this->warmup()) return EXIT_FAILURE;
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
        const std::vector<int> batch_row = {1, static_cast<int>(this->emdim)};
        const std::vector<int> row       = {static_cast<int>(this->emdim)};
        const std::vector<int>& dims     = this->batched_input ? batch_row : row;
        this->model.resize({dims, dims});
        this->model.feed_float("a", pre->getAttData());
        this->model.feed_float("b", next->getAttData());
        this->model.run();
        std::vector<float> out = this->model.fetch_float();
        const double association_nll = out.empty() ? 1e6 : static_cast<double>(out[0]);
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
            valid[i]       = a && b;
            if (a) std::copy_n(a, emdim, first.data() + i * emdim);
            if (b) std::copy_n(b, emdim, second.data() + i * emdim);
        }

        const std::vector<int> dims = {static_cast<int>(edge_count), static_cast<int>(emdim)};
        this->model.resize({dims, dims});
        this->model.feed_float("a", first.data());
        this->model.feed_float("b", second.data());
        this->model.run();
        std::vector<float> association_nll = this->model.fetch_float();
        if (std::getenv("DARTS_MNN_DEBUG") && !association_nll.empty()) {
            std::cerr << "[mnn-q] E=" << edge_count << " nll0=[" << association_nll[0]
                      << "," << association_nll[1] << "," << association_nll[2]
                      << "] a0=[" << first[0] << "," << first[1] << "," << first[2]
                      << "] b0=[" << second[0] << "," << second[1] << "]" << std::endl;
        }
        if (association_nll.size() != edge_count)
            throw std::runtime_error("MNN quantizer returned an invalid NLL tensor");
        weights.reserve(weights.size() + edge_count);
        for (size_t i = 0; i < edge_count; ++i)
            weights.push_back(valid[i] && std::isfinite(association_nll[i]) && association_nll[i] >= 0.0f
                                  ? association_nll[i]
                                  : 1e6);
    }

    ~MnnQuantizer() {}
};
const char* MnnQuantizer::MODEL_PATH_KEY = "qmodel.path";

/**
 * @brief
 *
 */
class MnnDecider : public Decider {
   private:
    MnnQuantizer quantizer;
    MnnIndicator indicator;

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

    ~MnnDecider() {}
};

REGISTER_Decider(MnnDecider);

/**
 * @brief a cell recongnizer that use hmm
 *
 */
class MnnRecongnizer : public CellRecognizer {
    // MNN contract: token ids [pieces], inclusive spans [candidates, 2], and
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

    MnnModel model;

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
        // WordPiece bounds, hence ends[end - 1] in the model input.
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

        model.resize({{static_cast<int>(token_ids.size())},
                      {static_cast<int>(spans.size()), 2}});
        model.feed_int64("sents", token_ids.data());
        model.feed_int64("spaninfo", span_values.data());
        model.run();
        std::vector<int> shape;
        probabilities = model.fetch_float(&shape);
        if (probabilities.size() != spans.size())
            throw std::runtime_error("span recognizer returned an invalid probability tensor");
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
        if (this->model.load(modelpath) != EXIT_SUCCESS) {
            std::cerr << "load model " << MODEL_PATH_KEY << " from path " << modelpath << " failed!" << std::endl;
            return EXIT_FAILURE;
        }

        return EXIT_SUCCESS;
    }

    void addWords(const AtomList& dstSrc, SegPath& cmap) const override {
        std::vector<std::pair<size_t, size_t>> spans;
        std::vector<float> probabilities;
        try {
            predictSpans(dstSrc, spans, probabilities);
        } catch (const std::exception& error) {
            std::cerr << "MNN span recognizer inference failed: " << error.what() << std::endl;
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
    ~MnnRecongnizer() {}
};

const char* MnnRecongnizer::MODEL_PATH_KEY  = "model.path";
const char* MnnRecongnizer::WORDPIECE_PARAM = "wordpiece.name";
const char* MnnRecongnizer::MAX_SPAN_KEY     = "max.span";
const char* MnnRecongnizer::THRESHOLD_KEY    = "threshold";

REGISTER_Recognizer(MnnRecongnizer);

class MnnSyntaxRecongnizer : public CellRecognizer {
   private:
    static const char* MODEL_PATH_KEY;
    static const char* LABEL_PATH_KEY;
    static const char* LABEL_THRESHOLD_KEY;
    static const char* WORDPIECE_PARAM;
    static const char* MAX_SPAN_KEY;
    static const char* THRESHOLD_KEY;

    size_t max_span = 5;
    float threshold = 0.5f;
    // 双头因子化模型（2026-10-25）：threshold* 门 成词概率 σ(w)；
    // label.threshold 门 联合标签概率 σ(w)·softmax(l)。旧单输出模型
    // threshold* 仍门 联合 argmax 概率（行为不变）。
    bool factorized = false;
    float label_threshold = 0.0f;
    std::shared_ptr<WordPice> wordpiece;
    std::vector<std::string> labels;
    MnnModel model;

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
        auto label_option = params.find(LABEL_THRESHOLD_KEY);
        try {
            if (label_option != params.end()) label_threshold = std::stof(label_option->second);
        } catch (const std::exception& error) {
            std::cerr << "invalid " << LABEL_THRESHOLD_KEY << ": " << error.what() << std::endl;
            return EXIT_FAILURE;
        }
        if (label_threshold < 0.0f || label_threshold > 1.0f) return EXIT_FAILURE;

        auto dependency = plugins.find(WORDPIECE_PARAM);
        if (dependency == plugins.end()) return EXIT_FAILURE;
        wordpiece = std::dynamic_pointer_cast<WordPice>(dependency->second);
        if (!wordpiece) return EXIT_FAILURE;

        auto label_path = params.find(LABEL_PATH_KEY);
        auto model_path = params.find(MODEL_PATH_KEY);
        if (label_path == params.end() || model_path == params.end() ||
            loadLabels(label_path->second)) return EXIT_FAILURE;
        const std::string resolved = getResource(model_path->second);
        if (model.load(resolved) != EXIT_SUCCESS) return EXIT_FAILURE;
        factorized = model.has_output("word_probabilities");
        return EXIT_SUCCESS;
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
            model.resize({{static_cast<int>(token_ids.size())},
                          {static_cast<int>(spans.size()), 2}});
            model.feed_int64("sents", token_ids.data());
            model.feed_int64("spaninfo", span_values.data());
            model.run();
            std::vector<int> shape;
            const std::vector<float> flat = model.fetch_float(&shape);
            if (shape.size() != 2 || shape[0] != static_cast<int>(spans.size()) ||
                shape[1] != static_cast<int>(labels.size()))
                throw std::runtime_error("invalid syntax probability tensor");
            const float* probabilities = flat.data();
            std::vector<float> word_probabilities;
            if (factorized) word_probabilities = model.fetch_float_named("word_probabilities");
            Cursor cursor = path.Head();
            for (size_t span = 0; span < spans.size(); ++span) {
                const float* row = probabilities + span * labels.size();
                if (factorized) {
                    // 成词门：σ(w)（threshold* / 按长度）；标签=联合 argmax；
                    // 联合标签门：σ(w)·softmax(l) ≥ label.threshold
                    const float word_p = word_probabilities[span];
                    if (!std::isfinite(word_p) || word_p < threshold) continue;
                    const size_t type =
                        std::max_element(row + 1, row + labels.size()) - row;
                    const float joint_p = row[type];
                    if (!std::isfinite(joint_p) || joint_p < label_threshold) continue;
                    auto word = std::make_shared<Word>(atoms, spans[span].first, spans[span].second);
                    word->addLabel(labels[type]);
                    word->addLabel("NEURAL_" + labels[type]);
                    cursor = path.addCell(word, cursor);
                    continue;
                }
                const size_t type = std::max_element(row, row + labels.size()) - row;
                if (type == 0 || !std::isfinite(row[type]) || row[type] < threshold) continue;
                auto word = std::make_shared<Word>(atoms, spans[span].first, spans[span].second);
                word->addLabel(labels[type]);
                // 神经提名标记：embed 时优先取（与训练 reader 金标节点
                // type=金标词性同源），词典合并候选不受影响
                word->addLabel("NEURAL_" + labels[type]);
                cursor = path.addCell(word, cursor);
            }
        } catch (const std::exception& error) {
            std::cerr << "MNN syntax recognizer inference failed: " << error.what() << std::endl;
        }
    }

    ~MnnSyntaxRecongnizer() {}
};

const char* MnnSyntaxRecongnizer::LABEL_THRESHOLD_KEY = "label.threshold";
const char* MnnSyntaxRecongnizer::MODEL_PATH_KEY = "model.path";
const char* MnnSyntaxRecongnizer::LABEL_PATH_KEY = "label.path";
const char* MnnSyntaxRecongnizer::WORDPIECE_PARAM = "wordpiece.name";
const char* MnnSyntaxRecongnizer::MAX_SPAN_KEY = "max.span";
const char* MnnSyntaxRecongnizer::THRESHOLD_KEY = "threshold";
REGISTER_Recognizer(MnnSyntaxRecongnizer);

}  // namespace darts
#endif  // SRC_IMPL_NETWORDQI_HPP_
