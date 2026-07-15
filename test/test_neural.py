import math
import unittest


try:
    import torch
except ImportError:
    torch = None


@unittest.skipUnless(torch is not None, "PyTorch is an optional training dependency")
class NeuralModelTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        from darts.devel.model import JointSegmentationTrainer, Quantizer, SpanRecognizer, WordEncoder
        from darts.devel.utils import GraphLossSparse

        cls.Quantizer = Quantizer
        cls.SpanRecognizer = SpanRecognizer
        cls.WordEncoder = WordEncoder
        cls.GraphLossSparse = GraphLossSparse
        cls.JointSegmentationTrainer = JointSegmentationTrainer

    def test_quantizer_returns_association_negative_log_probability(self):
        model = self.Quantizer(4, 2)
        with torch.no_grad():
            model.Kmap.weight.zero_()
            model.Kmap.bias.copy_(torch.tensor([1.0, 0.0]))
            model.Qmap.weight.zero_()
            model.Qmap.bias.copy_(torch.tensor([1.0, 0.0]))
            model.logit_scale.zero_()
        nll = model(torch.zeros(1, 4), torch.zeros(1, 4)).item()
        expected = -math.log(torch.sigmoid(torch.tensor(1.0)).item())
        self.assertAlmostEqual(nll, expected, places=6)
        self.assertGreaterEqual(nll, 0.0)

    def test_quantizer_initialization_has_usable_probability_range(self):
        model = self.Quantizer(8, 4)
        with torch.no_grad():
            model.Kmap.weight.zero_()
            model.Qmap.weight.zero_()
            model.Kmap.bias.copy_(torch.tensor([1.0, 0.0, 0.0, 0.0]))
            model.Qmap.bias.copy_(torch.tensor([1.0, 0.0, 0.0, 0.0]))
        probability = torch.exp(-model(torch.zeros(1, 8), torch.zeros(1, 8))).item()
        self.assertGreater(probability, 0.8)

    def test_word_encoder_attention_pools_every_position(self):
        encoder = self.WordEncoder(vocab_num=32, hidden_size=16, wtype_num=-1, num_layers=1,
                                   num_heads=4, max_positions=32)
        token_ids = torch.tensor([[1, 2, 3, 4]])
        lengths = torch.tensor([4])
        words = torch.tensor([[0, 0, 3]])
        output = encoder(token_ids, lengths, words)
        output[0, 0].backward()
        gradients = encoder.vocab_embedding[0].weight.grad[token_ids[0]].abs().sum(dim=1)
        self.assertEqual(output.shape, (1, 16))
        self.assertTrue(torch.all(gradients > 0))

    def test_span_recognizer_accepts_overlapping_candidates(self):
        model = self.SpanRecognizer(vocab_num=32, hidden_size=16)
        token_ids = torch.tensor([[1, 2, 3, 4]])
        lengths = torch.tensor([4])
        spans = torch.tensor([
            [0, 0, 1, 2, 1],
            [0, 1, 3, 3, 1],
            [0, 0, 3, 4, 0],
        ])
        loss = model(token_ids, lengths, spans)
        self.assertEqual(loss.ndim, 0)
        self.assertTrue(torch.isfinite(loss))

    def test_sparse_graph_loss_backpropagates_association_nll(self):
        word_info = torch.tensor([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
        graph = torch.tensor([
            [0, 0, 1, 1],
            [0, 0, 2, 0],
            [0, 1, 2, 1],
        ])
        association_nll = torch.tensor([0.2, 1.1, 0.3], requires_grad=True)
        loss = self.GraphLossSparse()(word_info, graph, association_nll)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.all(torch.isfinite(association_nll.grad)))

    def test_joint_tasks_reference_one_encoder(self):
        model = self.JointSegmentationTrainer(vocab_num=32, hidden_size=16, wtype_num=4)
        self.assertIs(model.recognizer.encoder, model.graph_quantizer.predictor)
        self.assertIs(model.encoder, model.recognizer.encoder)

    def test_both_joint_losses_update_shared_encoder(self):
        model = self.JointSegmentationTrainer(vocab_num=32, hidden_size=16, wtype_num=4)
        token_ids = torch.tensor([[1, 2, 3, 4]])
        lengths = torch.tensor([4])
        spans = torch.tensor([[0, 0, 1, 2, 1], [0, 1, 3, 3, 0]])
        words = torch.tensor([[0, 0, 0, 0], [0, 0, 1, 1], [0, 2, 3, 2], [0, 3, 3, 3]])
        graph = torch.tensor([[0, 0, 1, 1], [0, 0, 2, 0], [0, 1, 3, 1], [0, 2, 3, 0]])
        recognizer_loss = model.recognizer(token_ids, lengths, spans)
        quantizer_loss = model.graph_quantizer(token_ids, lengths, words, graph)
        (recognizer_loss + quantizer_loss).backward()
        gradient = model.encoder.vocab_embedding[0].weight.grad
        self.assertIsNotNone(gradient)
        self.assertGreater(float(gradient.abs().sum()), 0.0)


class NeuralReaderTests(unittest.TestCase):

    def _reader_types(self):
        try:
            from darts import PyAtomList
            from darts.devel.reader import GraphSampleReader, piece_bounds
        except (ImportError, OSError) as error:
            self.skipTest(f"native darts training reader unavailable: {error}")
        return PyAtomList, GraphSampleReader, piece_bounds

    def test_piece_bounds_preserves_zero_offset_and_multi_piece_atom(self):
        _atom_list, _reader, piece_bounds = self._reader_types()
        starts, ends = piece_bounds([(10, 0), (11, 0), (12, 1), (2, -2)], 2)
        self.assertEqual(starts, [0, 2])
        self.assertEqual(ends, [1, 2])

    def test_gold_spans_use_sentence_atom_indexes(self):
        atom_list, reader, _piece_bounds = self._reader_types()
        english_atoms = atom_list("ABC DEF", skip_space=True, normal_before=False)
        self.assertEqual(reader._gold_spans(["ABC", "DEF"], english_atoms), [(0, 1), (1, 2)])
        split_atoms = atom_list("ABC 123", skip_space=True, normal_before=False)
        self.assertEqual(reader._gold_spans(["ABC", "123"], split_atoms), [(0, 1), (1, 2)])

    def test_corpus_whitespace_is_not_part_of_atom_indexes(self):
        atom_list, reader, _piece_bounds = self._reader_types()
        tokens = "南京市   长江\t大桥".split()
        atoms = atom_list(" ".join(tokens), skip_space=True, normal_before=False)
        self.assertEqual(reader._gold_spans(tokens, atoms), [(0, 3), (3, 5), (5, 7)])


if __name__ == "__main__":
    unittest.main()
