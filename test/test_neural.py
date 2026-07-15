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

    def test_word_encoder_attention_pools_every_position(self):
        encoder = self.WordEncoder(vocab_num=32, hidden_size=16, wtype_num=-1, num_layers=1,
                                   num_heads=4, max_positions=32)
        token_ids = torch.tensor([[1, 2, 3, 4]])
        lengths = torch.tensor([4])
        words = torch.tensor([[0, 0, 3]])
        output = encoder(token_ids, lengths, words)
        output[0, 0].backward()
        gradients = encoder.vocab_embeding[0].weight.grad[token_ids[0]].abs().sum(dim=1)
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


if __name__ == "__main__":
    unittest.main()
