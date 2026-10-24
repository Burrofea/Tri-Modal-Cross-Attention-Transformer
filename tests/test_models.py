"""Unit tests for neural architectures, dynamic gating, cross-attention, and losses."""
import unittest
import torch
import torch.nn as nn

from src.models.tokenizers import AudioTokenizer, VideoTokenizer, RPPGTokenizer
from src.models.gating import DynamicGatingModule
from src.models.cross_attention import CrossAttentionBlock
from src.models.trimodal_transformer import TriModalTransformer
from src.models.rppg_baseline import RPPGBaselineModel
from src.training.losses import MultiTaskConsistencyLoss

class TestModels(unittest.TestCase):
    def setUp(self):
        self.B = 4
        self.H = 192
        self.device = torch.device("cpu")

        # Synthetic batch
        self.batch = {
            "mel": torch.randn(self.B, 64, 500, device=self.device),
            "emb": torch.randn(self.B, 16, 512, device=self.device),
            "tok": torch.randn(self.B, 5, 147, device=self.device),
            "rm": torch.tensor([[1, 1, 1, 1, 1],
                                [1, 1, 0, 0, 0],
                                [0, 0, 0, 0, 0],  # Case A: completely missing face
                                [1, 1, 1, 1, 0]], dtype=torch.bool, device=self.device),
            "gate_in": torch.randn(self.B, 5, 4, device=self.device),
            "y": torch.tensor([[0., 0., 0.],
                               [1., 0., 1.],
                               [0., 1., 1.],
                               [1., 1., 1.]], device=self.device),
        }

    def test_tokenizers(self):
        atok = AudioTokenizer(hidden_dim=self.H, num_tokens=25)
        vtok = VideoTokenizer(in_dim=512, hidden_dim=self.H, num_tokens=16)
        rtok = RPPGTokenizer(in_dim=147, hidden_dim=self.H, num_windows=5)

        a_out = atok(self.batch["mel"])
        v_out = vtok(self.batch["emb"])
        r_out = rtok(self.batch["tok"])

        self.assertEqual(a_out.shape, (self.B, 25, self.H))
        self.assertEqual(v_out.shape, (self.B, 16, self.H))
        self.assertEqual(r_out.shape, (self.B, 5, self.H))

    def test_dynamic_gating(self):
        gate_mod = DynamicGatingModule(in_features=4, hidden_dim=32)
        g = gate_mod(self.batch["gate_in"], self.batch["rm"])

        self.assertEqual(g.shape, (self.B, 5, 1))
        # Ensure values in [0, 1]
        self.assertTrue(torch.all(g >= 0.0) and torch.all(g <= 1.0))

        # Check that completely invalid row (index 2) is strictly 0
        self.assertTrue(torch.all(g[2] == 0.0))
        # Check that partially invalid window (index 1, window 2) is strictly 0
        self.assertEqual(float(g[1, 2, 0].item()), 0.0)

    def test_cross_attention_block(self):
        block = CrossAttentionBlock(embed_dim=self.H, num_heads=4, dropout=0.1)
        query = torch.randn(self.B, 16, self.H)
        context = torch.randn(self.B, 30, self.H)
        bias = torch.zeros(self.B, 1, 1, 30)

        out = block(query, context, bias)
        self.assertEqual(out.shape, (self.B, 16, self.H))
        self.assertFalse(torch.isnan(out).any())

    def test_trimodal_transformer_forward_and_backward(self):
        model = TriModalTransformer(hidden_dim=self.H, num_heads=4, num_layers=2)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        criterion = MultiTaskConsistencyLoss()

        # Stage 1 test
        out_s1 = model(self.batch, stage=1)
        self.assertIn("aux_a", out_s1)
        self.assertIn("aux_v", out_s1)
        self.assertIn("aux_r", out_s1)
        loss_s1 = criterion(out_s1, self.batch["y"], stage=1)
        self.assertFalse(torch.isnan(loss_s1["loss"]))

        # Stage 2 test (Main multi-task cross-attention)
        optimizer.zero_grad()
        out_s2 = model(self.batch, stage=2)
        self.assertIn("logits_video", out_s2)
        self.assertIn("logits_audio", out_s2)
        self.assertIn("logits_any", out_s2)

        self.assertEqual(out_s2["logits_video"].shape, (self.B,))
        self.assertEqual(out_s2["logits_audio"].shape, (self.B,))
        self.assertEqual(out_s2["logits_any"].shape, (self.B,))

        loss_s2 = criterion(out_s2, self.batch["y"], stage=2)
        self.assertFalse(torch.isnan(loss_s2["loss"]))

        loss_s2["loss"].backward()
        optimizer.step()

    def test_trimodal_ablation_modes(self):
        model = TriModalTransformer(hidden_dim=self.H, num_heads=4, num_layers=2)
        model.eval()

        # Test disabling rPPG
        out_no_rppg = model(self.batch, stage=2, disable_modalities=("rppg",))
        self.assertFalse(torch.isnan(out_no_rppg["logits_any"]).any())

        # Test disabling audio
        out_no_audio = model(self.batch, stage=2, disable_modalities=("audio",))
        self.assertFalse(torch.isnan(out_no_audio["logits_any"]).any())

        # Test disabling gating
        out_no_gate = model(self.batch, stage=2, use_dynamic_gating=False)
        self.assertFalse(torch.isnan(out_no_gate["logits_any"]).any())

    def test_rppg_baseline_model(self):
        baseline = RPPGBaselineModel(in_dim=147, hidden_dim=128)
        out = baseline(self.batch)

        self.assertIn("logits_video", out)
        self.assertIn("probabilities", out)
        self.assertEqual(out["probabilities"].shape, (self.B,))
        self.assertTrue(torch.all(out["probabilities"] >= 0.0) and torch.all(out["probabilities"] <= 1.0))

if __name__ == "__main__":
    unittest.main()
