import unittest

import torch

from wan.modules.causal_model_packforcing import CausalWanModelPackForcing
from wan.modules.pack_cache import (
    IdentityCompressor,
    PackBlockRegistry,
    PACK_CACHE_MODE_FINALIZED_CHUNK_ONLY,
    PACK_CACHE_MODE_ROLLOUT_COMPATIBLE,
    commit_fullres_block,
    init_empty_pack_layer_cache,
    should_update_pack_cache,
)


class PackForcingCacheSemanticsTest(unittest.TestCase):
    def test_should_update_pack_cache_by_mode(self) -> None:
        self.assertTrue(
            should_update_pack_cache(
                grad_enabled=False,
                cache_mode=PACK_CACHE_MODE_ROLLOUT_COMPATIBLE,
                pack_cache_commit=False,
            )
        )
        self.assertFalse(
            should_update_pack_cache(
                grad_enabled=False,
                cache_mode=PACK_CACHE_MODE_FINALIZED_CHUNK_ONLY,
                pack_cache_commit=False,
            )
        )
        self.assertTrue(
            should_update_pack_cache(
                grad_enabled=False,
                cache_mode=PACK_CACHE_MODE_FINALIZED_CHUNK_ONLY,
                pack_cache_commit=True,
            )
        )
        self.assertFalse(
            should_update_pack_cache(
                grad_enabled=True,
                cache_mode=PACK_CACHE_MODE_ROLLOUT_COMPATIBLE,
                pack_cache_commit=True,
            )
        )

    def test_cache_sessions_keep_state_isolated(self) -> None:
        model = CausalWanModelPackForcing(
            model_type="t2v",
            patch_size=(1, 2, 2),
            text_len=8,
            in_dim=16,
            dim=64,
            ffn_dim=128,
            freq_dim=16,
            text_dim=64,
            out_dim=16,
            num_heads=4,
            num_layers=2,
            pack_enable=True,
            pack_compress_mode="identity",
        )
        handle_rollout = model.create_pack_cache_handle(PACK_CACHE_MODE_ROLLOUT_COMPATIBLE)
        handle_finalized = model.create_pack_cache_handle(PACK_CACHE_MODE_FINALIZED_CHUNK_ONLY)

        session_rollout = model._require_pack_cache_session(
            handle_rollout.session_id,
            handle_rollout.mode,
        )
        session_finalized = model._require_pack_cache_session(
            handle_finalized.session_id,
            handle_finalized.mode,
        )

        token_count = session_rollout.layer_cache[0].cfg.frame_seq_len
        k = torch.zeros(1, token_count, 4, 16)
        v = torch.zeros(1, token_count, 4, 16)
        latent = torch.zeros(1, 16, 1, 2, 2)
        for start_frame in (0, 3, 6):
            commit_fullres_block(
                session_rollout.layer_cache[0],
                k,
                v,
                start_frame=start_frame,
                compressor=IdentityCompressor(),
            )
            model._commit_source_latent_block(
                session_rollout.source_state,
                latent,
                start_frame=start_frame,
            )

        self.assertIs(session_rollout.layer_cache[0].registry, session_rollout.source_state.registry)
        self.assertEqual(session_rollout.source_state.total_committed_blocks, 3)
        self.assertEqual(len(session_rollout.source_state.recent_blocks), 1)
        self.assertEqual(len(session_rollout.source_state.mid_blocks), 0)
        self.assertEqual(session_rollout.layer_cache[0].total_committed_blocks, 3)

        self.assertEqual(session_finalized.source_state.total_committed_blocks, 0)
        self.assertEqual(len(session_finalized.source_state.recent_blocks), 0)
        self.assertEqual(len(session_finalized.source_state.mid_blocks), 0)
        self.assertEqual(session_finalized.layer_cache[0].total_committed_blocks, 0)

        model.reset_pack_cache_session(handle_rollout)
        self.assertEqual(session_rollout.source_state.total_committed_blocks, 0)
        self.assertEqual(session_rollout.layer_cache[0].total_committed_blocks, 0)
        self.assertEqual(session_finalized.source_state.total_committed_blocks, 0)
        self.assertEqual(session_finalized.layer_cache[0].total_committed_blocks, 0)

    def test_shared_registry_reconciles_mid_transition_for_all_layers(self) -> None:
        model = CausalWanModelPackForcing(
            model_type="t2v",
            patch_size=(1, 2, 2),
            text_len=8,
            in_dim=16,
            dim=64,
            ffn_dim=128,
            freq_dim=16,
            text_dim=64,
            out_dim=16,
            num_heads=4,
            num_layers=2,
            pack_enable=True,
            pack_compress_mode="identity",
            pack_sink_blocks=2,
            pack_recent_blocks=1,
            pack_mid_bank_capacity_blocks=3,
        )
        registry = PackBlockRegistry()
        layer_cfg = model.pack_cfg.build_cache_config(
            frame_seq_len=1560,
            num_frame_per_block=model.num_frame_per_block,
        )
        layer0 = init_empty_pack_layer_cache(layer_cfg, registry)
        layer1 = init_empty_pack_layer_cache(layer_cfg, registry)

        k = torch.zeros(1, layer_cfg.frame_seq_len, 4, 16)
        v = torch.zeros(1, layer_cfg.frame_seq_len, 4, 16)
        compressor = IdentityCompressor()

        for start_frame in (0, 1, 2, 3):
            commit_fullres_block(layer0, k, v, start_frame=start_frame, compressor=compressor)
            commit_fullres_block(layer1, k, v, start_frame=start_frame, compressor=compressor)

        self.assertEqual(registry.total_committed_blocks, 4)
        self.assertEqual(len(layer0.mid_blocks), 1)
        self.assertEqual(len(layer1.mid_blocks), 1)
        self.assertEqual(layer0.mid_blocks[0].meta.start_frame, 2)
        self.assertEqual(layer1.mid_blocks[0].meta.start_frame, 2)


if __name__ == "__main__":
    unittest.main()
