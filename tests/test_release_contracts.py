"""Dependency-light release boundaries; never starts GPUs or network calls."""
import json
from pathlib import Path
import unittest
from optiatlas.paths import ROOT, load_settings, resolve
from optiatlas.cli import prepared_config


class ReleaseContracts(unittest.TestCase):
    def setUp(self):
        self.settings=load_settings(ROOT/"configs/paths.example.json")

    def test_rl_is_loaded_on_sft_base(self):
        cfg=prepared_config("train-rl","policy_rl95",self.settings)
        self.assertEqual(cfg["actor_rollout_ref"]["model"]["path"],self.settings["paths"]["policy_sft_base"])
        self.assertNotEqual(cfg["actor_rollout_ref"]["model"]["path"],self.settings["paths"]["policy_raw_base"])
        self.assertEqual(cfg["optics_agent_rl"]["reward_contract_version"],"prompt_physics_product_v1")

    def test_final_and_strict_are_distinct_configs(self):
        final=prepared_config("train-generator","generator_final",self.settings)
        strict=prepared_config("train-generator","generator_strict_agent",self.settings)
        self.assertNotEqual(final["output_dir"],strict["output_dir"])
        self.assertNotEqual(final["release_dataset"],strict["release_dataset"])
        self.assertEqual(final["reward"]["component_normalization"],"prompt_group")
        self.assertEqual(final["finetune"]["lora"],{"rank":32,"alpha":64,"dropout":0})
        self.assertEqual(final["upstream"]["data_epochs"],2)

    def test_evaluation_uses_exp53_and_native_protocol(self):
        cfg=prepared_config("generate","evaluation_final",self.settings)
        self.assertTrue(cfg["benchmark"]["image"]["state_sha256"].startswith("9ecae447"))
        self.assertEqual(cfg["benchmark"]["image"]["cfg_scale"],4)
        self.assertEqual(cfg["benchmark"]["image"]["num_steps"],40)
        self.assertEqual(cfg["benchmark"]["budget"]["max_image_calls"],3)
        self.assertEqual(cfg["benchmark"]["actor"]["temperature"],0)

    def test_missing_paths_fail(self):
        with self.assertRaises(ValueError):resolve("${paths.missing}",self.settings)

    def test_all_selected_recipes_resolve(self):
        for path in (ROOT/"configs").glob("*.json"):
            if path.stem.startswith("generator_"):
                prepared_config("train-generator",path.stem,self.settings)
            elif path.stem.startswith("policy_rl"):
                prepared_config("train-rl",path.stem,self.settings)
            elif path.stem.startswith("evaluation_"):
                prepared_config("generate",path.stem,self.settings)

    def test_strict_views_use_identical_ordered_image_source(self):
        dense=prepared_config("train-generator","generator_strict_dense",self.settings)
        agent=prepared_config("train-generator","generator_strict_agent",self.settings)
        self.assertEqual(dense["data"]["path"],agent["data"]["path"])
        self.assertEqual(dense["release_dataset"],{"variant":"agent","caption_field":"dense_caption"})
        self.assertEqual(agent["release_dataset"],{"variant":"agent","caption_field":"agentv4_rl_prompt"})

    def test_no_machine_paths_in_configs(self):
        for path in (ROOT/"configs").glob("*.json"):
            self.assertNotIn("/mnt/",path.read_text(),str(path))


if __name__=="__main__":unittest.main()
