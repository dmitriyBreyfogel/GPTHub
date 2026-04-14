import unittest

from app.core.model_metadata import enrich_model_descriptor, infer_model_modalities


class ModelMetadataTests(unittest.TestCase):
    def test_configured_models_map_to_expected_modalities(self) -> None:
        self.assertEqual(["text"], infer_model_modalities("qwen2.5-72b-instruct", None))
        self.assertEqual(["vision"], infer_model_modalities("qwen3-vl-30b-a3b-instruct", None))
        self.assertEqual(["audio"], infer_model_modalities("whisper-turbo-local", None))
        self.assertEqual(["image_generation"], infer_model_modalities("qwen-image-lightning", None))

    def test_embedding_and_tts_models_are_not_marked_as_text(self) -> None:
        self.assertEqual(["embedding"], infer_model_modalities("bge-m3", None))
        self.assertEqual(["tts"], infer_model_modalities("tts-1", None))

    def test_enrich_model_descriptor_adds_modalities_and_capabilities(self) -> None:
        enriched = enrich_model_descriptor(
            {
                "id": "qwen3-vl-30b-a3b-instruct",
                "name": "Qwen Vision",
                "object": "model",
                "owned_by": "mws",
            }
        )

        meta = enriched["info"]["meta"]
        self.assertEqual(["vision"], meta["gpthub_modalities"])
        self.assertTrue(meta["capabilities"]["vision"])
        self.assertTrue(meta["capabilities"]["file_upload"])


if __name__ == "__main__":
    unittest.main()
