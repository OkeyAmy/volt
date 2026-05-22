import sys
import types
import unittest
from contextlib import contextmanager

from app.agents.review_generator import ReviewGenerationError, ReviewGenerator
from app.services.counterfactual_service import CounterfactualService
from app.services.rating_service import RatingService
from app.services.recommendation_service import RecommendationService


@contextmanager
def patched_modules(**modules):
    previous = {name: sys.modules.get(name) for name in modules}
    sys.modules.update(modules)
    try:
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = value


class FakeNumpy:
    @staticmethod
    def clip(value, lower, upper):
        return max(lower, min(upper, value))


class FakePandas:
    @staticmethod
    def DataFrame(rows):
        return rows


class FakeRatingModel:
    def __init__(self, prediction):
        self.prediction = prediction

    def predict(self, _frame):
        return [self.prediction]


class FakeLowClassifier:
    def predict_proba(self, frame):
        return [[0.95, 0.05]]


class RatingServiceTest(unittest.TestCase):
    def test_predict_from_features_clamps_model_output(self):
        service = RatingService(
            model=FakeRatingModel(8.2),
            feature_cols=["quality_signal", "price_level"],
            low_classifier=FakeLowClassifier(),
            low_threshold=0.75,
            high_threshold=0.90,
        )

        with patched_modules(
            numpy=FakeNumpy,
            pandas=FakePandas,
        ):
            self.assertEqual(service.predict_from_features({"quality_signal": 0.7}), 5.0)

    def test_text_complaints_adjust_serving_features(self):
        base = {
            "quality_signal": 0.7,
            "service_signal": 0.1,
            "value_signal": 0.1,
            "usability_signal": 0.9,
            "strictness": 0.2,
            "text": "battery stopped working and seller refused refund",
        }

        adjusted = RatingService._augment_with_text_signals(base)

        self.assertLess(adjusted["quality_signal"], base["quality_signal"])
        self.assertLess(adjusted["service_signal"], base["service_signal"])
        self.assertGreater(adjusted["strictness"], base["strictness"])


class StepRatingService:
    def __init__(self):
        self.calls = 0

    def predict_from_features(self, features):
        self.calls += 1
        if features.get("service_signal") == -0.5:
            return 3.0
        if features.get("value_signal") == -0.5:
            return 3.5
        return 4.0


class CounterfactualServiceTest(unittest.TestCase):
    def test_counterfactuals_compute_regret_and_robustness(self):
        service = CounterfactualService(StepRatingService())

        result = service.run_counterfactuals(
            {
                "price_level": 1,
                "service_signal": 0.0,
                "usability_signal": 0.0,
                "value_signal": 0.0,
                "quality_signal": 0.0,
            }
        )

        self.assertEqual(result["original_rating"], 4.0)
        self.assertEqual(result["regret_risk"], 0.3)
        self.assertEqual(result["robustness"], 0.7)
        self.assertEqual(len(result["counterfactuals"]), 5)


class FakeProduct(dict):
    pass


class FakeCatalog:
    def __init__(self, rows):
        self.rows = rows

    def iterrows(self):
        return iter(enumerate(self.rows))


class FakeRanker:
    def predict(self, frame):
        row = frame[0]
        return [row["quality_signal"] * 10]


class ConstantRatingService:
    def predict_from_features(self, features):
        return 4.0 if features["quality_signal"] < 0.9 else 5.0


class ConstantCounterfactualService:
    def run_counterfactuals(self, _features):
        return {"regret_risk": 0.1, "robustness": 0.9}


class RecommendationServiceTest(unittest.TestCase):
    def test_recommendation_uses_normalized_ranker_scores(self):
        service = RecommendationService(
            rating_service=ConstantRatingService(),
            counterfactual_service=ConstantCounterfactualService(),
            catalog=FakeCatalog(
                [
                    FakeProduct(
                        product_id="low",
                        product_name="Low score",
                        category="Test category",
                        quality_signal=0.1,
                        service_signal=0,
                        value_signal=0,
                        usability_signal=0,
                        price_level=1,
                        aspect_quality=0,
                        aspect_price=0,
                        aspect_service=0,
                        aspect_value=0,
                        aspect_usability=0,
                        aspect_delivery=0,
                    ),
                    FakeProduct(
                        product_id="high",
                        product_name="High score",
                        category="Test category",
                        quality_signal=1.0,
                        service_signal=0,
                        value_signal=0,
                        usability_signal=0,
                        price_level=1,
                        aspect_quality=0,
                        aspect_price=0,
                        aspect_service=0,
                        aspect_value=0,
                        aspect_usability=0,
                        aspect_delivery=0,
                    ),
                ]
            ),
            ranker=FakeRanker(),
            ranker_cols=["quality_signal"],
        )

        with patched_modules(pandas=FakePandas):
            results = service.recommend(
                {
                    "budget_sensitivity": 0.5,
                    "service_sensitivity": 0.5,
                    "quality_sensitivity": 0.5,
                    "strictness": 0.5,
                    "tone": 2,
                    "review_length": 50,
                },
                top_k=2,
            )

        self.assertEqual([item["product_id"] for item in results], ["high", "low"])
        self.assertEqual(results[0]["category"], "Test category")
        self.assertLessEqual(results[0]["final_score"], 1.5)
        self.assertNotIn("_ranker_raw", results[0])


class ReviewGeneratorTest(unittest.TestCase):
    def test_parse_json_accepts_markdown_fenced_json(self):
        parsed = ReviewGenerator._parse_json(
            '```json\n{"rating": 4, "review": "Solid", "reasoning_summary": "Matched"}\n```'
        )

        result = ReviewGenerator._validate_review_payload(parsed)

        self.assertEqual(result["rating"], 4.0)
        self.assertEqual(result["review"], "Solid")

    def test_validate_review_payload_rejects_bad_rating(self):
        with self.assertRaises(ReviewGenerationError):
            ReviewGenerator._validate_review_payload(
                {"rating": 99, "review": "Bad", "reasoning_summary": "Invalid"}
            )

    def test_validate_review_payload_rejects_empty_review(self):
        with self.assertRaises(ReviewGenerationError):
            ReviewGenerator._validate_review_payload(
                {"rating": 4, "review": " ", "reasoning_summary": "Invalid"}
            )

    def test_generate_falls_back_without_llm_configuration(self):
        settings = types.SimpleNamespace(
            gemini_api_key=None,
            gemini_model_name="unused",
            enable_llm_generation=False,
        )
        generator = ReviewGenerator(settings=settings)

        result = generator.generate(
            persona={
                "budget_sensitivity": 0.8,
                "service_sensitivity": 0.4,
                "quality_sensitivity": 0.9,
                "strictness": 0.6,
                "tone": 3,
            },
            product={
                "product_name": "Budget power bank",
                "category": "Electronics",
            },
            rating=4.2,
            counterfactuals={},
        )

        self.assertEqual(result["rating"], 4.0)
        self.assertIn("Budget power bank", result["review"])


if __name__ == "__main__":
    unittest.main()
