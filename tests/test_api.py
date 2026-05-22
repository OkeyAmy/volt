import importlib.util
import unittest


FASTAPI_AVAILABLE = importlib.util.find_spec("fastapi") is not None


@unittest.skipUnless(FASTAPI_AVAILABLE, "fastapi is not installed in this environment")
class ApiSmokeTest(unittest.TestCase):
    def setUp(self):
        from fastapi.testclient import TestClient

        from app.config import get_settings
        from app.main import ServiceContainer, create_app

        class FakeRating:
            def predict_from_features(self, _features):
                return 4.2

        class FakeCounterfactual:
            def run_counterfactuals(self, _features):
                return {
                    "original_rating": 4.2,
                    "counterfactuals": [],
                    "regret_risk": 0.1,
                    "robustness": 0.9,
                }

        class FakeReview:
            def generate(self, persona, product, rating, counterfactuals):
                return {
                    "rating": 4,
                    "review": "Works well for the persona.",
                    "reasoning_summary": "Strong predicted fit.",
                }

        class FakeRecommendation:
            def recommend(self, persona, top_k=10):
                return [
                    {
                        "product_id": "prod_1",
                        "product_name": "Sample product",
                        "category": "Electronics",
                        "predicted_rating": 4.2,
                        "ranker_score": 0.5,
                        "regret_risk": 0.1,
                        "robustness": 0.9,
                        "final_score": 0.88,
                        "reason": "Predicted 4.2/5.",
                    }
                ][:top_k]

        container = ServiceContainer(
            settings=get_settings(),
            rating_service=FakeRating(),
            counterfactual_service=FakeCounterfactual(),
            recommendation_service=FakeRecommendation(),
            review_generator=FakeReview(),
        )
        self.client = TestClient(create_app(container))

    def persona(self):
        return {
            "budget_sensitivity": 0.8,
            "service_sensitivity": 0.7,
            "quality_sensitivity": 0.9,
            "strictness": 0.6,
            "tone": 2,
            "review_length": 50,
        }

    def product(self):
        return {
            "product_name": "Sample durable product",
            "category": "Electronics",
            "description": "A compact device with strong battery life.",
            "quality_signal": 0.7,
            "service_signal": -0.2,
            "value_signal": 0.1,
            "usability_signal": 0.9,
            "price_level": 1,
            "aspect_quality": 1,
            "aspect_price": 0,
            "aspect_service": -1,
            "aspect_value": 0,
            "aspect_usability": 1,
            "aspect_delivery": -1,
            "product_text": "Works well and feels durable.",
        }

    def test_health(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "ok"})

    def test_generate_review(self):
        response = self.client.post(
            "/task-a/generate-review",
            json={"persona_features": self.persona(), "product_features": self.product()},
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["predicted_rating"], 4.2)

    def test_generate_review_accepts_simple_payload(self):
        response = self.client.post(
            "/generate-review",
            json={
                "persona": "Budget conscious Nigerian student who values durability",
                "product": {
                    "name": "20,000mAh power bank",
                    "category": "Electronics",
                    "description": "Affordable power bank with strong battery backup.",
                },
            },
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["generated_review"]["rating"], 4.0)

    def test_recommend(self):
        response = self.client.post(
            "/task-b/recommend",
            json={"persona_features": self.persona(), "top_k": 1},
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["recommendations"][0]["product_id"], "prod_1")
        self.assertEqual(response.json()["recommendations"][0]["category"], "Electronics")

    def test_recommend_accepts_simple_payload(self):
        response = self.client.post(
            "/recommend",
            json={
                "persona": "Budget conscious student who needs durable electronics",
                "top_k": 1,
            },
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["recommendations"][0]["product_id"], "prod_1")


if __name__ == "__main__":
    unittest.main()
