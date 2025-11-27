from locust import HttpUser, task, between
import json
import locust_env

class CW2LoadTest(HttpUser):
    wait_time = between(1, 3)

    def on_start(self):
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {locust_env.API_KEY}"
        }

    @task
    def predict(self):
        payload = {
            "data": [
                [0.12, 0.03, 5, 1, 10, 0.2, 0, 1, 3, 0.1]
            ]
        }
        self.client.post(
            locust_env.ENDPOINT_URL,
            data=json.dumps(payload),
            headers=self.headers,
            name="sinlge_prediction"
        )
