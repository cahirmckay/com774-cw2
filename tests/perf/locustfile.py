from locust import HttpUser, task, between
import json
import locust_env

class CW2LoadTest(HttpUser):
    wait_time = between(1, 3)
    host = locust_env.HOST 

    def on_start(self):
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {locust_env.API_KEY}"
        }

    @task
    def predict(self):
        payload = {
            "data": [
                [0.11, -0.42, 1.23, -0.88, 0.56, -1.12, 0.34, 1.08, -0.59, 0.21]
            ]
        }
        self.client.post(
            "/score", 
            data=json.dumps(payload),
            headers=self.headers
        )

