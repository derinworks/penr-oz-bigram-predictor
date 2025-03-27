import time
import torch
from requests import Response
from torch.nn.functional import one_hot
import requests

# Prepare Prediction server config
prediction_server_url = "http://127.0.0.1:8000"
model_request = {
    "model_id": "bigram-predictor"
}

def get_one_hot_encoded(idx: int, total: int) -> list[float]:
    return one_hot(torch.tensor(idx, dtype=torch.int64), total).float().tolist()

def request_prediction_progress() -> Response:
    progress_resp = requests.get(f"{prediction_server_url}/progress/", params=model_request)
    progress_status, progress_body = progress_resp.status_code, progress_resp.json()
    print(f"{progress_status=}")
    if progress_resp.status_code == 200:
        if len(progress_body["progress"]) > 0:
            costs = [progress["cost"] for progress in progress_body["progress"]]
            cost = sum(costs) / len(costs)
            avg_cost = progress_body["average_cost"]
            print(f"{cost=}")
            print(f"{avg_cost=}")
    else: # barf possible error body
        print(f"{progress_body=}")
    return progress_resp

def make_prediction(input_vector: list[float]) -> list[float]:
    prediction_request = model_request | {
        "input": {
            "activation_vector": input_vector
        },
    }
    resp = requests.post(f"{prediction_server_url}/output/", json=prediction_request)

    if resp.status_code == 200:
        return resp.json()['output_vector']

    raise RuntimeError(f"Failed to receive a good prediction: {resp.status_code} - {resp.json()}")

if __name__ == "__main__":
    # User selection
    user_selection = input('Choose (S) generate sample or (T) perform training:').upper()
    print(f"{user_selection=}")

    # Read example in
    with open("example.txt", "r", encoding="utf-8") as f:
        example = f.read()

    # Extract tokens
    tokens = sorted(set(ch for ch in example.lower() if ch.isalpha()))
    num_tokens = len(tokens)
    s2i = {s: i for i, s in enumerate(tokens)}

    # Create prediction model if not already
    model_resp = request_prediction_progress()
    if model_resp.status_code == 404:
        create_model_request = model_request | {
            "layer_sizes": [num_tokens, num_tokens],
            "weight_algo": "he",
            "bias_algo": "zeros",
            "activation_algos": ["softmax"],
        }
        create_model_resp = requests.post(f"{prediction_server_url}/model/", json=create_model_request)
        print(f"{create_model_resp.status_code} - {create_model_resp.json()}")
    elif model_resp.status_code != 200:
        raise RuntimeError(f"Prediction Service error: {model_resp.status_code} - {model_resp.json()}")

    # Perform according to user selection
    if user_selection == 'T':
        # Build training data
        training_data = []
        for sp in zip(example, example[1:]):
            if all(s in s2i.keys() for s in sp):
                training_data.append(tuple(get_one_hot_encoded(s2i[s], num_tokens) for s in sp))

        # Prepare training request parameters
        train_model_request = model_request | {
            "epochs": 10,
            "learning_rate": 0.1,
            "decay_rate": 0.99,
            "dropout_rate": 0.0,
            "l2_lambda": 0.0,
            "adam_beta1": 0.9,
            "adam_beta2": 0.999,
            "adam_epsilon": 1e-8,
        }

        # Prepare training request
        training_request = train_model_request | {
            "training_data": [{
                "activation_vector": input_vector,
                "target_vector": target_vector,
            } for input_vector, target_vector in training_data],
        }
        print(f"Prepared training data of size {len(training_data)}...")

        # Ask for training length
        num_trainings = int(input('How many times shall we perform training?'))
        print(f"{num_trainings=}")

        for i in range(num_trainings):
            # Submit training request to prediction service
            training_resp = requests.put(f"{prediction_server_url}/train/", json=training_request)
            print(f"Submitted: {training_resp.status_code} - {training_resp.json()}")
            # wait a bit
            time.sleep(1)
            # check progress
            request_prediction_progress()
            # mark end of training request
            print(f"###### Finished Training Round {i+1} of {num_trainings} ########")

    else: # Generate sample
        # Build reverse lookup
        i2s = {i: s for i, s in enumerate(tokens)}

        # Select token
        token_idx: int = torch.randint(0, num_tokens, (1,)).item()

        # Ask for sample length
        sample_len = int(input('How long shall the sample be?'))
        print(f"{sample_len=}")

        # Build prediction
        sample = i2s[token_idx]
        for _ in range(sample_len - 1):
            # Predict next token
            output_vector = make_prediction(get_one_hot_encoded(token_idx, num_tokens))
            output_idx: int = torch.multinomial(torch.tensor(output_vector), num_samples=1).item()
            # Append next token
            sample += i2s[output_idx]
            # Set next token as current for next iteration
            token_idx = output_idx

        # Present sample
        print(f"{sample=}")
