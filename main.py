import random
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

def request_prediction_progress(min_num_progress = 0, timeout_secs = 30) -> Response:
    # keep requesting until condition met or times out
    for _ in range(timeout_secs):
        # wait a second
        time.sleep(1)
        # check progress
        progress_resp = requests.get(f"{prediction_server_url}/progress/", params=model_request)
        progress_status, progress_body = progress_resp.status_code, progress_resp.json()
        print(f"{progress_status=}")
        if progress_resp.status_code == 200:
            if len(progress_body["progress"]) > 0: # log info about progress
                costs = [progress["cost"] for progress in progress_body["progress"]]
                cost = sum(costs) / len(costs)
                avg_cost = progress_body["average_cost"]
                print(f"{cost=}")
                print(f"{avg_cost=}")
            if len(progress_body["progress"]) < min_num_progress:
                continue # checking
        else: # barf possible error body
            print(f"{progress_body=}")
        return progress_resp # done
    # timed out
    raise TimeoutError("Training took too long")

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
    user_selection = input('Choose (S) generate samples or (T) perform training:').upper()
    print(f"{user_selection=}")

    # Read example in
    with open("example.txt", "r", encoding="utf-8") as f:
        example = f.read()
    example = example.lower() # work with lowercase only

    # Extract tokens
    tokens = sorted(list(set(ch for ch in example if ch.isalpha())))
    tokens.insert(0, " ") # space denotes word break
    num_tokens = len(tokens)
    print(f"{num_tokens=}")
    s2i = {s: i for i, s in enumerate(tokens)}

    # Create prediction model if not already
    model_resp = request_prediction_progress()
    if model_resp.status_code == 404:
        create_model_request = model_request | {
            "layer_sizes": [num_tokens, num_tokens],
            "weight_algo": "gaussian",
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
        training_data: list[tuple] = []
        # Start with word break as input
        input_idx = 0
        for s in example:
            label_idx = s2i[s]
            # check no consecutive spaces and token is in vocabulary
            if not all(idx == 0 for idx in [input_idx, label_idx]) and s in s2i.keys():
                # add to encoded input and target label to training data
                enc_input = get_one_hot_encoded(input_idx, num_tokens)
                training_data.append((enc_input, [label_idx]))
                # next input is the label
                input_idx = label_idx

        # Prepare training request parameters
        training_epochs = 10
        train_model_request = model_request | {
            "epochs": training_epochs,
            "learning_rate": 0.005,
            "decay_rate": 0.999,
            "dropout_rate": 0.0,
            "l2_lambda": 0.01,
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
            request_prediction_progress(training_epochs)
            # mark end of training request
            print(f"###### Finished Training Round {i+1} of {num_trainings} ########")

    else: # Generate sample
        # Build reverse lookup
        i2s = {i: s for i, s in enumerate(tokens)}

        # Ask for number of words
        num_samples = int(input('How many samples would you like?'))
        print(f"{num_samples=}")

        # Build prediction
        token_idx = 0
        for _ in range(num_samples):
            # Reset sample
            sample = ""
            # Generate tokens until word break seen for chosen number of words
            while len(sample) < 10: # avoid really long samples
                # Predict next token
                output_vector = make_prediction(get_one_hot_encoded(token_idx, num_tokens))
                output_idx: int = torch.multinomial(torch.tensor(output_vector), num_samples=1).item()
                # Set next token as current for next prediction
                token_idx = output_idx
                # Check word break
                if output_idx == 0:
                    break
                # Append next token
                sample += i2s[output_idx]
            # Present sample
            print(f"{sample=}")
