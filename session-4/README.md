# EMLOPS Session 4

### Projekt Dog Breed Klassifier
---

## Features

1. **Dockerfile**: The project includes a Dockerfile to containerize train, eval and infer environment.
2. **DevContainer**: A development container is provided for easy setup and consistent development experience.
3. **Pre-trained Model Evaluation**: `eval.py` script to evaluate the model on the validation dataset.
4. **Inference Script**: `infer.py` script to run inference on a sample of 10 images.
5. **Dataset Handling**: Kaggle API integration to download the Dog Breed Image Dataset.
6. **Volume Mounts**: Ensures data persistence while running Docker containers.

---

## Dataset

We use the [Dog Breed Image Dataset](https://www.kaggle.com/datasets/khushikhushikhushi/dog-breed-image-dataset). The dataset can be downloaded using the Kaggle API. 

Follow [Kaggle API documentation](https://www.kaggle.com/docs/api#interacting-with-datasets) for authentication and dataset download.

---

## Project Structure

```
.
├── docker-compose.yaml
├── __init__.py
├── model-eval
│   ├── Dockerfile.eval
│   ├── eval.py
│   └── logs
│       └── eval_log.log
├── model-infer
│   ├── Dockerfile.infer
│   └── infer.py
├── model-train
│   ├── Dockerfile.train
│   ├── __init__.py
│   ├── logs
│   │   └── train_log.log
│   └── train.py
├── pyproject.toml
├── README.md
├── src
│   ├── checkpoints
│   ├── datamodules
│   │   ├── catdog_datamodule.py
│   │   ├── dogbreed_datamodule.py
│   │   └── __init__.py
│   ├── infer
│   │   └── logs
│   │       └── infer_log.log
│   ├── __init__.py
│   ├── logs
│   ├── models
│   │   ├── catdog_classifier.py
│   │   └── __init__.py
│   └── utils
│       ├── __init__.py
│       ├── logging_utils.py
│       └── split_dataset.py
└── uv.lock

```

---

## Setup Instructions

### Prerequisites

1. Install [Docker](https://docs.docker.com/get-docker/).
2. Clone this repository:

   ```bash
   git clone <repository_url>
   cd <repository_name>
   ```

4. Place your `kaggle.json` (Kaggle API token) in the project directory.

---

### Building the Docker Image

Build the Docker image with the following command:

```bash
docker build -t dog-breed-classifier .
```

---

## Running the Project

### Train the Model

Use Docker to train the model with the following command:

```bash
docker run --rm -v $(pwd)/data:/app/data dog-breed-classifier python src/train.py
```

- **Volume Mounts**: `-v $(pwd)/data:/app/data` ensures the dataset and output files are stored persistently on your local system.

### Evaluate the Model

To evaluate the model on the validation dataset:

```bash
docker run --rm -v $(pwd)/data:/app/data dog-breed-classifier python eval.py
```

This will print the validation metrics.

### Run Inference

To run inference on 10 images:

```bash
docker run --rm -v $(pwd)/data:/app/data dog-breed-classifier python infer.py
```

---

### Using Docker Compose

A `docker-compose.yml` file is included for easier management of the containerized environment. Use the following commands:

#### Build and Start the Services

```bash
docker-compose up --build
```

This command builds the Docker image and starts the services defined in `docker-compose.yml`.

#### Train the Model Using Docker Compose

```bash
docker-compose run app python src/train.py
```

#### Evaluate the Model Using Docker Compose

```bash
docker-compose run app python eval.py
```

#### Run Inference Using Docker Compose

```bash
docker-compose run app python infer.py
```

#### Stop the Services

```bash
docker-compose down
```

---

## Additional Notes

- **DataModule**: The `datamodule.py` script is used to load and preprocess the dataset.
- **Model Checkpoints**: Ensure that trained model checkpoints are stored in the `data/` directory for evaluation and inference.
- **DevContainer**: Open the project in a compatible IDE (e.g., VS Code) with DevContainer support for seamless development.

---

## Contributing

Feel free to fork this repository and submit pull requests to contribute to the project.

