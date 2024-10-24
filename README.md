# RGB to NIR Predictor

## Requirements

- Python 3.x
- Required Python libraries:
- `h5py`
- `torch`
- `torchvision`
- `torchmetrics`
- `transformers`
- `pytorch_msssim`
- `configargparse`

## Usage

1. Clone the repository:
    ```bash
    git clone https://github.com/Global-Ecosystem-Health-Observatory/NIRPredict.git
    cd NIRPredict
    ```
2. Setup a virtual environment:
    ```bash
    python -m venv venv
    source venv/bin/activate
    ```

3. Install required dependencies (if using `requirements.txt`):
    ```bash
    pip install -r requirements.txt
    ```

4. Ensure you have an h5 file comprising patched aerial images alongside corresponding labels for dead tree locations.

5. Train and evaluate NIR Predict model:
    ```bash
    python -m nirpredict.main /path/to/config.txt
    ```

## Contributing

If you would like to contribute, please fork the repository and create a pull request with your changes. All contributions are welcome!

## License

This project is licensed under the GPLv3 License - see the [LICENSE](LICENSE) file for details.

## Contact

For any questions or feedback, please contact Anis at [aniskhan25@gmail.com].

### Results

1. Model evaluation with different activation functions. The best performing 'relu' is used for TreeMort model training and evaluation.

| Activation Used    | Test Loss  | Mean Absolute Error (MAE) | Peak Signal-to-Noise Ratio (PSNR) | Structural Similarity Index (SSIM) |
| ------------------ | ---------- | ------------------------- | --------------------------------- | ---------------------------------- |
| tanh               | 0.1419     | 0.2609                    | 10.1881 dB                        | 0.5388                             |
| **relu**           | **0.0449** | **0.1140**                | **16.8038 dB**                    | **0.7602**                         |
| scaled sigmoid     | 0.2523     | 0.4031                    | 7.0292 dB                         | 0.4587                             |
| normalized sigmoid | 0.0465     | 0.1178                    | 16.5438 dB                        | 0.7569                             |

2. TreeMort model training and evaluation with predicted NIR band

|                  | NIR              | Mean IOU Pixels | Mean IOU Trees |
| ---------------- | ---------------- | --------------- | -------------- |
| Finland 25cm     | **RGBNIR**       | **0.263**       | **0.774**      |
|                  | RGB              | 0.241           | 0.734          |
|                  | RGBNIR${}_p$     | 0.235           | 0.750          |
| Switzerland 10cm | RGBNIR           | -               | -              |
|                  | RGB              | 0.219           | 0.572          |
|                  | **RGBNIR${}_p$** | **0.246**           | **0.607**          |
