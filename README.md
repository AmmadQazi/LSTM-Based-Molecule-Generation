# AI and Life Science: Molecule Generation Challenge

This repository contains the source code, data, and model checkpoints for the **Molecule Generation Challenge** as part of the AI and Life Science course. The aim of this challenge is to generate molecules using AI techniques, with a focus on SMILES (Simplified Molecular Input Line Entry System) data preprocessing and deep learning models.

## Folder Structure

- **`.ipynb_checkpoints/`**: Checkpoints for the Jupyter notebook during the molecule generation process.
- **`logs/`**: Logs from the model training process, including details on loss, accuracy, and other metrics.
- **`model_checkpoints/`**: Checkpoints of the trained model saved at different epochs.
- **`resources/`**: Additional resources and dependencies used for the challenge.

## Files

- **`Challenge_2.ipynb`**: Jupyter notebook containing the code for preprocessing SMILES data, training the model, and generating molecules.
- **`Challenge2.pptx`**: A PowerPoint presentation providing a detailed overview of the project and the results.
- **`evaluation.tar`**: Evaluation data or scripts used to assess the model’s performance on the generated molecules.
- **`K12343657_Molecule_Generation.zip`**: A ZIP archive containing important data or supplementary files for the molecule generation challenge.
- **`sample_submission.txt`**: A sample submission file illustrating the format and structure for submitting results.
- **`smiles_train.txt`**: The dataset containing SMILES strings used to train the model.

## Getting Started

### Prerequisites

To run this project, ensure that you have the following installed:

- Python 3.x
- Jupyter Notebook
- RDKit (for SMILES normalization and tokenization)
- PyTorch (for building and training the model)

### Running the Notebook

1. Clone the repository:
   ```bash
   git clone <repository-url>
   ```
   
2. Navigate to the project directory:
   ```bash
   cd <repository-directory>
   ```
   
3. Open the Jupyter notebook:
   ```bash
   jupyter notebook Challenge_2.ipynb
   ```

4. Run the notebook cells to preprocess the SMILES data, train the model, and generate molecules.

## Model Details

### Preprocessing

- SMILES strings are normalized using RDKit.
- Tokenization involves converting SMILES into a sequence of character indices for use in the model.

### Model Architecture

The model uses an LSTM (Long Short-Term Memory) network to capture dependencies within the sequence data, followed by a fully connected layer for character prediction.

- **Embedding Layer**: Converts each character index into a high-dimensional vector representation.
- **LSTM Layer**: Processes the sequences with two stacked LSTM layers.
- **Fully Connected Layer**: Maps the LSTM output to the SMILES character space.

### Training

- **Loss Function**: Cross-entropy loss.
- **Optimizer**: Adam with a learning rate of 0.001.
- **Batch Size**: 32.
- **Epochs**: 12 epochs.

### Outputs

- Generated molecules based on the model predictions.
- Evaluation metrics such as QED score, validity, novelty, and uniqueness of the generated molecules.

## Results

- **Total Generated Molecules**: 11,200.
- **Final Selection**: 10,000 molecules selected based on the QED score.
- **Model Performance**: 
  - FCD = 0.919 
  - Novelty = 0.982 
  - Uniqueness = 1.0 
  - Validity = 1.0

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Thanks to the AI and Life Science course for providing the challenge.
- Special mention to RDKit and PyTorch communities for providing essential libraries.