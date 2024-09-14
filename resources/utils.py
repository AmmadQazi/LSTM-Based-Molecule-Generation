
import os
import pkgutil
import tempfile
from rdkit import Chem
from multiprocessing import Pool
import fcd
import numpy as np


def loadmodel():
    chemnet_model_filename = 'ChemNet_v0.13_pretrained.h5'
    model_bytes = pkgutil.get_data('fcd', chemnet_model_filename)

    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = os.path.join(tmpdir, 'chemnet.h5')

        with open(model_path, 'wb') as f:
            f.write(model_bytes)

        print(f'Saved ChemNet model to \'{model_path}\'')

        return fcd.load_ref_model(model_path)


def getstats(smiles, model):
    predictions = fcd.get_predictions(model, smiles)
    mean = predictions.mean(0)
    cov = np.cov(predictions.T)
    return mean, cov

def _cansmi(smi):
    """Try except is needed in case rdkit throws an error"""
    try:
        mol = Chem.MolFromSmiles(smi, sanitize=True)
        can_smi = Chem.MolToSmiles(mol)
    except:
        can_smi = None
    return can_smi


def canonicalize_smiles(smiles, njobs=8):
    r"""calculates canonical smiles
    Arguments:
        smiles (list): List of smiles
        njobs (int): How many workers to use

    Returns:
        canonical_smiles: A list of canonical smiles. None if invalid smiles.
    """

    with Pool(njobs) as pool:
        # pairs of mols and canonical smiles
        canonical_smiles = pool.map(_cansmi, smiles)

    return canonical_smiles

def get_Pareto_fronts(scores):
    """Identify the Pareto fronts from a given set of scores.
    
    Parameters
    ----------
    scores : numpy.ndarray
        An (n_points, n_scores) array of scores.
            
    Returns
    -------
    list of numpy.ndarray
        A list containing the indices of points belonging to each Pareto front.
    """
    
    # Initialize
    population_size = scores.shape[0]
    population_ids = np.arange(population_size)
    all_fronts = []

    # Identify Pareto fronts
    while population_size > 0:
        # Identify the current Pareto front
        pareto_front = np.ones(population_size, dtype=bool)
        for i in range(population_size):
            for j in range(population_size):
                # Strictly j better than i in all scores (i dominated by j) 
                # -> i not in Pareto front
                if all(scores[j] >= scores[i]) and any(scores[j] > scores[i]):
                    pareto_front[i] = 0
                    break

        # Add the current Pareto front to the list of all fronts
        current_front_ids = population_ids[pareto_front]
        all_fronts.append(current_front_ids)

        # Remove the current Pareto front from consideration in future iterations
        scores = scores[~pareto_front]
        population_ids = population_ids[~pareto_front]
        population_size = scores.shape[0]

    return all_fronts