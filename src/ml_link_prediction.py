"""
Machine learning-based link prediction module for network graphs.

This module provides functions for link prediction using machine learning
approaches, extracting features from network structure and training classifiers.
"""

import logging
import pandas as pd
import numpy as np
import networkx as nx
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report
)
from sklearn.model_selection import cross_val_score, KFold

# Import link prediction functions for feature extraction
from link_prediction import (
    common_neighbors_score,
    jaccard_coefficient_score,
    adamic_adar_score
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def extract_edge_features(
    G: nx.Graph,
    edge_list: List[Tuple[int, int]]
) -> pd.DataFrame:
    """
    Extract features for a list of edges.
    
    For each edge (u, v), extracts:
    - Common neighbors count
    - Jaccard coefficient
    - Adamic-Adar score
    - Degree of node u
    - Degree of node v
    - Product of degrees
    - Sum of degrees
    - Clustering coefficient of u
    - Clustering coefficient of v
    
    Args:
        G: NetworkX graph
        edge_list: List of (u, v) tuples representing edges
    
    Returns:
        DataFrame with feature columns and edge information
    
    Raises:
        TypeError: If G is not a NetworkX graph
        ValueError: If edge_list is empty
    """
    if not isinstance(G, (nx.Graph, nx.DiGraph, nx.MultiGraph, nx.MultiDiGraph)):
        raise TypeError(f"Expected NetworkX graph, got {type(G)}")
    
    if not edge_list:
        raise ValueError("edge_list is empty")
    
    logger.info(f"Extracting features for {len(edge_list)} edges...")
    
    # Calculate similarity-based features
    cn_scores = common_neighbors_score(G, edge_list)
    jc_scores = jaccard_coefficient_score(G, edge_list)
    aa_scores = adamic_adar_score(G, edge_list)
    
    # Extract node-based features
    features = []
    
    for idx, (u, v) in enumerate(tqdm(edge_list, desc="Extracting features")):
        feature_dict = {}
        
        # Common neighbors, Jaccard, Adamic-Adar
        feature_dict['common_neighbors'] = cn_scores[idx]
        feature_dict['jaccard_coefficient'] = jc_scores[idx]
        feature_dict['adamic_adar'] = aa_scores[idx]
        
        # Node degrees
        degree_u = G.degree(u) if u in G else 0
        degree_v = G.degree(v) if v in G else 0
        feature_dict['degree_u'] = degree_u
        feature_dict['degree_v'] = degree_v
        feature_dict['degree_product'] = degree_u * degree_v
        feature_dict['degree_sum'] = degree_u + degree_v
        
        # Clustering coefficients
        try:
            clustering_u = nx.clustering(G, u) if u in G else 0.0
        except:
            clustering_u = 0.0
        
        try:
            clustering_v = nx.clustering(G, v) if v in G else 0.0
        except:
            clustering_v = 0.0
        
        feature_dict['clustering_u'] = clustering_u
        feature_dict['clustering_v'] = clustering_v
        
        features.append(feature_dict)
    
    # Create DataFrame
    df = pd.DataFrame(features)
    
    # Add edge information
    df.insert(0, 'node1', [u for u, v in edge_list])
    df.insert(1, 'node2', [v for u, v in edge_list])
    
    logger.info(f"Feature extraction complete: {len(df)} edges, {len(df.columns)-2} features")
    
    return df


def prepare_training_data(
    G_train: nx.Graph,
    pos_edges: List[Tuple[int, int]],
    neg_edges: List[Tuple[int, int]]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prepare training data from positive and negative edges.
    
    Extracts features for both positive and negative edges, combines them,
    and creates feature matrix X and label vector y.
    
    Args:
        G_train: Training graph
        pos_edges: List of positive edges (label=1)
        neg_edges: List of negative edges (label=0)
    
    Returns:
        Tuple of (X_train, y_train) where:
            - X_train: Feature matrix (numpy array)
            - y_train: Label vector (numpy array)
    
    Raises:
        TypeError: If inputs are not of correct types
        ValueError: If edge lists are empty
    """
    if not isinstance(G_train, (nx.Graph, nx.DiGraph, nx.MultiGraph, nx.MultiDiGraph)):
        raise TypeError(f"Expected NetworkX graph, got {type(G_train)}")
    
    if not pos_edges or not neg_edges:
        raise ValueError("pos_edges and neg_edges must not be empty")
    
    logger.info(f"Preparing training data: {len(pos_edges)} positive, {len(neg_edges)} negative edges")
    
    # Extract features for positive edges
    logger.info("Extracting features for positive edges...")
    pos_features = extract_edge_features(G_train, pos_edges)
    
    # Extract features for negative edges
    logger.info("Extracting features for negative edges...")
    neg_features = extract_edge_features(G_train, neg_edges)
    
    # Combine features (exclude node1 and node2 columns)
    feature_columns = [col for col in pos_features.columns if col not in ['node1', 'node2']]
    
    X_pos = pos_features[feature_columns].values
    X_neg = neg_features[feature_columns].values
    
    # Combine and create labels
    X_train = np.vstack([X_pos, X_neg])
    y_train = np.array([1] * len(pos_edges) + [0] * len(neg_edges))
    
    logger.info(f"Training data prepared: {X_train.shape[0]} samples, {X_train.shape[1]} features")
    
    return X_train, y_train


def train_random_forest(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_estimators: int = 100,
    random_state: int = 42,
    scale_features: bool = True
) -> Tuple[RandomForestClassifier, Optional[StandardScaler]]:
    """
    Train Random Forest classifier for link prediction.
    
    Args:
        X_train: Feature matrix
        y_train: Label vector
        n_estimators: Number of trees in the forest (default: 100)
        random_state: Random seed for reproducibility (default: 42)
        scale_features: Whether to scale features (default: True)
    
    Returns:
        Tuple of (trained_model, scaler) where scaler is None if scaling is disabled
    
    Raises:
        ValueError: If inputs are invalid
    """
    if X_train.size == 0 or len(y_train) == 0:
        raise ValueError("Training data is empty")
    
    if X_train.shape[0] != len(y_train):
        raise ValueError("X_train and y_train must have same number of samples")
    
    logger.info(f"Training Random Forest (n_estimators={n_estimators})...")
    
    # Scale features if requested
    scaler = None
    if scale_features:
        logger.info("Scaling features...")
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
    else:
        X_train_scaled = X_train
    
    # Train model
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=-1,  # Use all available cores
        verbose=0
    )
    
    model.fit(X_train_scaled, y_train)
    
    logger.info(f"Model trained successfully")
    logger.info(f"Training accuracy: {model.score(X_train_scaled, y_train):.4f}")
    
    return model, scaler


def predict_links_ml(
    model: RandomForestClassifier,
    G: nx.Graph,
    edge_list: List[Tuple[int, int]],
    scaler: Optional[StandardScaler] = None
) -> pd.DataFrame:
    """
    Predict link probabilities using trained ML model.
    
    Args:
        model: Trained Random Forest model
        G: NetworkX graph
        edge_list: List of (u, v) tuples representing edges to predict
        scaler: Optional StandardScaler (if features were scaled during training)
    
    Returns:
        DataFrame with columns: [node1, node2, probability, prediction]
        Sorted by probability descending
    
    Raises:
        TypeError: If inputs are not of correct types
        ValueError: If edge_list is empty
    """
    if not isinstance(model, RandomForestClassifier):
        raise TypeError(f"Expected RandomForestClassifier, got {type(model)}")
    
    if not edge_list:
        raise ValueError("edge_list is empty")
    
    logger.info(f"Predicting links for {len(edge_list)} edges...")
    
    # Extract features
    features_df = extract_edge_features(G, edge_list)
    feature_columns = [col for col in features_df.columns if col not in ['node1', 'node2']]
    X = features_df[feature_columns].values
    
    # Scale if scaler provided
    if scaler is not None:
        X = scaler.transform(X)
    
    # Get predictions
    probabilities = model.predict_proba(X)[:, 1]  # Probability of class 1 (link exists)
    predictions = model.predict(X)
    
    # Create result DataFrame
    result_df = pd.DataFrame({
        'node1': features_df['node1'].values,
        'node2': features_df['node2'].values,
        'probability': probabilities,
        'prediction': predictions
    })
    
    # Sort by probability
    result_df = result_df.sort_values('probability', ascending=False)
    
    logger.info(f"Predictions complete: {np.sum(predictions)} positive predictions")
    
    return result_df


def evaluate_ml_model(
    model: RandomForestClassifier,
    G: nx.Graph,
    pos_test: List[Tuple[int, int]],
    neg_test: List[Tuple[int, int]],
    scaler: Optional[StandardScaler] = None
) -> Tuple[Dict[str, float], pd.DataFrame]:
    """
    Evaluate ML model performance on test data.
    
    Args:
        model: Trained Random Forest model
        G: NetworkX graph
        pos_test: List of positive test edges
        neg_test: List of negative test edges
        scaler: Optional StandardScaler (if features were scaled during training)
    
    Returns:
        Tuple of (metrics_dict, predictions_df) where:
            - metrics_dict: Dictionary with evaluation metrics
            - predictions_df: DataFrame with predictions and probabilities
    
    Raises:
        TypeError: If inputs are not of correct types
        ValueError: If test edge lists are empty
    """
    if not isinstance(model, RandomForestClassifier):
        raise TypeError(f"Expected RandomForestClassifier, got {type(model)}")
    
    if not pos_test or not neg_test:
        raise ValueError("pos_test and neg_test must not be empty")
    
    logger.info(f"Evaluating ML model: {len(pos_test)} positive, {len(neg_test)} negative test edges")
    
    # Get predictions
    all_test_edges = pos_test + neg_test
    predictions_df = predict_links_ml(model, G, all_test_edges, scaler=scaler)
    
    # Create true labels
    true_labels = np.array([1] * len(pos_test) + [0] * len(neg_test))
    pred_labels = predictions_df['prediction'].values
    probabilities = predictions_df['probability'].values
    
    # Calculate metrics
    precision = precision_score(true_labels, pred_labels, zero_division=0)
    recall = recall_score(true_labels, pred_labels, zero_division=0)
    f1 = f1_score(true_labels, pred_labels, zero_division=0)
    
    try:
        auc_roc = roc_auc_score(true_labels, probabilities)
    except Exception as e:
        logger.warning(f"Could not calculate AUC-ROC: {e}")
        auc_roc = np.nan
    
    try:
        auc_pr = average_precision_score(true_labels, probabilities)
    except Exception as e:
        logger.warning(f"Could not calculate AUC-PR: {e}")
        auc_pr = np.nan
    
    # Confusion matrix
    cm = confusion_matrix(true_labels, pred_labels)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    
    metrics = {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc_roc': auc_roc,
        'auc_pr': auc_pr,
        'true_positives': int(tp),
        'true_negatives': int(tn),
        'false_positives': int(fp),
        'false_negatives': int(fn),
        'accuracy': (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
    }
    
    logger.info(f"Evaluation complete: F1={f1:.4f}, AUC-ROC={auc_roc:.4f}, AUC-PR={auc_pr:.4f}")
    
    return metrics, predictions_df


def get_feature_importance(
    model: RandomForestClassifier,
    feature_names: List[str]
) -> pd.DataFrame:
    """
    Extract and sort feature importances from Random Forest model.
    
    Args:
        model: Trained Random Forest model
        feature_names: List of feature names
    
    Returns:
        DataFrame with columns: [feature, importance] sorted by importance descending
    
    Raises:
        TypeError: If model is not RandomForestClassifier
        ValueError: If feature_names length doesn't match model features
    """
    if not isinstance(model, RandomForestClassifier):
        raise TypeError(f"Expected RandomForestClassifier, got {type(model)}")
    
    if len(feature_names) != model.n_features_in_:
        raise ValueError(f"Number of feature names ({len(feature_names)}) doesn't match "
                        f"model features ({model.n_features_in_})")
    
    importances = model.feature_importances_
    
    df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    })
    
    df = df.sort_values('importance', ascending=False)
    
    logger.info("Feature importances extracted")
    
    return df


def cross_validate_model(
    G: nx.Graph,
    all_edges: List[Tuple[int, int]],
    labels: np.ndarray,
    k: int = 5,
    n_estimators: int = 100,
    random_state: int = 42,
    scale_features: bool = True
) -> Dict[str, float]:
    """
    Perform k-fold cross-validation for link prediction model.
    
    Args:
        G: NetworkX graph
        all_edges: List of all edges (positive and negative)
        labels: Array of labels (1 for positive, 0 for negative)
        k: Number of folds (default: 5)
        n_estimators: Number of trees in Random Forest (default: 100)
        random_state: Random seed (default: 42)
        scale_features: Whether to scale features (default: True)
    
    Returns:
        Dictionary with average metrics across folds:
            - 'mean_accuracy': Mean accuracy
            - 'std_accuracy': Standard deviation of accuracy
            - 'mean_f1': Mean F1 score
            - 'std_f1': Standard deviation of F1
            - 'mean_auc_roc': Mean AUC-ROC
            - 'std_auc_roc': Standard deviation of AUC-ROC
    
    Raises:
        ValueError: If inputs are invalid
    """
    if len(all_edges) != len(labels):
        raise ValueError("all_edges and labels must have same length")
    
    if k < 2:
        raise ValueError(f"k must be at least 2, got {k}")
    
    logger.info(f"Performing {k}-fold cross-validation...")
    
    # Extract features for all edges
    features_df = extract_edge_features(G, all_edges)
    feature_columns = [col for col in features_df.columns if col not in ['node1', 'node2']]
    X = features_df[feature_columns].values
    y = labels
    
    # Scale features if requested
    if scale_features:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
    
    # Initialize model
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=-1
    )
    
    # Perform cross-validation
    kf = KFold(n_splits=k, shuffle=True, random_state=random_state)
    
    accuracies = []
    f1_scores = []
    auc_rocs = []
    
    for fold, (train_idx, val_idx) in enumerate(tqdm(kf.split(X), total=k, desc="Cross-validation")):
        X_train_fold, X_val_fold = X[train_idx], X[val_idx]
        y_train_fold, y_val_fold = y[train_idx], y[val_idx]
        
        # Train model
        model.fit(X_train_fold, y_train_fold)
        
        # Evaluate
        y_pred = model.predict(X_val_fold)
        y_proba = model.predict_proba(X_val_fold)[:, 1]
        
        acc = (y_pred == y_val_fold).mean()
        f1 = f1_score(y_val_fold, y_pred, zero_division=0)
        
        try:
            auc_roc = roc_auc_score(y_val_fold, y_proba)
        except:
            auc_roc = np.nan
        
        accuracies.append(acc)
        f1_scores.append(f1)
        auc_rocs.append(auc_roc)
    
    # Calculate statistics
    results = {
        'mean_accuracy': np.mean(accuracies),
        'std_accuracy': np.std(accuracies),
        'mean_f1': np.mean(f1_scores),
        'std_f1': np.std(f1_scores),
        'mean_auc_roc': np.nanmean(auc_rocs),
        'std_auc_roc': np.nanstd(auc_rocs)
    }
    
    logger.info(f"Cross-validation complete: Mean F1={results['mean_f1']:.4f}, "
               f"Mean AUC-ROC={results['mean_auc_roc']:.4f}")
    
    return results


def main():
    """
    Main function demonstrating usage of the ML link prediction module.
    """
    print("=" * 60)
    print("ML Link Prediction Module - Demo")
    print("=" * 60)
    
    try:
        import networkx as nx
        from preprocessing import create_train_test_split
        
        # Create a sample graph
        print("\n1. Creating sample graph...")
        G = nx.erdos_renyi_graph(n=1000, p=0.01, seed=42)
        print(f"   Created graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        
        # Create train/test split
        print("\n2. Creating train/test split...")
        G_train, pos_test, neg_test = create_train_test_split(G, test_ratio=0.2, seed=42)
        print(f"   Train: {G_train.number_of_edges()} edges")
        print(f"   Test: {len(pos_test)} positive + {len(neg_test)} negative edges")
        
        # Prepare training data
        print("\n3. Preparing training data...")
        X_train, y_train = prepare_training_data(G_train, pos_test[:500], neg_test[:500])
        print(f"   Training data: {X_train.shape[0]} samples, {X_train.shape[1]} features")
        
        # Train model
        print("\n4. Training Random Forest model...")
        model, scaler = train_random_forest(X_train, y_train, n_estimators=50, random_state=42)
        print("   ✅ Model trained")
        
        # Get feature importance
        feature_names = ['common_neighbors', 'jaccard_coefficient', 'adamic_adar',
                        'degree_u', 'degree_v', 'degree_product', 'degree_sum',
                        'clustering_u', 'clustering_v']
        importance_df = get_feature_importance(model, feature_names)
        print("\n5. Feature Importances:")
        print(importance_df.to_string(index=False))
        
        # Evaluate model
        print("\n6. Evaluating model on test set...")
        metrics, predictions = evaluate_ml_model(
            model, G_train, pos_test[:100], neg_test[:100], scaler=scaler
        )
        print("   Evaluation metrics:")
        for key, value in metrics.items():
            if isinstance(value, float):
                print(f"     {key}: {value:.4f}")
            else:
                print(f"     {key}: {value}")
        
        print("\n" + "=" * 60)
        print("Demo completed successfully!")
        print("=" * 60)
    
    except Exception as e:
        logger.error(f"Demo failed: {e}", exc_info=True)
        print(f"\n✗ Error: {e}")
        raise


if __name__ == "__main__":
    main()

