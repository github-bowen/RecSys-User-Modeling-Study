import numpy as np

def calculate_success_rate_at_k(ranked_items, true_item, k=10):
    """Calculate Success Rate@k (Hit Rate@k).
    
    Args:
        ranked_items: List of item IDs, sorted by recommendation score (highest first)
        true_item: The ground truth item ID
        k: Cutoff threshold
        
    Returns:
        1 if true_item appears in the top-k items, 0 otherwise
    """
    return int(true_item in ranked_items[:k])

def calculate_mrr(ranked_items, true_item):
    """Calculate Mean Reciprocal Rank.
    
    Args:
        ranked_items: List of item IDs, sorted by recommendation score (highest first)
        true_item: The ground truth item ID
        
    Returns:
        Reciprocal of the rank of the true item, 0 if not found
    """
    if true_item in ranked_items:
        rank = ranked_items.index(true_item) + 1  # +1 because indices start at 0
        return 1.0 / rank
    return 0.0

def calculate_precision_at_k(ranked_items, true_item, k=10):
    """Calculate Precision@k.
    
    Args:
        ranked_items: List of item IDs, sorted by recommendation score (highest first)
        true_item: The ground truth item ID
        k: Cutoff threshold
        
    Returns:
        Precision@k score
    """
    # Since there is only one relevant item in our leave-one-out evaluation,
    # Precision@k is 1/k if the item is in the top-k, otherwise 0
    if true_item in ranked_items[:k]:
        return 1.0 / k
    return 0.0

def evaluate_recommendations(recommender, test_data, train_data, all_items, k=10):
    """Evaluate a recommender system on test data.
    
    Args:
        recommender: Trained recommender object
        test_data: Test data with user-item pairs
        train_data: Training data
        all_items: List of all item IDs
        k: Cutoff threshold
        
    Returns:
        Dictionary with average metrics: 'success_rate', 'mrr', 'precision'
    """
    success_rate_list = []
    mrr_list = []
    precision_list = []
    
    for _, row in test_data.iterrows():
        user_id = row['user_id']
        true_item = row['item_id']
        
        # Generate 100 negative samples plus the true item
        user_rated_items = train_data[train_data['user_id'] == user_id]['item_id'].values
        negative_items = list(set(all_items) - set(user_rated_items) - {true_item})
        
        # Sample 100 negative items
        if len(negative_items) > 100:
            sampled_negatives = np.random.choice(negative_items, size=100, replace=False)
        else:
            sampled_negatives = np.random.choice(negative_items, size=100, replace=True)
        
        # Combine with the true item
        test_items = [true_item] + sampled_negatives.tolist()
        
        # Get ranking from recommender
        ranked_items = recommender.rank(user_id, test_items)
        
        # Calculate metrics
        sr = calculate_success_rate_at_k(ranked_items, true_item, k)
        mrr = calculate_mrr(ranked_items, true_item)
        precision = calculate_precision_at_k(ranked_items, true_item, k)
        
        success_rate_list.append(sr)
        mrr_list.append(mrr)
        precision_list.append(precision)
    
    # Calculate average metrics
    avg_success_rate = np.mean(success_rate_list)
    avg_mrr = np.mean(mrr_list)
    avg_precision = np.mean(precision_list)
    
    return {
        'success_rate': avg_success_rate,
        'mrr': avg_mrr,
        'precision': avg_precision
    } 