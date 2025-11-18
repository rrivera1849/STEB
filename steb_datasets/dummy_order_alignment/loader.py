import numpy as np

def load_dummy_order_alignment_dataset(path: str):
    """
    Loads a dummy dataset for testing order alignment tasks.

    This dataset provides multiple test cases, each with:
    - ordered_texts: A reference set of texts ordered along a stylistic dimension
    - unordered_texts: A test set of texts that need to be aligned to the reference order
    - true_indices: The true order indices of the unordered texts

    Returns a list of records, each containing these three components.
    """
    records = []

    # First set: Formality dimension
    ordered_texts_1 = [
        "hey whats up lol",  # Very informal
        "hi there, how are you doing?",  # Informal
        "Hello, how are you today?",  # Neutral
        "Good afternoon, I hope you are well.",  # Formal
        "Greetings, I trust this message finds you in good health.",  # Very formal
    ]

    unordered_texts_1_original = [
        "yo what's going on",  # Very informal (position 0)
        "hey, how's it going?",  # Informal (position 1)
        "Hello, how are you?",  # Neutral (position 2)
        "Good day, how do you do?",  # Formal (position 3)
        "Salutations, I hope this finds you well.",  # Very formal (position 4)
    ]

    true_indices_1 = [0, 1, 2, 3, 4]
    np.random.seed(42)
    shuffle_1 = np.random.permutation(len(unordered_texts_1_original))
    unordered_texts_1 = [unordered_texts_1_original[i] for i in shuffle_1]
    shuffled_indices_1 = [true_indices_1[i] for i in shuffle_1]

    records.append({
        "ordered_texts": ordered_texts_1,
        "unordered_texts": unordered_texts_1,
        "true_indices": shuffled_indices_1,
    })

    # Second set: Complexity dimension (simple to complex sentences)
    ordered_texts_2 = [
        "I run.",  # Very simple
        "She runs fast.",  # Simple
        "The dog runs in the park.",  # Medium
        "The large dog runs quickly through the green park.",  # Complex
        "The exceptionally large dog runs remarkably quickly through the beautifully maintained green park.",  # Very complex
    ]

    unordered_texts_2_original = [
        "We walk.",  # Very simple (position 0)
        "He walks slowly.",  # Simple (position 1)
        "The cat walks on the street.",  # Medium (position 2)
        "The small cat walks carefully across the busy street.",  # Complex (position 3)
        "The incredibly small cat walks extraordinarily carefully across the dangerously busy street.",  # Very complex (position 4)
    ]

    true_indices_2 = [0, 1, 2, 3, 4]
    np.random.seed(43)
    shuffle_2 = np.random.permutation(len(unordered_texts_2_original))
    unordered_texts_2 = [unordered_texts_2_original[i] for i in shuffle_2]
    shuffled_indices_2 = [true_indices_2[i] for i in shuffle_2]

    records.append({
        "ordered_texts": ordered_texts_2,
        "unordered_texts": unordered_texts_2,
        "true_indices": shuffled_indices_2,
    })

    return records