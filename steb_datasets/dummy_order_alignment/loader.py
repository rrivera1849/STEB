import numpy as np

def load_dummy_order_alignment_dataset(path: str):
    """
    Loads a dummy dataset for testing order alignment tasks.

    This dataset provides multiple test cases, each with:
    - ordered_texts: A reference set of texts ordered along a stylistic dimension
    - unordered_texts: A test set of texts that need to be aligned to the reference order
    - distractor_texts: A set of texts that are not part of the ordered sequence
    - true_indices: The true order indices of the unordered texts, with -1 for distractors

    Returns a list of records, each containing these components.
    """
    records = []

    # First set: Formality dimension
    ordered_texts_1 = [
        "hey whats up lol",
        "hi there, how are you doing?",
        "Hello, how are you today?",
        "Good afternoon, I hope you are well.",
        "Greetings, I trust this message finds you in good health.",
    ]
    unordered_texts_1_original = [
        "yo what's going on",
        "hey, how's it going?",
        "Hello, how are you?",
        "Good day, how do you do?",
        "Salutations, I hope this finds you well.",
    ]
    distractor_texts_1 = [
        "this is a distractor",
        "another unrelated sentence",
    ]
    true_indices_1 = [0, 1, 2, 3, 4]

    combined_unordered_texts_1 = unordered_texts_1_original + distractor_texts_1
    combined_true_indices_1 = true_indices_1 + [-1] * len(distractor_texts_1)

    np.random.seed(42)
    shuffle_1 = np.random.permutation(len(combined_unordered_texts_1))

    unordered_texts_1 = [combined_unordered_texts_1[i] for i in shuffle_1]
    true_indices_1_shuffled = [combined_true_indices_1[i] for i in shuffle_1]

    records.append({
        "ordered_texts": ordered_texts_1,
        "unordered_texts": unordered_texts_1,
        "true_indices": true_indices_1_shuffled,
    })

    # Second set: Complexity dimension
    ordered_texts_2 = [
        "I run.",
        "She runs fast.",
        "The dog runs in the park.",
        "The large dog runs quickly through the green park.",
        "The exceptionally large dog runs remarkably quickly through the beautifully maintained green park.",
    ]
    unordered_texts_2_original = [
        "We walk.",
        "He walks slowly.",
        "The cat walks on the street.",
        "The small cat walks carefully across the busy street.",
        "The incredibly small cat walks extraordinarily carefully across the dangerously busy street.",
    ]
    distractor_texts_2 = [
        "a completely random sentence",
        "this one is also a distractor",
    ]
    true_indices_2 = [0, 1, 2, 3, 4]

    combined_unordered_texts_2 = unordered_texts_2_original + distractor_texts_2
    combined_true_indices_2 = true_indices_2 + [-1] * len(distractor_texts_2)

    np.random.seed(43)
    shuffle_2 = np.random.permutation(len(combined_unordered_texts_2))

    unordered_texts_2 = [combined_unordered_texts_2[i] for i in shuffle_2]
    true_indices_2_shuffled = [combined_true_indices_2[i] for i in shuffle_2]

    records.append({
        "ordered_texts": ordered_texts_2,
        "unordered_texts": unordered_texts_2,
        "true_indices": true_indices_2_shuffled,
    })

    return records
