import numpy as np

from textattack.shared import AttackedText
from textattack.transformations import WordSwap


class QueryAugmentation:
    def __init__(self, 
        augmentation_size=1, 
        transformations=[],
        constraints=[], 
    ):
        self.augmentation_size = augmentation_size
        self.transformations = transformations
        self.constraints = constraints

        self.transformation = MixedTransformations(
            transformations = self.transformations,
            augmentation_size=self.augmentation_size
        )

    def augment(self, text):
        attacked_text = AttackedText(text)
        transformed_texts = self.transformation(attacked_text, self.constraints)
        perturbed_texts = [transformed_text.printable_text() for transformed_text in transformed_texts]
        return perturbed_texts


class MixedTransformations(WordSwap):
    def __init__(self, transformations, augmentation_size=1):
        super().__init__()
        self.transformations = transformations
        self.augmentation_size = augmentation_size
        self.transformation_id = 0

    def _get_transformations(self, current_text, indices_to_modify):
        indices_to_modify = list(indices_to_modify)
        words = current_text.words
        if len(indices_to_modify) == 0:
            return [current_text for _ in range(self.augmentation_size)]

        transformed_texts = []
        for _ in range(self.augmentation_size):
            # Calculate sampling probabilities
            candidate_words = [words[word_id].lower() for word_id in indices_to_modify]
            # Sampling word to transform
            id_to_transform = np.random.choice(len(candidate_words))
            word_to_transform = candidate_words[id_to_transform]
            # Get list of transformed words
            trial = 0
            transformed_words = []
            while len(transformed_words) == 0:
                for transformed_word in self.transformations[self.transformation_id]._get_replacement_words(word_to_transform):
                    if transformed_word == word_to_transform:
                        # Skip current transformation
                        self.transformation_id = (self.transformation_id + 1) % len(self.transformations)
                        continue
                    transformed_words.append(transformed_word)
                trial += 1
                if trial >= 10:
                    transformed_words.append(word_to_transform)
            transformed_word = np.random.choice(transformed_words)
            transformed_word_id = indices_to_modify[id_to_transform]
            # Update transformation_id
            self.transformation_id = (self.transformation_id + 1) % len(self.transformations)
            # Get transformed_text
            transformed_text = current_text.replace_word_at_index(transformed_word_id, transformed_word)
            transformed_texts.append(transformed_text)
        return transformed_texts