import re
from torch.utils.data import Dataset, DataLoader
import pickle
import random
from tqdm import tqdm

def spa_to_wayu_dictionary(spanish_word, max_matches=5):
    dictionary_path = 'assets/spanish_to_wayuunaiki_short.csv'

    with open(dictionary_path, 'r', encoding='utf-8') as f:
        all_matches = []
        line = f.readline()
        while line != '' and len(all_matches) < max_matches:
            data = line.strip().split(',')
            if re.search(rf'\b{re.escape(spanish_word)}\b', data[0], re.IGNORECASE):
                all_matches.append(data)
            line = f.readline()

    if len(all_matches) > 0:
        result = " <matches> " + '\n'.join(f'{spa}: {wayuu}' for spa, wayuu in all_matches) + " </matches>"
    else:
        result = " <matches> No matches found </matches>"

    return result

class TextDataset(Dataset):
    def __init__(self, spa_path, wayuu_path):
        with open(spa_path, 'r', encoding='utf-8') as f:
            self.spa_lines = [line.strip() for line in f if line.strip()]

        with open(wayuu_path, 'r', encoding='utf-8') as f:
            self.wayuu_lines = [line.strip() for line in f if line.strip()]

    def __len__(self):
        return len(self.spa_lines)

    def __getitem__(self, idx):
        spa = self.spa_lines[idx]
        wayuu = self.wayuu_lines[idx]
        
        return spa, wayuu

spanish_train_file = 'datasets/train.es.txt'
wayuu_train_file = 'datasets/train.guc.txt'
batch_size = 1000
dataset_size_to_create = 59715
steps = dataset_size_to_create / batch_size

# Load the dataset
dataset = TextDataset(spanish_train_file, wayuu_train_file)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

translate_prompt_template_tool="""Translate the following Spanish text into Wayuunaiki.
Begin by identifying any words or phrases you're unsure how to translate. Then, you may look up those words using the dictionary tool by wrapping the Spanish word in <spa_to_wayuu> and </spa_to_wayuu>,
and doind that for every unknown word. The dictionary will return matches enclosed in <matches> and </matches>. You can use the dictionary as many times as necessary.
Once you have all the information you need, provide the final translation enclosed in <answer> and </answer>. For example: <answer> xxx </answer>.

Spanish text: {}"""

# Create the dataset
dataset = []
i = 0
for (spa_batch, wayuu_batch) in tqdm(dataloader):
    if i > steps:
        break
    for spa, wayuu in zip(spa_batch, wayuu_batch):
        prompt = translate_prompt_template_tool.format(spa)

        # Make random searches in the dictionary, between 0 and 4
        num_searches = random.randint(0, 4)
        spa_words = spa.split()
        random.shuffle(spa_words)
        answer = ''
        if len(spa_words) >= num_searches:
            words_to_translate = spa_words[:num_searches]
            for word in words_to_translate:
                translation = spa_to_wayu_dictionary(word)
                answer += f' <spa_to_wayuu> {word} </spa_to_wayuu>' + translation
        answer += f' <answer> {wayuu} </answer>'

        dataset.append((prompt, answer))

    i += 1

# Save the dataset
with open('datasets/sft_dataset2.pkl', 'wb') as f:
    pickle.dump(dataset, f)
print(f"Dataset created with {len(dataset)} samples.")