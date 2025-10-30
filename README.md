
## 🚀 关于
# Chinese Clickbait Detection: Dataset and Method

## 📦 Dependencies

- Python 3.7+
- [OpenPrompt](https://github.com/thunlp/OpenPrompt)
- PyTorch
- transformers

To install the dependencies, run the following command in the project root directory:

```bash
pip install -r requirements.txt
```
## 🧩 Project Structure

```bash

SPT-CD/
├── .idea/                      # IDE configuration files
├── ckpts/                      # Model checkpoints
├── Baseline/                   # Baseline models and comparison scripts
├── datasets/                   # Datasets and preprocessing scripts
│   └── HanCD/                  # Chinese clickbait dataset
├── model/                      # Model architecture and components
├── openprompt/                 # OpenPrompt-related implementations
├── prompts/                    # Prompt templates and verbalizers
├── result/                     # Experimental results and logs
├── scripts/             # Script directory
│   └── TextClassification/
│       ├── manual_template.txt
│       ├── cpt_verbalizer.txt
│       └── ...
├── Strategy/                   # Few-shot and zero-shot strategies
│
├── fewshot.py                  # Few-shot learning entry point
├── zeroshot.py                 # Zero-shot learning entry point
├── requirements.txt            # Python dependencies list
└── readme.md                   # Project documentation

```


## 📄 Dataset Information

The project uses the following Chinese clickbait detection datasets, stored in the `datasets/HanCD` directory:

- **HanCD**: This directory contains four sub-datasets:
  - **Paper**
  - **Sina**
  - **Tencent**
  - **Wechat**

Each dataset includes the following files:
- `train.csv`: The labeled training set.
- `test.csv`: The labeled test set.
- `classes.txt`: A file listing the categories (0 for non-clickbait, 1 for clickbait).

The data in each dataset includes the following fields:
- **Label**: The category label (0 for non-clickbait, 1 for clickbait).
- **News Title**: The title of the news article.
- **News Content**: The content of the news article.
- **Source**: The source of the news article (e.g., Paper, Sina, Tencent, Wechat).

Each of these datasets provides the necessary resources for training and evaluating clickbait detection models.


## 🚀 Usage Guide

### 1. Install requirements

First, clone and install OpenPrompt:

```bash
git clone https://github.com/thunlp/OpenPrompt.git
cd OpenPrompt
pip install -e .
```

### 2. Add Custom Components
After installing OpenPrompt, you need to copy the following custom files from the project into the corresponding directories of OpenPrompt:

- prompts/knowledgeable_verbalizer.py
- prompts/pipeline_base.py
- prompts/text_classification_dataset.py

Copy them into the respective directories in OpenPrompt:
```bash
cp prompts/knowledgeable_verbalizer.py OpenPrompt/openprompt/prompts/
cp prompts/pipeline_base.py OpenPrompt/openprompt/prompts/
cp prompts/text_classification_dataset.py OpenPrompt/openprompt/data_utils/

```

### 3. Set Up Prompt Templates and Verbalizers

In the scripts/TextClassification/ directory, you will find subfolders corresponding to each dataset. Each dataset folder contains three essential files:
```bash
- manual_template.txt: Contains the manually set templates.

- manual_verbalizer.txt: Contains the original label words.

- ptuning_template.txt: Contains the soft prompt templates.

- spt_verbalizer.txt: Contains the extended label words.
```
Please make sure these files are configured according to your dataset and task.

Additionally, in the fewshot.py file, you can set different verbalizer mappings. The relevant code is at line 103:

```bash
# Set verbalizer mapping
```
Make sure to adjust this section according to the verbalizer mapping you are using.

### 4. Verbalizer Optimization Strategies
The Strategy folder contains four different verbalizer optimization strategies. Before running the code, you will need to download the following models:

- crawl-300d-2M-subword (https://fasttext.cc/docs/en/english-vectors.html)

- hfl/chinese-bert-wwm (https://huggingface.co/hfl/chinese-bert-wwm)

After downloading, make sure to modify the model paths in the relevant code to point to the correct locations where you have saved the models. These models will be used to extend the label words and stored in the spt_verbalizer.txt file.

### 5. Training and Evaluation
Few-shot learning
```bash
python fewshot.py \
  --result_file ./output_fewshot.txt \
  --dataset wechat \
  --template_id 0 \
  --seed 123 \
  --shot 10 \
  --verbalizer manual
```

Zero-shot learning

```bash
python zeroshot.py \
  --dataset wechat \
  --verbalizer manual
```
Make sure to adjust the file paths as needed.


## 👥 Using Your Own Dataset
(1) Format your dataset to match the style we provided and split it into training and testing sets. Ensure your dataset includes text and label columns, structured as follows:
```bash
label,title,text,source
"clickbait", "211大学高材生22年不与家人联系，掏出身份证时，金华警方都惊了", "这竟然是一张15位数字的“身份证...", "Paper"
"not_clickbait","普及燃气安全知识 筑牢燃气安全防线","为全面提高群众安全意识，深入普及“安全用气”知识...", "Paper"
...
label,title,text,source
"clickbait", "Top student from 211 university hasn't contacted family for 22 years, when showing ID, Jinhua police were shocked", "This turned out to be a 15-digit 'ID card'...", "Paper"
"not_clickbait", "Popularizing gas safety knowledge to strengthen gas safety defenses", "To comprehensively raise public safety awareness and widely promote 'safe gas use' knowledge...", "Paper"

```

(2) In the scripts/TextClassification/ directory, create the necessary files, including verbalizer.txt and template.txt, following our examples. Ensure the categories in verbalizer.txt match those in your dataset. Example content for the verbalizer.txt file:
```bash
clickbait: ["Shocking", "Unbelievable", "This secret"]
not_clickbait: ["Guide", "Explained","Facts"]

```

(3) Modify openprompt/data_utils/text_classification_dataset.py according to the data processing functions we provided. This typically involves preprocessing, splitting, and converting your input data.





## 📌 Notes



We are about to integrate various methods in this field and release an **out-of-the-box toolkit** — stay tuned!

Ensure **OpenPrompt** is correctly installed and that custom components are added to the appropriate directories.

Adjust training parameters such as `shot`, `template_id`, etc., as needed.

Place the dataset in the `datasets/` directory and adjust the paths accordingly.
