Firstly install OpenPrompt https://github.com/thunlp/OpenPrompt
Then copy prompts/knowledgeable_verbalizer.py, pipeline_base.py and text_classification_dataset.py to Openprompt/openprompt/prompts/

Next, you can obtain the synonyms of the category label test from the Chinese synonym website, and use the method in Strateg to filter and optimize the expansion words。
The expanded tags are then added to the appropriate verbalizer file in scripts.
example shell scripts:

python fewshot.py --result_file ./output_fewshot.txt --dataset wechat --template_id 0 --seed 123 --shot 10 --verbalizer manual


Note that the file paths should be changed according to the running environment. 

The datasets are downloadable via OpenPrompt.
