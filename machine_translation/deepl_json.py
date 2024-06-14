# #!/usr/bin/env python3
# import json
# import argparse
# import pathlib
# import deepl
# import os


# auth_key = "YOUR_KEY_HERE"  
# translator = deepl.Translator(auth_key)


# def translate(data_path, output_path):
#     with open(data_path, "r") as fin, open(output_path, "w") as fo:
        
#         tsv_writer = csv.writer(fo, delimiter='\t')
#         tsv_writer.writerow(["DE", "EN"])

#         for i, line in enumerate(fin.readlines()):
#             line=line.strip()
#             result = translator.translate_text(line, target_lang="EN-US")
#             tsv_writer.writerow([line, result])
            

# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--data',type=pathlib.Path)
#     parser.add_argument('--output',type=pathlib.Path)

#     args = parser.parse_args()
#     translate(args.data, args.output)

# if __name__ == "__main__":
#     main()


#!/usr/bin/env python3
import json
import argparse
import pathlib
import deepl
import os

# Deepl authentication key
auth_key = "190ade38-f097-44ef-b836-8d1cc1be3f9d:fx"
translator = deepl.Translator(auth_key)

def translate_text(text, source_lang="EN", target_lang="ZH"):
    # Translate text from English to Chinese (Simplified)
    cleaned_text = text.strip()
    result = translator.translate_text(cleaned_text, source_lang=source_lang, target_lang=target_lang)
    return result.text


def translate_file(input_file_path, output_file_path, keys_to_translate, source_lang="EN", target_lang="ZH"):
    with open(input_file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)

    if keys_to_translate == ['Statement']:  # Special handling for top-level JSON files
        for record_id in data:
            if 'Statement' in data[record_id]:
                data[record_id]['Statement'] = translate_text(data[record_id]['Statement'], source_lang, target_lang)
    else:  # General handling for nested JSON files
        for key in keys_to_translate:
            if key in data:
                data[key] = [translate_text(text, source_lang, target_lang) for text in data[key]]

    with open(output_file_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)

def process_folder(folder_path, output_folder, is_root=True, source_lang="EN", target_lang="ZH"):
    os.makedirs(output_folder, exist_ok=True)  # Ensure the output directory exists
    for entry in os.listdir(folder_path):
        input_path = os.path.join(folder_path, entry)
        output_path = os.path.join(output_folder, entry)
        if os.path.isdir(input_path):
            # Recursive call for subdirectories
            process_folder(input_path, output_path, is_root=False, source_lang="EN", target_lang="ZH")
        elif input_path.endswith('.json'):
            # Process JSON files
            keys_to_translate = ['Statement'] if is_root else ['Intervention', 'Eligibility', 'Results', 'Adverse Events']
            translate_file(input_path, output_path, keys_to_translate, source_lang, target_lang)



def main():
    parser = argparse.ArgumentParser(description="Translate JSON files from one language to another using Deepl.")
    parser.add_argument('--data', type=pathlib.Path, required=True, help='Input folder containing JSON files')
    parser.add_argument('--output', type=pathlib.Path, required=True, help='Output folder for translated JSON files')
    parser.add_argument('--src_lang', type=str, default="EN", help='Source language code')
    parser.add_argument('--tgt_lang', type=str, default="ZH", help='Target language code')

    args = parser.parse_args()

    # Process the folder
    process_folder(args.data, args.output, args.src_lang, args.tgt_lang)

    print("Translation completed for all files!")

if __name__ == "__main__":
    main()
