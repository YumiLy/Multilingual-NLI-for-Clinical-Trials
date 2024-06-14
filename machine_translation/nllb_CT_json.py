import json
import ctranslate2
import sentencepiece as spm
import os
import argparse
import pathlib

def translate_text(text, translator, sp, src_lang, tgt_lang, beam_size):
    source_sents_subworded = sp.encode_as_pieces([text])
    source_sents_subworded = [[src_lang] + sent + ["</s>"] for sent in source_sents_subworded]
    # Set the translation options
    options = {
        "batch_type": "tokens",
        "max_batch_size": 2048,
        "beam_size": beam_size,
        "target_prefix": [[tgt_lang] for _ in source_sents_subworded]
    }

    translations = translator.translate_batch(source_sents_subworded, **options)
    translated_texts = [sp.decode(translation.hypotheses[0]) for translation in translations]
    translated_text = translated_texts[0].replace(tgt_lang, '').strip()
    return translated_text

def translate_file(input_file_path, output_file_path, translator, sp, src_lang, tgt_lang, beam_size):
    with open(input_file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    
    for key in ['Intervention', 'Eligibility', 'Results', 'Adverse Events']:
        if key in data:
            data[key] = [translate_text(item, translator, sp, src_lang, tgt_lang, beam_size) for item in data[key]]

    with open(output_file_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)

# 遍历处理文件夹和子文件夹的所有内容
def process_folder(folder_path, output_folder, translator, sp, src_lang, tgt_lang, beam_size):
    os.makedirs(output_folder, exist_ok=True)
    for entry in os.listdir(folder_path):
        input_path = os.path.join(folder_path, entry)
        output_path = os.path.join(output_folder, entry)
        if os.path.isdir(input_path):
            process_folder(input_path, output_path, translator, sp, src_lang, tgt_lang, beam_size)
        elif input_path.endswith('.json'):
            translate_file(input_path, output_path, translator, sp, src_lang, tgt_lang, beam_size)

def main():
    parser = argparse.ArgumentParser(description="Translate JSON files from one language to another using specified models.")
    parser.add_argument('--data', type=pathlib.Path, required=True, help='Input folder containing JSON files')
    parser.add_argument('--output', type=pathlib.Path, required=True, help='Output folder for translated JSON files')
    parser.add_argument('--ct_model_path', type=pathlib.Path, required=True, help='Path to the CTranslate2 model directory')
    parser.add_argument('--sp_model_path', type=pathlib.Path, required=True, help='Path to the SentencePiece model file')
    parser.add_argument('--src_lang', type=str, default="eng_Latn", help='Source language code')
    parser.add_argument('--tgt_lang', type=str, default="zho_Hans", help='Target language code')
    parser.add_argument('--beam_size', type=int, default=4, help='Beam size for translation')

    args = parser.parse_args()

    # Load the models
    sp = spm.SentencePieceProcessor()
    sp.load(str(args.sp_model_path))
    translator = ctranslate2.Translator(str(args.ct_model_path), device="cuda")  # or "cpu"

    # Process the folder
    process_folder(args.data, args.output, translator, sp, args.src_lang, args.tgt_lang, args.beam_size)

    print("Translation completed for all files!")

if __name__ == "__main__":
    main()
