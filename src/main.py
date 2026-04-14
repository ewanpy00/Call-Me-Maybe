import torch
from src.MaskingDependencies import MaskingDependencies
import json
from llm_sdk import llm_sdk

FUNCTIONS_PATH = "data/input/functions_definition.json"
PROMPTS_PATH = "data/input/function_calling_tests.json"
OUTPUT_PAtH = "data/output/function_calling_results.json"


# def logits_masking(logits):



def load_json(file_path: str):
    try:
        with open(file_path, "r", encoding='utf-8') as f:
            result = json.load(f)
            return result
    except Exception as e:
        print("An error occured while reading the prompt file.\n", e)
        return None


import json

def main():
    prompts = load_json("data/input/function_calling_tests.json")
    functions = load_json("data/input/functions_definition.json")
    model = llm_sdk.Small_LLM_Model(model_name="Qwen/Qwen3-0.6B")
    
    mask = MaskingDependencies(model, functions)
    end_token_id = model.encode("}")[0][-1].item()
    
    # Создаем список для хранения всех результатов
    output_results = []
    
    for prompt in prompts:
        user_query = prompt['prompt']

        system_prompt = f"""You are a function calling assistant. You must output strictly valid JSON matching the function schema.
Available functions:
{json.dumps(functions, indent=2)}

Example:
User query: add 10 and 20
Result: {{"name": "fn_add_numbers", "parameters": {{"a": 10, "b": 20}}}}
"""

        final_prompt = f"{system_prompt}\n\nUser query: {user_query}\nResult: {{"
        ids = model.encode(final_prompt)[0].tolist()
        
        print(f"Processing Query: {user_query}")
        
        for i in range(50):
            logits = model.get_logits_from_input_ids(ids)
            masked_logits = mask.apply_constrain(logits, ids)
            next_token_id = torch.argmax(masked_logits).item()
            
            ids.append(next_token_id)

            char = model.decode([next_token_id])
            
            if "}" in char:
                current_gen = model.decode(ids).split("Result: ")[-1]
                open_b = current_gen.count('{')
                close_b = current_gen.count('}')
                if open_b == close_b and open_b > 0:
                    break

        # Декодируем и чистим JSON
        full_text = model.decode(ids)
        final_json_str = full_text.split("Result: ")[-1].strip()

        # Пытаемся распарсить строку в объект Python, чтобы сохранить как валидный JSON
        try:
            json_obj = json.loads(final_json_str)
        except Exception as e:
            print(f"Warning: Could not parse model output as JSON: {e}")
            json_obj = final_json_str # Сохраняем как строку, если парсинг не удался

        # Добавляем в общий список в нужном формате
        output_results.append({
            "prompt": user_query,
            "answer": json_obj
        })
        
        print(f"Done.")
        print("-" * 20)

    # ЗАПИСЬ В ФАЙЛ
    output_path = "data/output/results.json"
    try:
        with open(output_path, "w", encoding='utf-8') as f:
            # indent=4 сделает файл читаемым для человека
            # ensure_ascii=False сохранит символы как есть (важно для кириллицы/японского)
            json.dump(output_results, f, indent=4, ensure_ascii=False)
        print(f"Successfully saved results to {output_path}")
    except Exception as e:
        print(f"Error saving file: {e}")

if __name__ == "__main__":
    main()