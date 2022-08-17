
import io,os
from tqdm import tqdm

from beta_code.gpt3_prompt_generate.GPT3_Call import call_gpt3

this_dir = os.path.split(os.path.realpath(__file__))[0]


prompt_file_path = os.path.join(this_dir,'human_filter_prompt.txt')
completion_file_path = os.path.join(this_dir,'candi_completion.txt')

def main():
    # input_text + '\n--gen--' + call_gpt_.last_response.data['choices'][0]['text']
    all_prompts = io.open(prompt_file_path, 'r').read().split('\n\n')
    for idx in tqdm(range(len(all_prompts))):
        prompt = all_prompts[idx]
        response = call_gpt3(prompt)
        all_content = prompt + '\n--gen--\n' + response.last_response.data['choices'][0]['text'] + '\n\n'
        io.open('candi_completion.txt', 'a').write(all_content)

    print()

if __name__ == '__main__':
    main()
