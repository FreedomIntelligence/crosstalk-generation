import os
import openai




def call_gpt3(prompt_text):
  response = openai.Completion.create(
    model="davinci",
    prompt=prompt_text,
    temperature=0.2,
    max_tokens=256,
    top_p=1,
    frequency_penalty=0.85,
    presence_penalty=0.4
  )
  return response


if __name__ == '__main__':
    input_text = '###美国枪击案###\n0:我最近发现一个秘密！\n1:给咱们大伙分享说说?\n0:最近在忙什么呀\n1:说出来吓死你\n0:什么呀这么神秘\n1:你听说那个美国枪击案了么\n0:听说了，都在讨论这个事情'
    call_gpt_ = call_gpt3(input_text)
    print()


