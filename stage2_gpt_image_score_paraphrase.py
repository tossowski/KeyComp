import openai
import os
from datasets import load_dataset
import time

with open("API_KEY", "r") as f:
    API_KEY = f.read()


openai.api_key = API_KEY
auth_token = os.environ["HF_TOKEN"]  # Replace with an auth token, which you can get from your huggingface account: Profile -> Settings -> Access Tokens -> New Token
winoground = load_dataset("facebook/winoground", use_auth_token=auth_token)["test"]

def get_chatgpt_prediction(prompts):
    responses = []
    for i, prompt in enumerate(prompts):
        messages = [{"role": "system", "content": "You are a helpful assistant."}]
        for j, response in enumerate(responses):
            messages.append({"role": "user", "content": prompts[j]})
            messages.append({"role": "assistant", "content": response})
        messages.append({"role": "user", "content": prompt})

        received_response = False
        while not received_response:
            try:
                completion = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    temperature = 1,
                    messages=messages
                )
                received_response = True
            except:
                print("Rate Limit error")
            time.sleep(20)
        responses.append(completion.choices[0].message.content)
    return responses

paraphrases = []
with open("chatgpt_outputs/paraphrases.txt", "r") as f:
    for line in f:
        #print(line.split("|||"))
        example_num, p1, p2 = line.split("|||")
        paraphrases.append([p1,p2])

examples = []
labels = []
summaries = []
predictions = []
explanations = []
prompts = []

folder = "test2"
filenames = os.listdir(folder)
sorted_filenames = sorted(filenames, key = lambda x: int(x.split("_")[0]))
nums = list(range(0,400))
for num in nums:


    # caption_0 = winoground[num]['caption_0']
    # caption_1 = winoground[num]['caption_1']

    caption_0 = paraphrases[num][0]
    caption_1 = paraphrases[num][1]

    

    with open(os.path.join(folder, f"{num}_{0}", "best.txt")) as f:
        description_0 = f.read()
        truncated_0 = ". ".join(description_0.split("."))
    
    with open(os.path.join(folder, f"{num}_{1}", "best.txt")) as f:
        description_1 = f.read()
        truncated_1 = ". ".join(description_1.split("."))
    
    # responses = get_chatgpt_prediction([f"Can you paraphrase the caption \"{caption_0}\" in a natural way? Just answer with the paraphrase and put it in quotes."])
    # paraphrase_0 = responses[0]
    # paraphrase_0 = paraphrase_0.split('"')[1::2][0]
    # print(paraphrase_0)
    
    # responses = get_chatgpt_prediction([f"Can you paraphrase the caption \"{caption_1}\" in a natural way? Just answer with the paraphrase and put it in quotes."])
    # paraphrase_1 = responses[0]
    # paraphrase_1 = paraphrase_1.split('"')[1::2][0]
    # print(paraphrase_1)
    # examples.append(num)
    # paraphrases.append([paraphrase_0, paraphrase_1])

    # with open("chatgpt_outputs/paraphrases_2.txt", "w") as f:
    #    for i in range(len(examples)):
    #        f.write(f"{examples[i]}|||{paraphrases[i][0]}|||{paraphrases[i][1]}\n")

    print(f"Image 1: {truncated_0}\n\nImage 2: {truncated_1}\n\nConsider the caption \"{caption_0}\". Select the better image for this caption.\nA: Image 1\nB: Image 2\nStart your answer with A or B. Briefly explain your decision in 1-2 sentences.")

    responses = get_chatgpt_prediction([f"Image 1: {truncated_0}\n\nImage 2: {truncated_1}\n\nConsider the caption \"{caption_0}\". Select the better image for this caption.\nA: Image 1\nB: Image 2\nStart your answer with A or B. Briefly explain your decision in 1-2 sentences."])
    prediction_0 = responses[0].replace("\n", "\t")
    print("Answer: ", prediction_0)

    responses = get_chatgpt_prediction([f"Image 1: {truncated_0}\n\nImage 2: {truncated_1}\n\nConsider the caption \"{caption_1}\". Select the better image for this caption.\nA: Image 1\nB: Image 2\nStart your answer with A or B. Briefly explain your decision in 1-2 sentences."])
    prediction_1 = responses[0].replace("\n", "\t")
    print("Answer: ", prediction_1)
    examples.append(num)
    predictions.append([prediction_0, prediction_1])


    with open("chatgpt_outputs/best_description_image_score_paraphrase_1.txt", "w") as f:
       for i in range(len(examples)):
           f.write(f"{examples[i]}|||{predictions[i][0]}|||{predictions[i][1]}\n")
   