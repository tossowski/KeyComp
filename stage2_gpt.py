import openai
import os
from datasets import load_dataset
import time

with open("API_KEY_2", "r") as f:
    API_KEY = f.read()


openai.api_key = API_KEY
auth_token = "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"  # Replace with an auth token, which you can get from your huggingface account: Profile -> Settings -> Access Tokens -> New Token
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
            time.sleep(2)
        responses.append(completion.choices[0].message.content)
    return responses

    

    for i, response in enumerate(responses):
        messages = [{"role": "system", "content": "You are a helpful assistant."}]
        messages.append({"role": "user", "content": prompts[i]})
        messages.append({"role": "assistant", "content": response})
        messages.append({"role": "user", "content": prompts[i + 1]})

        completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            temperature = 0,
            messages=messages
        )


        responses.append(completion.choices[0].message.content)


    
    return responses

for run_num in range(3):
    examples = []
    labels = []
    summaries = []
    predictions = []
    explanations = []
    filenames = os.listdir("test2")
    sorted_filenames = sorted(filenames, key = lambda x: int(x.split("_")[0]))

    for folder in sorted_filenames:
        
        example_num = int(folder.split("_")[0])
        label = folder.split("_")[1]
        

        caption_0 = winoground[example_num]['caption_0']
        caption_1 = winoground[example_num]['caption_1']
        with open(os.path.join("test2", folder, "best.txt")) as f:
            description = f.read()
            truncated = description
            #truncated = ". ".join(description.split("."))

        # Adding Details
        # with open(os.path.join("all_descriptions", folder, "details.txt")) as f:
        #     detail_string = "Here are some detailed regions of the image:\n"
        #     for line in f:
        #         info, description = line.split("|||")
        #         dist_from_top, dist_from_left, height, width = list(map(int, info.split(",")))
        #         if description.startswith("I'm sorry") or description.startswith("It looks like the image you provided is not showing up"):
        #             continue
        #         detail_string += f"{dist_from_top} pixels down and {dist_from_left} pixels right from the top left of the image, with width {width} and height {height}, {description}\n"
        #print(detail_string)
            
            #print(truncated)

        #print(prompt)
        # f"What is the difference between \"{caption_0}\" and \"{caption_1}\"?", 
        #print(f"Consider the following description of an image:\n\n{truncated}\n\nWhich caption is more appropriate for this image: \"{caption_0}\" or \"{caption_1}\"?")
        #responses = get_chatgpt_prediction([f"Consider the following description of an image:\n\n{truncated}\n\nIf you had to choose, is \"{caption_0}\" a better caption for this image than \"{caption_1}\"? Answer with yes or no even if you are unsure and explain why."])
        #print(f"{truncated}\n\nSelect the best caption for this image:\nA:\"{caption_0}\"\nB:\"{caption_1}\"\nThink carefully step-by-step to arrive at the correct answer. Even if you are unsure, make your best guess and end your answer with A or B")
        #responses = get_chatgpt_prediction([f"{truncated}\n\nSelect the best caption for this image:\nA:\"{caption_0}\"\nB:\"{caption_1}\"\nThink carefully step-by-step to arrive at the correct answer. Even if you are unsure, make your best guess and end your answer with A or B"])
        responses = get_chatgpt_prediction([f"{truncated}\n\nSelect the best caption for this image:\nA:\"{caption_0}\"\nB:\"{caption_1}\"\nStart your answer with A or B. Even if you are unsure make a guess. Explain your decision in 1-2 sentences."])
        prediction = responses[0].replace("\n", "\t")
        explanation = "N/A"

        examples.append(example_num)
        labels.append(label)
        predictions.append(prediction)
        explanations.append(explanation)

        with open(os.path.join(f"final_test", f"explanation_prompting_best_description_{run_num}.txt"), "w") as f:
            for i, pred in enumerate(predictions):
                f.write(f"{examples[i]}_{labels[i]}|||{pred}\n")
        #responses = get_chatgpt_prediction([f"Explain the difference between \"{caption_0}\" and \"{caption_1}\"", prompt, f'Based on this information, which caption would be more appropriate: \"{caption_0}\" or \"{caption_1}\"? Answer with either \"{caption_0}\" or \"{caption_1}\"'])
        # print(responses)
        # summaries.append(summary)
        # predictions.append(prediction)
        # examples.append(example_num)
        # labels.append(label)
        # break
        # with open(os.path.join("chatgpt_outputs", "summaries.txt"), "w") as f:
        #     for i, summary in enumerate(summaries):
        #         f.write(f"{example_num}_{label}|||{summary}\n")
        # with open(os.path.join("chatgpt_outputs", "predictions.txt"), "w") as f:
        #     for i, pred in enumerate(predictions):
        #         f.write(f"{example_num}_{label}|||{pred}\n")
        # time.sleep(30)
