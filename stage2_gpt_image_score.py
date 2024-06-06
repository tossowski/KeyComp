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
        messages = [{"role": "system", "content": "You are a helpful expert at reasoning about images."}]
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
    prompts = []
    folder = "test"
    filenames = os.listdir(folder)
    sorted_filenames = sorted(filenames, key = lambda x: int(x.split("_")[0]))
    nums = list(range(400))
    for num in nums:

        # if num != 3:
        #     continue
        caption_0 = winoground[num]['caption_0']
        caption_1 = winoground[num]['caption_1']

        

        with open(os.path.join(folder, f"{num}_{0}", "description.txt")) as f:
            description_0 = f.read()
            truncated_0 = description_0.split("\n")[0]
        
        with open(os.path.join(folder, f"{num}_{1}", "description.txt")) as f:
            description_1 = f.read()
            truncated_1 = description_1.split("\n")[0]
        
        #print(prompt)
        # f"What is the difference between \"{caption_0}\" and \"{caption_1}\"?", 
        #print(f"Consider the following description of an image:\n\n{truncated}\n\nWhich caption is more appropriate for this image: \"{caption_0}\" or \"{caption_1}\"?")
        #responses = get_chatgpt_prediction([f"Consider the following description of an image:\n\n{truncated}\n\nIf you had to choose, is \"{caption_0}\" a better caption for this image than \"{caption_1}\"? Answer with yes or no even if you are unsure and explain why."])
        #print(f"Image 1: {description_0}\n\nImage 2: {description_1}\n\nConsider the caption \"{caption_0}\". Select the better image for this caption.\nA: Image 1\nB: Image 2\nStart your answer with A or B.")
        
        # with open(os.path.join(folder, f"{num}_{0}", "details.txt")) as f:
        #     detail_string_0 = "\n"
        #     for line in f:
        #         info, description = line.split("|||")
        #         dist_from_top, dist_from_left, height, width = list(map(int, info.split(",")))
        #         if description.startswith("I'm sorry") or description.startswith("It looks like the image you provided is not showing up"):
        #             continue
        #         detail_string_0 += f"{dist_from_top}, {dist_from_left}: {description}"
                
                #detail_string_0 += f"{dist_from_top} pixels down and {dist_from_left} pixels right from the top left of the image, with width {width} and height {height}, {description}\n"
        #print(detail_string_0)

        # with open(os.path.join(folder, f"{num}_{1}", "details.txt")) as f:
        #     detail_string_1 = "\n"
        #     for line in f:
        #         info, description = line.split("|||")
        #         dist_from_top, dist_from_left, height, width = list(map(int, info.split(",")))
        #         if description.startswith("I'm sorry") or description.startswith("It looks like the image you provided is not showing up"):
        #             continue
        #         detail_string_1 += f"{dist_from_top}, {dist_from_left}: {description}"
                #detail_string_1 += f"{dist_from_top} pixels down and {dist_from_left} pixels right from the top left of the image, with width {width} and height {height}, {description}\n"
        #print(detail_string_1)
        #prompts.append(f"Image 1: {truncated_0}\n\nImage 2: {truncated_1}\n\nConsider the caption \"{caption_0}\". Select the better image for this caption.\nA: Image 1\nB: Image 2\nStart your answer with A or B. Briefly explain your decision in 1-2 sentences.")
        #prompts.append(f"Image 1: {truncated_0}\n\nImage 2: {truncated_1}\n\nConsider the caption \"{caption_1}\". Select the better image for this caption.\nA: Image 1\nB: Image 2\nStart your answer with A or B. Briefly explain your decision in 1-2 sentences.")


    # with open("prompt_list", "w") as f:
    #     for i, prompt in enumerate(prompts):
    #         f.write(f"{i//2}_{i%2}\n")
    #         f.write(prompt + "\n")
    #         f.write("-----------------------------------------------------------------------------------------------------------------------\n")

        #print(f"Image 1: {detail_string_0}\n\nImage 2: {detail_string_1}\n\nConsider the caption \"{caption_0}\". Select the better image for this caption.\nA: Image 1\nB: Image 2\nStart your answer with A or B. Briefly explain your decision in 1-2 sentences.")
        # print(os.path.join(folder, f"{num}_{0}", "prompt.txt"))
        # with open(os.path.join(folder, f"{num}_{0}", "prompt.txt"), "w") as f:
        #     f.write(f"Image 1: {truncated_0}\n\nImage 2: {truncated_1}\n\nConsider the caption \"{caption_0}\". Select the better image for this caption.\nA: Image 1\nB: Image 2\nStart your answer with A or B. Briefly explain your decision in 1-2 sentences.")

        # with open(os.path.join(folder, f"{num}_{1}", "prompt.txt"), "w") as f:
        #     f.write(f"Image 1: {truncated_0}\n\nImage 2: {truncated_1}\n\nConsider the caption \"{caption_1}\". Select the better image for this caption.\nA: Image 1\nB: Image 2\nStart your answer with A or B. Briefly explain your decision in 1-2 sentences.")

        responses = get_chatgpt_prediction([f"Statement: {caption_0}\n\nSituation A: {truncated_0}\n\nSituation B: {truncated_1}\n\nThink step by step and fill in the blank: Situation {{}} is most consistent with the statement \"{caption_0}\" because ..."])

        # responses = get_chatgpt_prediction([f"Image 1: {truncated_0}\n\nImage 2: {truncated_1}\n\nConsider the caption \"{caption_0}\". Select the better image for this caption.\nA: Image 1\nB: Image 2\nStart your answer with A or B."])
        prediction_0 = responses[0].replace("\n", "\t")
        #print("Answer: ", prediction_0)


        responses = get_chatgpt_prediction([f"Statement: {caption_1}\n\nSituation A: {truncated_0}\n\nSituation B: {truncated_1}\n\nThink step by step and fill in the blank: Situation {{}} is most consistent with the statement \"{caption_1}\" because ..."])

        # responses = get_chatgpt_prediction([f"Image 1: {truncated_0}\n\nImage 2: {truncated_1}\n\nConsider the caption \"{caption_1}\". Select the better image for this caption.\nA: Image 1\nB: Image 2\nStart your answer with A or B."])
        prediction_1 = responses[0].replace("\n", "\t")
        # print("Answer: ", prediction_1)
        examples.append(num)
        predictions.append([prediction_0, prediction_1])





        #print(predictions)
        with open(f"final_test/cot_image_{run_num}.txt", "w") as f:
            for i in range(len(examples)):
                f.write(f"{examples[i]}|||{predictions[i][0]}|||{predictions[i][1]}\n")
        # examples.append(example_num)
        # labels.append(label)
        # predictions.append(prediction)
        # explanations.append(explanation)
        # with open(os.path.join("chatgpt_outputs", "explanations.txt"), "w") as f:
        #     for i, explanation in enumerate(explanations):
        #         f.write(f"{examples[i]}_{labels[i]}|||{explanation}\n")
        # with open(os.path.join("chatgpt_outputs", "predictions.txt"), "w") as f:
        #     for i, pred in enumerate(predictions):
        #         f.write(f"{examples[i]}_{labels[i]}|||{pred}\n")
        # time.sleep(30)
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
