#f = open("outputs/responses_caption_info.txt", "r")
f = open("outputs/responses_caption_info.txt", "r")

text = f.read()

# key = (example_num, image number)
# value = [description, prediction]
structured_output = {}
text_list = text.split("|||")
first_example = tuple(text_list[0].split(","))
first_example_description = text_list[1]
first_example_output = '1'
structured_output[first_example] = [first_example_description, first_example_output]

for i in range(2, len(text_list) - 2, 2):
    example = tuple(text_list[i].split("\n")[-1].split(","))
    description = text_list[i+1]

    for character in text_list[i + 2]:
        if character.isnumeric():
            prediction = character
            break
    structured_output[example] = [description, prediction]

print(structured_output[('0', '0')])
correct = 0
total = 0
stats = {} # key  = example num, val = [first image correct, second image correct]
for key, val in structured_output.items():
    label = key[1]
    example_num = key[0]
    if example_num not in stats:
        stats[example_num] = [False, False]
    description, prediction = val
    print(key, val)
    if int(prediction) - 1 == int(label):
        stats[example_num][int(label)] = True

for key, val in stats.items():
    if val[0] and val[1]:
        correct += 1
    total += 1
print(correct / total)