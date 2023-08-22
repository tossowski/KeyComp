first_example_correct = [False] * 400
second_example_correct = [False] * 400

rationales = []
responses= []
with open("responses.txt", "r") as f:
    for i, line in enumerate(f):
        example, rationale, response = line.split("|||")
        example_num, label = example.split(",")
        for char in response:
            if char.isnumeric():
                prediction = int(char)
                if prediction == int(label) + 1:
                    if int(label) == 0:
                        first_example_correct[i // 2] = True
                    else:
                        second_example_correct[i // 2] = True
                break
        rationales.append(rationale)
        responses.append(response)

count = 0
for i in range(len(first_example_correct)):
    if (first_example_correct[i] and second_example_correct[i]):
        count += 1
        print(i)
        print(rationales[i * 2])
        print("-----------")
        print(rationales[i * 2 + 1])
        print("--------------------------------------")
print(count / 400)