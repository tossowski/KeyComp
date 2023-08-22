first_example_correct = [False] * 400
second_example_correct = [False] * 400
responses= []
with open("outputs/responses.txt", "r") as f:
    for i, line in enumerate(f):
        example, response = line.split("|||")
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
        responses.append(response)

count = 0
for i in range(len(first_example_correct)):
    if (first_example_correct[i] and second_example_correct[i]):
        count += 1
        print(i)
        print(responses[i * 2])
        print(responses[i * 2 + 1])
        print("--------------------------------------")
print(count / 400)