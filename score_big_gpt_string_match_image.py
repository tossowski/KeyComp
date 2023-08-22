#f = open("outputs/responses_caption_info.txt", "r")
from datasets import load_dataset

auth_token = "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"  # Replace with an auth token, which you can get from your huggingface account: Profile -> Settings -> Access Tokens -> New Token
winoground = load_dataset("facebook/winoground", use_auth_token=auth_token)["test"]

stats = {}

categories = {
    "Non_Compositional": [72, 73, 74, 95, 96, 133, 149, 150, 164, 218, 221, 222, 224, 235, 237, 246, 274, 275, 321, 325, 326, 327, 332, 333, 334, 350, 364, 365, 398, 399],
    "Ambiguously_Correct": [3, 13, 36, 46, 75, 76, 77, 78, 82, 86, 88, 113, 119, 121, 131, 132, 133, 148, 189, 220, 221, 262, 263, 287, 295, 300, 303, 305, 307, 310, 319, 322, 332, 340, 343, 344, 348, 353, 355, 356, 363, 374, 377, 381, 386, 394],
    "Visually_Difficult": [4, 22, 23, 25, 27, 28, 31, 36, 58, 65, 69, 70, 77, 97, 116, 118, 134, 138, 159, 163, 172, 176, 182, 187, 200, 214, 226, 227, 232, 241, 255, 268, 286, 335, 352, 356, 373, 376],
    "Unusual_Image" : [31, 36, 38, 41, 42, 61, 62, 70, 78, 84, 93, 110, 114, 116, 128, 133, 136, 155, 159, 164, 173, 174, 188, 201, 203, 204, 206, 209, 218, 223, 239, 245, 246, 247, 254, 274, 275, 277, 280, 282, 293, 303, 307, 314, 319, 320, 327, 329, 339, 362, 367, 383, 384, 388, 393, 395],
    "Unusual_Text": [10, 41, 49, 58, 63, 68, 70, 152, 156, 159, 163, 174, 198, 201, 209, 214, 215, 221, 229, 233, 237, 253, 257, 264, 275, 287, 303, 315, 318, 323, 324, 326, 327, 335, 338, 342, 343, 345, 346, 351, 354, 359, 364, 376, 383, 385, 386, 387, 390, 394],
    "Complex_Reasoning": [16, 40, 44, 46, 55, 58, 81, 83, 93, 97, 103, 111, 116, 118, 128, 130, 135, 143, 144, 176, 190, 191, 192, 193, 199, 206, 208, 209, 210, 211, 217, 218, 219, 227, 228, 230, 234, 238, 241, 242, 249, 254, 258, 260, 262, 264, 267, 268, 275, 276, 281, 284, 286, 287, 292, 295, 296, 298, 299, 304, 311, 312, 316, 330, 331, 334, 336, 342, 347, 358, 361, 371, 373, 375, 382, 383, 392, 396],
    "NoTag": [0, 1, 2, 5, 6, 7, 8, 9, 11, 12, 14, 15, 17, 18, 19, 20, 21, 24, 26, 29, 30, 32, 33, 34, 35, 37, 39, 43, 45, 47, 48, 50, 51, 52, 53, 54, 56, 57, 59, 60, 64, 66, 67, 71, 79, 80, 85, 87, 89, 90, 91, 92, 94, 98, 99, 100, 101, 102, 104, 105, 106, 107, 108, 109, 112, 115, 117, 120, 122, 123, 124, 125, 126, 127, 129, 137, 139, 140, 141, 142, 145, 146, 147, 151, 153, 154, 157, 158, 160, 161, 162, 165, 166, 167, 168, 169, 170, 171, 175, 177, 178, 179, 180, 181, 183, 184, 185, 186, 194, 195, 196, 197, 202, 205, 207, 212, 213, 216, 225, 231, 236, 240, 243, 244, 248, 250, 251, 252, 256, 259, 261, 265, 266, 269, 270, 271, 272, 273, 278, 279, 283, 285, 288, 289, 290, 291, 294, 297, 301, 302, 306, 308, 309, 317, 328, 337, 341, 349, 357, 360, 366, 368, 369, 370, 372, 378, 379, 380, 389, 391, 397]
}

total = 0
with open("outputs/responses_naive_beam10_4.txt", "r") as f:

    for line in f:
        info = line.split("|||")
        example_num, label = list(map(int, info[0].split(",")))
        correct_caption = winoground[example_num][f"caption_{label}"].lower()
        response = info[-1].lower()
        answer = response[:len(correct_caption) + 100]
        
        try:
            idx = answer.index(correct_caption)
            stats[(example_num, label)] = True
        except:
            # No exact match/incorrect
            stats[(example_num, label)] = False

correct = 0
for i in range(len(winoground)):
    try:
        total += 1
        if stats[(i, 0)] and stats[(i, 1)]:
            correct += 1
        
    except:
        continue

print(correct/total)

text_stats = {}
with open("outputs/image_final_test_3.txt", "r") as f:
    for line in f:
        info = line.split("|||")
        example_num, label = list(map(int, info[0].split(",")))
        correct_caption = winoground[example_num][f"caption_{label}"].lower()
        response = info[-1].lower()
        answer = response[:len(correct_caption) + 45]
        if label == 1 and answer.startswith("b"):
            text_stats[(example_num, label)] = True
        elif label == 0 and answer.startswith("a"):
            text_stats[(example_num, label)] = True
        else:
            # No exact match/incorrect
            text_stats[(example_num, label)] = False

correct = 0
group_correct = 0
for i in range(len(winoground)):
    try:
        if text_stats[(i, 0)] and text_stats[(i, 1)]:
            correct += 1
        if stats[(i, 0)] and stats[(i, 1)] and text_stats[(i, 0)] and text_stats[(i, 1)]:
            group_correct += 1
    except:
        continue



print(f"Overall Performance: {correct / len(winoground) * 100:.2f}")
print(f"Overall Performance Group: {group_correct / len(winoground) * 100:.2f}")

print("----------------------")
print("Category Performances:")

for category in categories:
    ids = categories[category]
    correct = 0
    total = 0
    for id in ids:
        if stats[(id, 0)] and stats[(id, 1)]:
            correct += 1
        total += 1
        

    print(f"{category}: {correct / total * 100:.2f}")

 