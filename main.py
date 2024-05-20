import openai
from tqdm import tqdm
import editdistance
import numpy as np

# 采样数量，越多越精确
n_per_prompt = 10

# 参考模型配置（能确定为真模型）
reference_config = {
    "client": openai.OpenAI(
        base_url="https://api.ohmygpt.com/v1",
        api_key="KEY"
    ),
    "model": "gpt-3.5-turbo"
}

# 测试模型配置（待验证的模型）
test_config = {
    "client": openai.OpenAI(
        base_url="https://api.ohmygpt.com/v1",
        api_key="KEY"
    ),
    "model": "gpt-3.5-turbo"
}

prompt = "简要介绍 Transformer 深度学习模型。篇幅不超过 200 字。"

############################################

reference_results = []
reference_results2 = []
test_results = []

print("ROUND 1/3: Generating 1st reference results")
for i in tqdm(range(n_per_prompt)):
    completion = reference_config["client"].chat.completions.create(
        model=reference_config["model"],
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0,
        top_p=0
    )

    reference_results.append(completion.choices[0].message.content)

print("ROUND 2/3: Generating 2nd reference results")
for i in tqdm(range(n_per_prompt)):
    completion = reference_config["client"].chat.completions.create(
        model=reference_config["model"],
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0,
        top_p=0
    )

    reference_results2.append(completion.choices[0].message.content)

print("ROUND 3/3: Generating test results")
for i in tqdm(range(n_per_prompt)):
    completion = test_config["client"].chat.completions.create(
        model=test_config["model"],
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0,
        top_p=0
    )

    test_results.append(completion.choices[0].message.content)


############################################

edit_distance_matrix_ref = np.zeros((n_per_prompt, n_per_prompt))

for i, b in enumerate(reference_results):
    for j, r in enumerate(reference_results2):
        edit_distance_matrix_ref[i, j] = editdistance.eval(b, r)

edit_distance_matrix_test = np.zeros((n_per_prompt, n_per_prompt))

for i, b in enumerate(reference_results):
    for j, r in enumerate(test_results):
        edit_distance_matrix_test[i, j] = editdistance.eval(b, r)

############################################

print("######### distance matrix #########")
print("### ref vs ref ###")
print(edit_distance_matrix_ref)
print("### ref vs test ###")
print(edit_distance_matrix_test)

# calculate min, max
print("######### min / max distance #########")
print("### ref vs ref ###")
print(np.min(edit_distance_matrix_ref), np.max(edit_distance_matrix_ref))
print("### ref vs test ###")
print(np.min(edit_distance_matrix_test), np.max(edit_distance_matrix_test))

# calculate 25%, 50%, 75% percentile
print("######### 25%, 50%, 75% percentile #########")
print("### ref vs ref ###")
print(np.percentile(edit_distance_matrix_ref, [25, 50, 75]))
print("### ref vs test ###")
print(np.percentile(edit_distance_matrix_test, [25, 50, 75]))

