import os
import asyncio
from model import llm
async def get_prompts(df, id,type):
    if type=="file":
        file_path = f'data/csv/{id}/{id}.txt'
    else:
        file_path = f'data/postgres/{id}/{id}.txt'
    
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    with open(file_path, 'a+') as file:
        file.seek(0)
        previous_prompts = file.read()
        print("Loaded previous prompts:", previous_prompts)
        combined_input = f"""
            Previous prompts: {previous_prompts}
            Please analyze the following dataframe: {df}
            From this data, generate 4 diverse and insightful prompts that a user can ask.Dont repeat the previosu prompts
            These prompts need to be a maximum of 7 words. Ensure the prompts are contextually relevant 
            to the data provided. Format the output as a JSON object with each prompt as a separate key.
            Output the JSON without any backticks, markdown formatting, or the word "json" at the start. 
            Return only plain, valid JSON. 

            Example OUTPUT:
            {{
                "prompt1": "Prompt 1",
                "prompt2": "Prompt 2",
                "prompt3": "Prompt 3",
                "prompt4": "Prompt 4"
            }}
        """

    result = await asyncio.to_thread(
        llm.invoke, combined_input
    )

    questions_list = result.content.replace('```json', '').replace('```', '').strip('\"').split('\", \"')
    questions_dict = {f"question_{i + 1}": question for i, question in enumerate(questions_list)}
    print(questions_dict)
    with open(file_path, 'w') as file:
        file.write(result.content.strip())

    return questions_list