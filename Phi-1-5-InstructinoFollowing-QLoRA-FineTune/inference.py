from transformers import (
    AutoModelForCausalLM, 
    logging, 
    pipeline,
    AutoTokenizer
)

if __name__=="__main__":   

    #Load model and the tokenizer
    model = AutoModelForCausalLM.from_pretrained('outputs/phi_1_5_alpaca_qlora/best_model/')
    tokenizer = AutoTokenizer.from_pretrained('outputs/phi_1_5_alpaca_qlora/best_model/')

    pipe = pipeline(
                    task='text-generation', 
                    model=model, 
                    tokenizer=tokenizer, 
                    max_length=256,
                    eos_token_id=tokenizer.eos_token_id,
                    device='cuda'
                )
    
    # First Prompt
    prompt = """### Instruction:
                What are LLMs?

                ### Input:


                ### Response:
                """

    print(prompt)

    result = pipe(prompt)

    print(result[0]['generated_text'])

    # Second Prompt
    prompt = """### Instruction:
                Correct the grammar and spelling of the following sentence.

                ### Input:
                I am learn to drive a care.

                ### Response:
                """

    result = pipe(prompt)
    print(result[0]['generated_text'])