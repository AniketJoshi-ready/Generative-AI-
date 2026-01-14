# AJ'S FLAN T5 MODEL
#
# for REFERENCE SOME Q/A QUESTIONS ARE HERE :
# FOR CONTEXT.TXT
#What is the company’s minimum cash reserve requirement?

#Who must approve business expenses above ₹25,000?

#Are cryptocurrency investments allowed under this policy?

#How often are internal financial audits conducted?

#What is the company’s risk rating system?

#What happens to idle funds exceeding ₹10 lakh for more than 30 days?

#What are the payment terms for vendors?

#Who reviews treasury reports?





import os 
os.environ["TOKENIZER_PARALLELISM"]="false"
from transformers import AutoTokenizer,AutoModelForSeq2SeqLM
# importing tokenizer ,modelseqtoseq lm from hugging face transformers
# AutoTokenizer: A tokenizer loader that automatically picks the right tokenizer for the model you choose.
# AutoModelForSeq2SeqLM:A pre-trained model loader for Sequence-to-Sequence Language Model (Seq2SeqLM).


# choose instruction tuned model
MODEL_NAME = "google/flan-t5-base"

# A lightweight version of FLAN-T5.
# About 80 million parameters.


print(f"Aniket's FLAN-T5_Summarizer_Q&A_Assistant {MODEL_NAME} model loading...")


# Load tokenizer(handles text<->token)
# Autotokenizer picks the right tockenizer for the model

tokenizer=AutoTokenizer.from_pretrained(MODEL_NAME)

# load the sequence to sequence model
model=AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)








# this function takes the text prompt (string)use a flan t5 model to generate a continuation/ answer and returns the generated text

def AJ_run_flan(prompt:str,max_new_tokens:int=180)->str:

    # tokenisation: tokenise the input ,return pytorch tensors,truncate if too long 
    inputs=tokenizer(prompt,return_tensors="pt",truncation=True)

    #generation: generate text from with light samplingfor naturalness

    outputs=model.generate(**inputs,max_new_tokens=max_new_tokens,do_sample=True,top_p=0.8,temperature=0.9)
    
    #**inputs  : pass tokenised inputs (input id,attention mask),
    # max_new_tokens=max_new_tokens  : how many new tokens to generate ,
    # do_sample=True : enable random  sampling ,
    # top_p=0.9 : nucleus sampling: only consider tokens in the top 90% probability mass,
    # temperature=0.7 :control randomness (lower: safer , more : deterministic/creativive to much)

    # decode token ids back into clean string
    # example ids : [71,867,1234,42,1]
    # text : "helloo,how are you?"

    return tokenizer.decode(outputs[0],skip_special_tokens=True).strip()






# this function is used for summerization
#it creates prmpt with 4 to 6 bullet points
def AJ_Summerize_text(text:str)->str:
    # prompt template instructing the model to produce 4 to 6 bullet points
    prompt=("Summarize the following text into 5 bullet points. Each bullet should be concise and highlight key ideas:\n" + text)

    # allow slightly longer output for bullet lists
    return AJ_run_flan(prompt,max_new_tokens=100)



# this function is used to load the content from our local file 
#and return the complete file contents in one string
def AJ_load_context(path:str = "context.txt")->str:
    try:
        # read entire file as single string
        with open(path,"r",encoding="utf-8")as f:
            return f.read()
    except FileNotFoundError:
        return ""    
    


#this function ask flan to answer using only the given context
#if answer is not present, ask it to say 'not found'
def AJ_answer_from_context(question:str,context:str)->str:
    if not context.strip():
        return "context file not found or empty.create 'context.txt' first"
    # construct strict prompt for flan t5
    prompt=(f"You are a helpful assistant.Answer the question ONLY using the context.\nIf the answer is not the context ,reply exactly:Not found.\n\nContext:\n{context}\n\nQuestion:{question}\nAnswer:")


    #generate a concise answer grunded in the provided notes
    return AJ_run_flan(prompt,max_new_tokens=256)







def main():
    border="-"*80
    short_border="-"*20
    print(border)
    print(f"\n{short_border} Aniket's FLAN-T5 Model {short_border}")
    print("1. sumerize the data")
    print("2. Question and answer over local context.txt")
    print("0.Exit")
    print(border)

    while True:
        choice=input("\n choose an option (1/2/0):").strip()
        if choice=="0":
            print("exit bro")
            break
        elif choice=="1":
            print("Provide me text that you want to summerize:")

            # collecting multiple lines for summerization
            lines=[]
            while True:
                line=input()
                # it stop when an user enter on an empty line like eneter" "enter then stop
                if not line.strip():
                    break
                lines.append(line)
                
                
                # join the lines into single block of text
                text="\n".join(lines).strip()

                # if no text was provided, prmpt again
                if not text:
                    print("Aniket's FLAN Model says : No text recieved.")
                    continue

                # run summerrization and print result
                print("\nSummary generated by Aniket's FLAN model:")
                print(AJ_Summerize_text(text))

        elif choice=="2":
            # load the context from local file "context.txt"

            ctx=AJ_load_context("context.txt")

            if not ctx.strip():
                #help the user if context is missing /empty
                print("missing context.txt,create it in same folder and try again.")
                continue
            # ASK question related to provided context

            q=input("\nAsk a question about your context to AJ flan model :").strip()
            if not q:
                print("No question received.")
                continue

            #generate an answer grunded only in the context 
            print("\nAnswer from AJ flan model:")
            print(AJ_answer_from_context(q,ctx))


        else:
            print("please chose 1,2, or 0.")


if __name__=="__main__":
    main()    