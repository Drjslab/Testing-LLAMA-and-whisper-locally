from langchain.llms import LlamaCpp
# from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.prompts import PromptTemplate

# run following line before runnning the code;
# huggingface-cli download TheBloke/Llama-2-7b-Chat-GGUF llama-2-7b-chat.Q4_K_M.gguf --local-dir . --local-dir-use-symlinks False

model_file = "llama-2-7b-chat.Q4_K_M.gguf"

template_llama = """<s>[INST] <<SYS>>
                    You are a smart mini computer named Raspberry Pi.
                    Write a short but funny answer.</SYS>>
                    {question} [/INST]"""

template = template_llama

n_gpu_layers = 30  # Change this value based on your model and your GPU VRAM pool.
n_batch = 512  # Should be between 1 and n_ctx, consider the amount of VRAM in your GPU.

# callback_manager = CallbackManager([StreamingCustomCallbackHandler()])
llm = LlamaCpp(
    model_path=model_file,
    temperature=0.1,
    n_gpu_layers=n_gpu_layers,
    n_batch=256,
    verbose=True,
)

# llm_chain =LLMChain(prompt=template, llm=llm)
question = "Who is president of india?"
# llm_chain.run(question)
print("Wait...")
out = llm(question)
print(out)
print("--done--")
