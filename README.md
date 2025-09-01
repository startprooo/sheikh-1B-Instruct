# sheikh-1B-Instruct -  Large Language Model (LLM) is an instruct fine-tuned 
Extended vocabulary to 32768
Supports v3 Tokenizer
Supports function calling
 Function calling with transformers
To use this example, you'll need transformers version 4.42.0 or higher. Please see the function calling guide in the transformers docs for more information. 
Note that, for reasons of space, this example does not show a complete cycle of calling a tool and adding the tool call and tool results to the chat history so that the model can use them in its next generation. For a full tool calling example, please see the function calling guide, and note that Mistral does use tool call IDs, so these must be included in your tool calls and tool results. They should be exactly 9 alphanumeric characters. 
 Instruct model is a quick demonstration that the base model can be easily fine-tuned to achieve compelling performance. It does not have any moderation mechanisms. We're looking forward to engaging with the community on ways to make the model finely respect guardrails, allowing for deployment in environments requiring moderated outputs.
