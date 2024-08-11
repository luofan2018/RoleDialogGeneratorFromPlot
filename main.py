from transformers import AutoTokenizer, AutoModel
import gradio as gr

N_CONVERSATIONS = 3
# C:/frankluo/Github/RoleDialogGeneratorFromPlot/Fles/zhuxian1.txt
tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm3-6b", trust_remote_code=True)
model = AutoModel.from_pretrained("THUDM/chatglm3-6b", trust_remote_code=True, device='cuda')
model = model.eval()

def processData(txt_file_path):
    output_path = txt_file_path.replace(".md","_dialog.md")
    # read in the story
    with open(txt_file_path,"r",encoding="utf8") as f:
        story=f.read()
    # generate two names
    response, history = model.chat(tokenizer, f"请参照以下段落选择两个有交集的人物的名字。 “{story}” 以如下格式给出: A-B", history=[])
    print(response)
    try:
        name_a,name_b = response.split("-")
    except Exception as e:
        raise ValueError(f"response of {response} split issue:{e}")
    # generate chats
    conversations = []
    for i in range(N_CONVERSATIONS):
        response, history = model.chat(tokenizer,f"请生成{name_a}对{name_b}{'在这个场景内想说的话' if len(conversations)==0 else '的回复'}。以如下格式给出：{name_a}:",history=history)
        print(response)
        conversations.append(response)
        response, history = model.chat(tokenizer,f"请生成{name_b}对{name_a}的回复。以如下格式给出：{name_b}:",history=history)
        print(response)
        conversations.append(response)
    # compile response chats into markdown
    with open(output_path,"w",encoding="utf8") as f:
        for conversation in conversations:
            f.write(conversation+"  \n\n")
    
    return(output_path)



if  __name__ == "__main__":

    iface = gr.Interface(
        fn=processData,
        inputs=[
            gr.Textbox(label="请提供txt地址(无引号)："),
        ],
        outputs=[
            gr.Text(label="对话文档"),
        ],
        title="AI机编对话文案",
        description="chatglm生成对话文案",
    )
    
    iface.launch(
            #server_name=SERVER_NAME,server_port=SERVER_PORT
            )
    iface.close()