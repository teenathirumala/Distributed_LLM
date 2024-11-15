import torch
import Data_preprocess
import pickle



vocab_size,train,test,encode,decode=Data_preprocess.loader()
device='cuda'


with open('trial_2_1000iter.pkl', 'rb') as f:
    model=pickle.load(f)




while(True):
    msg=(input(("enter message")))
    
    context = torch.tensor(encode(msg), dtype=torch.long, device=device)
    generated_chars = decode(model.generate(context.unsqueeze(0), max_new_tokens=100)[0].tolist())
    print(generated_chars)