from dataset import MahjongGBDataset
from torch.utils.data import DataLoader
from model import TransformerModel
import torch.nn.functional as F
import torch
import os

if __name__ == '__main__':
    logdir = 'log/'
    checkpoint_dir = os.path.join(logdir, 'checkpoint/')
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # 检查是否存在预训练的模型参数
    pretrained_model_path = os.path.join(checkpoint_dir, 'pretrained_model.pkl')
    
    # Load dataset
    splitRatio = 0.9
    batchSize = 512
    trainDataset = MahjongGBDataset(0, splitRatio)
    validateDataset = MahjongGBDataset(splitRatio, 1)
    loader = DataLoader(dataset=trainDataset, batch_size=batchSize, shuffle=True)
    vloader = DataLoader(dataset=validateDataset, batch_size=batchSize, shuffle=False)
    
    # Load model
    input_dim = 147 * 36  # Adjust based on the input observation size
    nhead = 2
    num_encoder_layers = 1
    num_decoder_layers = 1
    dim_feedforward = 512
    num_classes = 235
    model = TransformerModel(input_dim, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, num_classes).to('cuda')

    if os.path.exists(pretrained_model_path):
        print(f'加载预训练模型参数: {pretrained_model_path}')
        model.load_state_dict(torch.load(pretrained_model_path))

    optimizer = torch.optim.SGD(model.parameters(), lr=5e-2)
    
    # Train and validate
    for e in range(2):
        print('Epoch', e)
        torch.save(model.state_dict(), os.path.join(checkpoint_dir, '%d.pkl' % e))
        for i, d in enumerate(loader):
            obs = d[0].view(d[0].size(0), -1).cuda()  # Flatten the input
            input_dict = {'is_training': True, 'obs': {'observation': obs, 'action_mask': d[1].cuda()}}
            logits = model(input_dict)
            loss = F.cross_entropy(logits, d[2].long().cuda())
            if i % 128 == 0:
                print('Iteration %d/%d'%(i, len(trainDataset) // batchSize + 1), 'policy_loss', loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print('Run validation:')
        correct = 0
        for i, d in enumerate(vloader):
            obs = d[0].view(d[0].size(0), -1).cuda()  # Flatten the input
            input_dict = {'is_training': False, 'obs': {'observation': obs, 'action_mask': d[1].cuda()}}
            with torch.no_grad():
                logits = model(input_dict)
                pred = logits.argmax(dim=1)
                correct += torch.eq(pred, d[2].cuda()).sum().item()
        acc = correct / len(validateDataset)
        print('Epoch', e + 1, 'Validate acc:', acc)
