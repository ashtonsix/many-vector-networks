import sys
sys.path.append('/home/salamander/fastai')

from timeit import default_timer as timer
from fastai.imports import *
from fastai.torch_imports import *
from fastai.learner import *
from fastai.column_data import *

class CatLearner(Learner):
    def __init__(self, data, models, **kwargs):
        super().__init__(data, models, **kwargs)

    def _get_crit(self, data): return F.cross_entropy

class CatModel(BasicModel):
    def get_layer_groups(self): return self.model

class CatDataset(Dataset):
    def __init__(self, x, y):
        self.x=x
        self.y=y
    def __len__(self):
        return len(self.y)
    def __getitem__(self, i):
        return [self.x[i], self.y[i]]

class CatModelData(ModelData):
    def __init__(self, path, val_x, val_y, trn_x, trn_y, batch_size):
        val_ds = CatDataset(val_x, val_y)
        trn_ds = CatDataset(trn_x, trn_y)
        val_dl = DataLoader(val_ds, batch_size, shuffle=True, num_workers=1)
        trn_dl = DataLoader(trn_ds, batch_size, shuffle=True, num_workers=1)
        super().__init__(path, val_dl, trn_dl)
    @classmethod
    def from_idxs(cls, path, val_idxs, x, y, batch_size):
        (val_x, trn_x), (val_y, trn_y) = split_by_idx(val_idxs, x, y)
        val_y = val_y[:, 0]
        trn_y = trn_y[:, 0]
        return cls(path, val_x, val_y, trn_x, trn_y, batch_size)

def get_learner(dataset, batch_size, Model, *args, **kwargs):
    x = pd.read_csv(f'data/{dataset}/{dataset}_py.dat', header=None)
    y = pd.read_csv(f'data/{dataset}/labels_py.dat', header=None)
    print(f"{len(x)} rows, {len(x.columns)} cols, {y[0].nunique()} unique labels in '{dataset}'\n")
    val_perc = 0.25
    val_idx = y.sample(int(len(y)*val_perc)).index
    data = CatModelData.from_idxs('results', val_idx, x.values, y.values, batch_size=batch_size)
    model = Model(len(x.columns), y[0].nunique(), *args, **kwargs)
    model = CatModel(to_gpu(model))
    learner = CatLearner(data, model, opt_fn=optim.SGD) #.half
    return learner

class RLinear(nn.Module):
    def __init__(self, input_dim, output_dim, dropout=0, norm=True):
        super(RLinear, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)
        self.bn = nn.BatchNorm1d(output_dim)
        self.dropout = dropout
        self.norm = norm
    def forward(self, x):
        x = self.fc(x)
        if self.norm: x = self.bn(x)
        x = F.relu(x)
        if self.dropout > 0: x = F.dropout(x, p=self.dropout, training=self.training)
        return x

class SLinear(nn.Module):
    def __init__(self, input_dim, output_dim, dropout=0, norm=True):
        super(SLinear, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)
        self.dropout = dropout
        self.norm = norm
        nn.init.normal(self.fc.weight, std=1/input_dim**.5)
        nn.init.constant(self.fc.bias, 0.)
    def forward(self, x):
        x = self.fc(x)
        if self.norm: x = F.selu(x)
        x = F.alpha_dropout(x, p=self.dropout, training=self.training)
        return x

class AbstractNet(nn.Module):
    def __init__(self, input_dim, output_dim, n_hidden_layers, hidden_dim, dropout=0.05):
        super(AbstractNet, self).__init__()
        self.input = self.Linear(input_dim, hidden_dim, dropout)
        self.n_hidden_layers = n_hidden_layers
        for i in range(self.n_hidden_layers): setattr(self, f'hidden_{i}', self.Linear(hidden_dim, hidden_dim, dropout))
        self.output = self.Linear(hidden_dim, output_dim, norm=False)
    def forward(self, x):
        x = self.input(x)
        for i in range(self.n_hidden_layers): x = getattr(self, f'hidden_{i}')(x)
        x = self.output(x)
        x = F.softmax(x)
        return x

class RNet(AbstractNet):
    Linear = RLinear

class SNet(AbstractNet):
    Linear = SLinear

class BatchLinear(nn.Module):
    def __init__(self, n_towers, in_features, out_features):
        super(BatchLinear, self).__init__()
        self.n_towers = n_towers
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(n_towers, in_features, out_features))
        self.bias = nn.Parameter(torch.Tensor(n_towers, out_features))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input):
        output = input.unsqueeze(2)
        output = torch.matmul(output, self.weight)
        output = output.squeeze(2)
        output += self.bias
        return output

    def extra_repr(self):
        return 'n_towers={} in_features={}, out_features={}'.format(
            self.n_towers, self.in_features, self.out_features
        )

class MLinear(nn.Module):
    def __init__(self, n_towers, tower_sz, dropout=0):
        super(MLinear, self).__init__()
        self.fc = BatchLinear(n_towers, tower_sz, tower_sz)
        self.dropout = dropout
        nn.init.normal(self.fc.weight, std=1/tower_sz**.5)
        nn.init.constant(self.fc.bias, 0.)
    def forward(self, x):
        x = self.fc(x)
        x = F.selu(x)
        x = F.alpha_dropout(x, p=self.dropout, training=self.training)
        return x

class MCombineLinear(nn.Module):
    def __init__(self, out_dim, n_towers, tower_sz, dropout=0):
        super(MCombineLinear, self).__init__()
        assert (out_dim / tower_sz).is_integer(), f'out_dim ({out_dim}) must be divisible by tower_sz ({tower_sz})'
        self.out_dim = out_dim
        self.n_towers = n_towers
        self.tower_sz = tower_sz
        self.fc = BatchLinear(tower_sz, n_towers, int(out_dim / tower_sz))
        self.dropout = dropout
        nn.init.normal(self.fc.weight, std=1/tower_sz**.5)
        nn.init.constant(self.fc.bias, 0.)
    def forward(self, x):
        x = x.view(-1, self.tower_sz, self.n_towers)
        x = self.fc(x)
        x = x.view(-1, self.out_dim)
        x = F.selu(x)
        x = F.alpha_dropout(x, p=self.dropout, training=self.training)
        return x

class MReshape(nn.Module):
    def __init__(self, in_dim, n_towers, tower_sz):
        super(MReshape, self).__init__()
        self.in_dim = in_dim
        self.n_towers = n_towers
        self.tower_sz = tower_sz
        self.n_repeat = int((n_towers * tower_sz) / in_dim)
        assert ((n_towers * tower_sz) / in_dim).is_integer(), \
            f'(n_towers [{n_towers}] * tower_sz [{tower_sz}]) [{n_towers * tower_sz}] must be divisible by in_dim ({in_dim})'
    def forward(self, x):
        x = x.view(x.shape[0], -1, self.tower_sz)
        x = x.repeat(1, self.n_repeat, 1)
        return x

class MBlockLinear(nn.Module):
    Combine = MCombineLinear
    def __init__(self, in_dim, out_dim, n_towers, tower_sz, n_layers, dropout=0):
        super(MBlockLinear, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.n_towers = n_towers
        self.tower_sz = tower_sz
        self.n_layers = n_layers
        self.input = MReshape(in_dim, n_towers, tower_sz)
        for i in range(self.n_layers): setattr(self, f'layer_{i}', MLinear(n_towers, tower_sz, dropout))
        self.combine = self.Combine(out_dim, n_towers, tower_sz, dropout=dropout)
        self.output = SLinear(out_dim, out_dim)
    def forward(self, x):
        x = self.input(x)
        for i in range(self.n_layers): x = getattr(self, f'layer_{i}')(x)
        x = self.combine(x)
        x = self.output(x)
        return x

class MNet(nn.Module):
    # blocks = [(in_dim, n_towers, tower_sz, n_layers), ...]
    def __init__(self, in_dim, out_dim, blocks, dropout=0.05):
        super(MNet, self).__init__()
        self.n_blocks = len(blocks)
        self.input = SLinear(in_dim, blocks[0][0], dropout)
        for i, block in enumerate(blocks):
            nxt_dim = blocks[i + 1][0] if i < len(blocks) - 1 else block[0]
            in_dim, n_towers, tower_sz, n_layers = block
            setattr(self, f'block_{i}', MBlockLinear(in_dim, nxt_dim, n_towers, tower_sz, n_layers, dropout))
        self.output = SLinear(blocks[len(blocks) - 1][0], out_dim, norm=False)
    def forward(self, x):
        x = self.input(x)
        for i in range(self.n_blocks): x = getattr(self, f'block_{i}')(x)
        x = self.output(x)
        return x

datasets = [
    'abalone',
    'annealing',
    'balance-scale',
    'bank',
    'blood',
    'car',
    'chess-krvkp',
    'conn-bench-vowel-deterding',
    'hill-valley',
    'image-segmentation',
    'led-display',
    'page-blocks',
    'spambase',
    'synthetic-control',
    'tic-tac-toe'
]

for ds in datasets:
    print('\nRNet')
    print('\nlr=0.1')
    rl = get_learner(ds, 256, RNet, 3, 256, 0.05)
    rl.fit(0.1, 4, cycle_len=1, cycle_mult=2, metrics=[accuracy])
    print('\nlr=0.01')
    rl = get_learner(ds, 256, RNet, 3, 256, 0.05)
    rl.fit(0.01, 4, cycle_len=1, cycle_mult=2, metrics=[accuracy])
    print('\nlr=0.001')
    rl = get_learner(ds, 256, RNet, 3, 256, 0.05)
    rl.fit(0.001, 4, cycle_len=1, cycle_mult=2, metrics=[accuracy])

    print('\nSNet')
    print('\nlr=0.1')
    sl = get_learner(ds, 256, SNet, 8, 256, 0.05)
    sl.fit(0.1, 4, cycle_len=1, cycle_mult=2, metrics=[accuracy])
    print('\nlr=0.01')
    sl = get_learner(ds, 256, SNet, 8, 256, 0.05)
    sl.fit(0.01, 4, cycle_len=1, cycle_mult=2, metrics=[accuracy])
    print('\nlr=0.001')
    sl = get_learner(ds, 256, SNet, 8, 256, 0.05)
    sl.fit(0.001, 4, cycle_len=1, cycle_mult=2, metrics=[accuracy])

    print('\nMNet - 1block')
    print('\nlr=0.1')
    ml = get_learner(ds, 256, MNet, [(256, 256, 16, 6)], 0.05)
    ml.fit(0.1, 4, cycle_len=1, cycle_mult=2, metrics=[accuracy])
    print('\nlr=0.01')
    ml = get_learner(ds, 256, MNet, [(256, 256, 16, 6)], 0.05)
    ml.fit(0.01, 4, cycle_len=1, cycle_mult=2, metrics=[accuracy])
    print('\nlr=0.001')
    ml = get_learner(ds, 256, MNet, [(256, 256, 16, 6)], 0.05)
    ml.fit(0.001, 4, cycle_len=1, cycle_mult=2, metrics=[accuracy])

    print('\nMNet - 3block')
    print('\nlr=0.1')
    ml = get_learner(ds, 256, MNet, [(256, 256, 16, 6), (128, 128, 16, 6), (64, 64, 16, 6)], 0.05)
    ml.fit(0.1, 4, cycle_len=1, cycle_mult=2, metrics=[accuracy])
    print('\nlr=0.01')
    ml = get_learner(ds, 256, MNet, [(256, 256, 16, 6), (128, 128, 16, 6), (64, 64, 16, 6)], 0.05)
    ml.fit(0.01, 4, cycle_len=1, cycle_mult=2, metrics=[accuracy])
    print('\nlr=0.001')
    ml = get_learner(ds, 256, MNet, [(256, 256, 16, 6), (128, 128, 16, 6), (64, 64, 16, 6)], 0.05)
    ml.fit(0.001, 4, cycle_len=1, cycle_mult=2, metrics=[accuracy])
