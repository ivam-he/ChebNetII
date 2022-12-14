from models import ChebNetII,ChebNetII_V,ChebNetII_B,ChebNetII_BV
from data_utils import normalize

def parse_method(args, dataset, n, c, d, device):
    if args.method == 'ChebNetII':
        model = ChebNetII(d, c, args).to(device)
    elif args.method == 'ChebNetII_V':
        model = ChebNetII_V(d, c, args).to(device)
    elif args.method == 'ChebNetII_B':
        model = ChebNetII_B(d, c, args).to(device)
    elif args.method == 'ChebNetII_BV':
        model = ChebNetII_BV(d, c, args).to(device)
    else:
        raise ValueError('Invalid method')
    return model


def parser_add_main_args(parser):
    parser.add_argument('--seed',type=int, default=42)
    parser.add_argument('--dev',type=int, default=0)
    parser.add_argument('--dataset', type=str, default='fb100')
    parser.add_argument('--sub_dataset', type=str, default='')
    parser.add_argument('--hidden_channels', type=int, default=32)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--method', '-m', type=str, default='link')
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--weight_decay', type=float, default=1e-3)
    parser.add_argument('--display_step', type=int,default=1, help='how often to print')
    parser.add_argument('--hops', type=int, default=1,help='power of adjacency matrix for certain methods')
    parser.add_argument('--num_layers', type=int, default=2,help='number of layers for deep methods')
    parser.add_argument('--runs', type=int, default=5,help='number of distinct runs')
    parser.add_argument('--cached', action='store_true',help='set to use faster sgc')
    parser.add_argument('--gat_heads', type=int, default=8,help='attention heads for gat')
    parser.add_argument('--lp_alpha', type=float, default=.1,help='alpha for label prop')
    parser.add_argument('--gpr_alpha', type=float, default=.1,help='alpha for gprgnn')
    parser.add_argument('--gcn2_alpha', type=float, default=.1,help='alpha for gcn2')
    parser.add_argument('--theta', type=float, default=.5,help='theta for gcn2')
    parser.add_argument('--directed', action='store_true',help='set to not symmetrize adjacency')
    parser.add_argument('--jk_type', type=str, default='max', choices=['max', 'lstm', 'cat'],help='jumping knowledge type')
    parser.add_argument('--rocauc', action='store_true',help='set the eval function to rocauc')
    parser.add_argument('--num_mlp_layers', type=int, default=1,help='number of mlp layers in h2gcn')
    #优化器
    parser.add_argument('--adam', action='store_true', help='use adam instead of adamW')

    parser.add_argument('--rand_split', action='store_true', help='use random splits')
    parser.add_argument('--no_bn', action='store_true', help='do not use batchnorm')
    parser.add_argument('--sampling', action='store_true', help='use neighbor sampling')
    parser.add_argument('--inner_activation', action='store_true', help='Whether linkV3 uses inner activation')
    parser.add_argument('--inner_dropout', action='store_true', help='Whether linkV3 uses inner dropout')
    parser.add_argument("--SGD", action='store_true', help='Use SGD as optimizer')
    parser.add_argument('--link_init_layers_A', type=int, default=1)
    parser.add_argument('--link_init_layers_X', type=int, default=1)

    #ChebnetII
    parser.add_argument('--K', type=int, default=10)
    parser.add_argument('--dprate', type=float, default=0.5)
    parser.add_argument('--prop_lr', type=float, default=0.01, help='learning rate for propagation layer.')
    parser.add_argument('--prop_wd', type=float, default=0.0005, help='learning rate for propagation layer.')

    parser.add_argument('--early_stopping', type=int, default=200)

    parser.add_argument('--name',type=str,default="opt")
