import argparse
from Model import Waveunet

## TRAIN PARAMETERS
parser = argparse.ArgumentParser()
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA (default: False)')
parser.add_argument('--num_workers', type=int, default=1,
                    help='Number of data loader worker threads (default: 1)')
parser.add_argument('--features', type=int, default=32,
                    help='# of feature channels per layer')
parser.add_argument('--log_dir', type=str, default='logs/waveunet',
                    help='Folder to write logs into')
parser.add_argument('--dataset_dir', type=str, default="/mnt/windaten/Datasets/MUSDB18HQ",
                    help='Dataset path')
parser.add_argument('--hdf_dir', type=str, default="hdf",
                    help='Dataset path')
parser.add_argument('--checkpoint_dir', type=str, default='checkpoints/waveunet',
                    help='Folder to write checkpoints into')
parser.add_argument('--load_model', type=str, default=None,
                    help='Reload a previously trained model (whole task model)')
parser.add_argument('--lr', type=float, default=1e-3,
                    help='initial learning rate (default: 5e-4)')
parser.add_argument('--min_lr', type=float, default=5e-5,
                    help='initial learning rate (default: 5e-4)')
parser.add_argument('--cycles', type=int, default=2,
                    help='Number of LR cycles per epoch')
parser.add_argument('--batch_size', type=int, default=4,
                    help="Batch size")
parser.add_argument('--levels', type=int, default=6,
                    help="Number DS/US blocks")
parser.add_argument('--depth', type=int, default=1,
                    help="Number of convs per block")
parser.add_argument('--sr', type=int, default=44100,
                    help="Sampling rate")
parser.add_argument('--channels', type=int, default=2,
                    help="Number of input audio channels")
parser.add_argument('--kernel_size', type=int, default=5,
                    help="Filter width of kernels. Has to be an odd number")
parser.add_argument('--output_size', type=float, default=2.0,
                    help="Output duration")
parser.add_argument('--strides', type=int, default=4,
                    help="Strides in Waveunet")
parser.add_argument('--patience', type=int, default=20,
                    help="Patience for early stopping on validation set")
parser.add_argument('--example_freq', type=int, default=200,
                    help="Write an audio summary into Tensorboard logs every X training iterations")
parser.add_argument('--loss', type=str, default="L1",
                    help="L1 or L2")
parser.add_argument('--conv_type', type=str, default="gn",
                    help="Type of convolution (normal, BN-normalised, GN-normalised): normal/bn/gn")
parser.add_argument('--res', type=str, default="fixed",
                    help="Resampling strategy: fixed sinc-based lowpass filtering or learned conv layer: fixed/learned")
parser.add_argument('--separate', type=int, default=1,
                    help="Train separate model for each source (1) or only one (0)")
parser.add_argument('--feature_growth', type=str, default="double",
                    help="How the features in each layer should grow, either (add) the initial number of features each time, or multiply by 2 (double)")

args = parser.parse_args()

INSTRUMENTS = ["bass", "drums", "other", "vocals"]
NUM_INSTRUMENTS = len(INSTRUMENTS)

# MODEL
num_features = [args.features*i for i in range(1, args.levels+1)] if args.feature_growth == "add" else \
               [args.features*2**i for i in range(0, args.levels)]
target_outputs = int(args.output_size * args.sr)
model = Waveunet(args.channels, num_features, args.channels, INSTRUMENTS, kernel_size=args.kernel_size,
                 target_output_size=target_outputs, depth=args.depth, strides=args.strides,
                 conv_type=args.conv_type, res=args.res, separate=args.separate)

print('model: ', model)
print('parameter count: ', str(sum(p.numel() for p in model.parameters())))