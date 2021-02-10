import argparse

def set_transformer_config(run="Experiment1"):
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--iterations', type=int, default=300000)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--latent_hiddens', type=int, default=512)
    parser.add_argument('--model_checkpoint', type=str, default=None)
    parser.add_argument('--print_every', type=int, default=500)

    args = parser.parse_args()
    print(args)
    with open("{}.txt".format(run), 'w') as file:
        file.write(str(args))
    return args