from .dataset import build_mmad, build_charades

def build_dataset(image_set, args):
    if args.dataset == 'mmad':
        return build_mmad(image_set, args)
    else:
        raise ValueError(f'dataset {args.dataset} not implemented')
