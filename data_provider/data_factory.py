from data_provider.data_loader import Dataset_PJM_OT, Dataset_NYIS_OT
        
from torch.utils.data import DataLoader

data_dict = {
    'PJM_OT': Dataset_PJM_OT,
    'NYIS_OT': Dataset_NYIS_OT,
}


def data_provider(args, flag):
    Data = data_dict[args.data]
    timeenc = 0 if args.embed != 'timeF' else 1

    if flag == 'test' or flag == 'val':
        shuffle_flag = False
        drop_last = True

        batch_size = 1  # bsz=1 for evaluation
        freq = args.freq
    else:
        shuffle_flag = True
        drop_last = True
        batch_size = args.batch_size  # bsz for train and valid
        freq = args.freq

    
        
    data_set = Data(
        root_path=args.root_path,
        data_path=args.data_path,
        flag=flag,
        pjm_sub_name=args.pjm_sub_name,
        size=[args.seq_len, args.label_len, args.pred_len],
        features=args.features,
        target=args.target,
        timeenc=timeenc,
        freq=freq,
        seasonal_patterns=args.seasonal_patterns,
        weather_train=args.weather_train,
        sub_train=args.sub_train,
        ext_feature=args.ext_feature,
        model_id=args.model_id,
    )
    print(flag, len(data_set))
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last)
    return data_set, data_loader
