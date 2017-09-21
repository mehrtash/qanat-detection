def get_cnn(model_id, model_params):
    if model_id == 'unet':
        from model.cnns.unet import Unet
        return EncoderDecoder(model_params)
